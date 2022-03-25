#![allow(dead_code)]
use std::fmt::{Debug, Display};
#[allow(unused_imports)]
use std::fs;

use ansi_term::Color::*;

use log::Level::Info;
use tract_core::internal::*;

use crate::display_params::DisplayParams;
use crate::*;

pub fn handle(
    params: &mut Parameters,
    options: &clap::ArgMatches,
    output_params: DisplayParams,
) -> CliResult<()> {
    let cumulative = options.is_present("cumulative");
    let resilent = options.is_present("resilient");
    if options.value_of("stage").is_some() {
        // --with is by pipeline and put in params
        return handle_reference_stage(cumulative, params, &output_params);
    } else if let Some(npz) = options.value_of("npz") {
        return handle_npz(cumulative, npz, params, &output_params);
    } else if options.is_present("twice") {
        return handle_twice(cumulative, params, &output_params);
    }
    if let Some(pbdir) = options.value_of("pbdir") {
        return handle_pbdir(cumulative, pbdir, params, &output_params);
    }
    if options.is_present("tf") {
        return handle_tensorflow(cumulative, resilent, params, &output_params);
    }
    bail!("No comparison target found")
}

#[cfg(not(feature = "conform"))]
pub fn handle_tensorflow(
    _cumulative: bool,
    _resilient: bool,
    _params: &mut Parameters,
    _output_params: &DisplayParams,
) -> CliResult<()> {
    bail!("`tf` feature is required for this to work");
}

#[cfg(feature = "conform")]
pub fn handle_tensorflow(
    cumulative: bool,
    resilient: bool,
    params: &mut Parameters,
    output_params: &DisplayParams,
) -> CliResult<()> {
    let tract = &params.tract_model;
    let mut tf = params.tf_model.take().unwrap();
    // First generate random values for the inputs.
    let input_facts = tract
        .input_outlets()
        .iter()
        .map(|&i| tract.outlet_typedfact(i))
        .collect::<TractResult<Vec<_>>>()?;
    let generated = crate::tensor::make_inputs(&*input_facts)?;

    // Execute the model on tensorflow first.
    info!("Running the model on tensorflow.");
    trace!("Inject inputs in tensorflow graph.");
    let pairs: Vec<_> = tract
        .input_outlets()
        .iter()
        .map(|s| &*tract.node_name(s.node))
        .zip(generated.iter().cloned())
        .collect();

    trace!("Execute the model on tensorflow.");
    let eval_order = tract.eval_order()?;

    let mut wanted_outputs: Vec<&str> = eval_order
        .iter()
        .filter(|&n| !tract.input_outlets().contains(&OutletId::new(*n, 0)))
        .map(|&n| tract.node_name(n))
        .collect();

    for o in tract.output_outlets() {
        let name = &*tract.node_name(o.node);
        if !wanted_outputs.contains(&name) {
            wanted_outputs.push(name);
        }
    }

    let mut all_values: HashMap<String, Vec<CliResult<Arc<Tensor>>>> = HashMap::new();
    if resilient {
        for name in wanted_outputs {
            all_values.insert(
                name.to_string(),
                vec![tf
                    .run(pairs.clone(), &name)
                    .map(|t| Arc::new(t[0].clone().into()))
                    .map_err(|e| e.into())],
            );
        }
    } else {
        tf.run_get_many(pairs, wanted_outputs)?.into_iter().for_each(|(k, v)| {
            all_values.insert(k.to_string(), vec![Ok(v[0].clone().into())]);
        });
    };

    for (ix, input) in tract.input_outlets().iter().enumerate() {
        let name = tract.node_name(input.node);
        all_values.insert(name.to_string(), vec![Ok(generated[ix].clone().into_arc_tensor())]);
    }
    dispatch_model_no_pulse!(params.tract_model, |m| compare(
        cumulative,
        m,
        &all_values,
        &params,
        &output_params
    ))
}

pub fn handle_npz(
    cumulative: bool,
    npz: &str,
    params: &Parameters,
    output_params: &DisplayParams,
) -> CliResult<()> {
    use tensor::for_npz;
    let mut npz = ndarray_npy::NpzReader::new(std::fs::File::open(npz)?)?;
    let mut values: HashMap<String, Vec<CliResult<Arc<Tensor>>>> = HashMap::new();
    for name in npz.names()? {
        if let Ok(value) = for_npz(&mut npz, &name) {
            if params.multiturn {
                let name = name
                    .split("/")
                    .nth(1)
                    .with_context(|| {
                        format!(
                            "npy filenames should be turn_XX/... in multiturn mode, got `{}'",
                            name
                        )
                    })?
                    .trim_end_matches(".npy");
                values.entry(name.to_string()).or_default().push(Ok(value.into_arc_tensor()));
            } else {
                let name = name.trim_end_matches(".npy");
                values.insert(name.to_string(), vec![Ok(value.into())]);
            }
        }
    }
    dispatch_model_no_pulse!(params.tract_model, |m| compare(
        cumulative,
        m,
        &values,
        &params,
        output_params
    ))
}

#[cfg(not(feature = "onnx"))]
pub fn handle_pbdir(
    cumulative: bool,
    pbdir: &str,
    params: &Parameters,
    output_params: &DisplayParams,
) -> CliResult<()> {
    bail!("`onnx` feature is required for this to work");
}

#[cfg(feature = "onnx")]
pub fn handle_pbdir(
    cumulative: bool,
    pbdir: &str,
    params: &Parameters,
    output_params: &DisplayParams,
) -> CliResult<()> {
    let mut values: HashMap<String, Vec<CliResult<Arc<Tensor>>>> = HashMap::new();
    for entry in fs::read_dir(pbdir)? {
        use std::convert::TryInto;
        let entry = entry?;
        let file = fs::File::open(entry.path())?;
        let tensor = tract_onnx::tensor::proto_from_reader(file)?;
        values.insert(tensor.name.to_string(), vec![Ok(Arc::new(tensor.try_into()?))]);
    }
    dispatch_model_no_pulse!(params.tract_model, |m| compare(
        cumulative,
        m,
        &values,
        &params,
        output_params
    ))
}

pub fn handle_twice(
    cumulative: bool,
    params: &Parameters,
    output_params: &DisplayParams,
) -> CliResult<()> {
    let reference_model =
        params.tract_model.downcast_ref::<TypedModel>().context("Only work with a typed model")?;
    handle_with_model(cumulative, params, output_params, &reference_model)
}

pub fn handle_reference_stage(
    cumulative: bool,
    params: &Parameters,
    output_params: &DisplayParams,
) -> CliResult<()> {
    let reference_model =
        params.reference_model.as_ref().context("No reference model. need --with ?")?;
    let reference_model = reference_model
        .downcast_ref::<TypedModel>()
        .context("Only work with a typed reference model")?;
    handle_with_model(cumulative, params, output_params, &reference_model)
}

pub fn handle_with_model(
    cumulative: bool,
    params: &Parameters,
    output_params: &DisplayParams,
    reference_model: &TypedModel,
) -> CliResult<()> {
    let mut values: HashMap<String, Vec<CliResult<Arc<Tensor>>>> = HashMap::new();

    let plan = SimplePlan::new(reference_model)?;
    let mut state = SimpleState::new(plan)?;
    for inputs in crate::tensor::retrieve_or_make_inputs(reference_model, params)? {
        state.run_plan_with_eval(inputs, |session, state, node, input| -> TractResult<_> {
            let result: TVec<Arc<Tensor>> = tract_core::plan::eval(session, state, node, input)?;
            if node.outputs.len() == 1 {
                values.entry(node.name.clone()).or_default().push(Ok(result[0].clone()));
            }
            for (output_slot, v) in result.iter().enumerate() {
                if let Some(tag) = reference_model.outlet_label((node.id, output_slot).into()) {
                    if tag != node.name {
                        values.entry(tag.to_string()).or_default().push(Ok(v.clone()));
                    }
                }
            }
            Ok(result)
        })?;
    }
    dispatch_model_no_pulse!(params.tract_model, |m| compare(
        cumulative,
        m,
        &values,
        params,
        output_params
    ))
}

pub fn compare<F, O>(
    cumulative: bool,
    tract: &Graph<F, O>,
    all_values: &HashMap<String, Vec<CliResult<Arc<Tensor>>>>,
    params: &Parameters,
    output_params: &DisplayParams,
) -> CliResult<()>
where
    F: Fact + Clone + for<'a> From<&'a Tensor> + Hash,
    O: AsRef<dyn Op> + AsMut<dyn Op> + Display + Debug + Clone + Hash,
    Graph<F, O>: Model,
{
    // Execute the model step-by-step on tract.
    let plan = SimplePlan::new(tract)?;
    let mut state = SimpleState::new(plan)?;

    let mut annotations = crate::annotations::Annotations::from_model(tract as &dyn Model)?
        .with_graph_def(tract, &params.graph)?;

    let mut failing = std::collections::HashSet::new();
    let mut unchecked = std::collections::HashSet::new();
    let mut ok = 0;
    for (turn, inputs) in tensor::retrieve_or_make_inputs(tract, params)?.into_iter().enumerate() {
        state.run_plan_with_eval(
            inputs,
            |session_state, state, node, input| -> TractResult<TVec<Arc<Tensor>>> {
                let mut tags = annotations.node_mut(node.id.into());
                let reference: Option<TVec<Arc<Tensor>>> = (0..node.outputs.len())
                    .map(|ix| {
                        let get_value = |label: &str| {
                            all_values
                                .get(label)
                                .and_then(|v| v.get(turn))
                                .and_then(|r| r.as_ref().ok())
                                .cloned()
                        };
                        tract
                            .outlet_label((node.id, ix).into())
                            .and_then(get_value)
                            .or_else(|| get_value(&node.name))
                    })
                    .collect();
                let mut tested = None;
                let mut error = None;
                if tract.input_outlets()?.iter().any(|o| o.node == node.id) {
                    tags.style = Some(Blue.into());
                } else if node.op().validation() == Validation::Random {
                    tags.style = Some(Blue.into());
                    tags.labels.push(Blue.paint("Random").to_string());
                } else if node.op_is::<tract_core::ops::unimpl::UnimplementedOp>() {
                    tags.style = Some(Red.into());
                    tags.labels.push(Red.paint("Unimplemented").to_string());
                    failing.insert(node.id);
                } else {
                    let obtained = tract_core::plan::eval(session_state, state, node, input);

                    match obtained {
                        Err(e) => {
                            error = Some(format!("{}: {}", Red.bold().paint("ERROR"), e));
                        }
                        Ok(obtained) => {
                            tested = Some(obtained.clone());
                            if let Some(reference) = &reference {
                                if reference.len() != obtained.len() {
                                    error = Some("Output number mismatch".to_string());
                                } else {
                                    for ix in 0..node.outputs.len() {
                                        if let Err(e) =
                                            obtained[ix].close_enough(&reference[ix], true)
                                        {
                                            error = Some("Mismatch value".to_string());
                                            let mut msg = vec![Red
                                                .bold()
                                                .paint(format!(
                                                    "At turn {}, wrong value for output {}, {}",
                                                    turn, ix, e
                                                ))
                                                .to_string()];
                                            msg.push(format!("got     : {:?}", obtained[ix]));
                                            msg.push(format!("ref     : {:?}", reference[ix]));
                                            tags.sections.push(msg);
                                        } else {
                                            debug!(
                                                "At turn {}, matching value for {:?}",
                                                turn,
                                                OutletId::new(node.id, ix)
                                            )
                                        }
                                    }
                                }
                            } else {
                                unchecked.insert(node.id);
                            }
                        }
                    }
                }
                if let Some(e) = error {
                    annotations.node_mut(node.id.into()).style = Some(Red.into());
                    annotations.node_mut(node.id.into()).labels.push(e);
                    failing.insert(node.id);
                } else {
                    ok += 1;
                }
                let result = if cumulative { tested.or(reference) } else { reference.or(tested) };
                Ok(result.with_context(|| {
                    format!("Failure to compute and no reference value for {}", node)
                })?)
            },
        )?;
    }

    for node in tract.nodes() {
        let color: ansi_term::Style = if failing.contains(&node.id) {
            Red.into()
        } else if unchecked.contains(&node.id) {
            White.into()
        } else {
            Green.bold()
        };
        annotations.node_mut(node.id.into()).style = Some(color);
    }

    if log_enabled!(Info) {
        terminal::render(tract, &annotations, &output_params)?;
    } else {
        for f in failing.iter().sorted() {
            terminal::render_node(tract, *f, &annotations, &output_params)?;
        }
    }

    if failing.len() > 0 {
        bail!("{} error(s).", failing.len())
    } else {
        println!("{}", Green.paint(format!("{} node(s) passed the comparison.", ok)));
    };
    Ok(())
}
