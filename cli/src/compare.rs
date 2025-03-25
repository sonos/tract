#![allow(dead_code)]
use std::fmt::{Debug, Display};
#[allow(unused_imports)]
use std::fs;

use nu_ansi_term::Color::*;

use log::Level::Info;
use tract_core::internal::*;

use crate::dump::annotate_with_graph_def;
use crate::*;
use tract_libcli::display_params::DisplayParams;
use tract_libcli::tensor::RunParams;

pub fn handle(
    params: &mut Parameters,
    _matches: &clap::ArgMatches,
    sub_matches: &clap::ArgMatches,
    output_params: DisplayParams,
) -> TractResult<()> {
    let run_params = crate::tensor::run_params_from_subcommand(params, sub_matches)?;

    let cumulative = sub_matches.is_present("cumulative");
    let resilent = sub_matches.is_present("resilient");
    if sub_matches.value_of("stage").is_some() {
        // --with is by pipeline and put in params
        return handle_reference_stage(cumulative, params, &output_params, &run_params);
    } else if let Some(npz) = sub_matches.value_of("npz") {
        return handle_npz(cumulative, npz, params, &output_params, &run_params);
    } else if sub_matches.is_present("twice") {
        return handle_twice(cumulative, params, &output_params, &run_params);
    }
    if let Some(pbdir) = sub_matches.value_of("pbdir") {
        return handle_pbdir(cumulative, pbdir, params, &output_params, &run_params);
    }
    if sub_matches.is_present("tf") {
        return handle_tensorflow(cumulative, resilent, params, &output_params, &run_params);
    }
    bail!("No comparison target found")
}

#[cfg(not(feature = "conform"))]
pub fn handle_tensorflow(
    _cumulative: bool,
    _resilient: bool,
    _params: &mut Parameters,
    _output_params: &DisplayParams,
    _run_params: &RunParams,
) -> TractResult<()> {
    bail!("`tf` feature is required for this to work");
}

#[cfg(feature = "conform")]
pub fn handle_tensorflow(
    cumulative: bool,
    resilient: bool,
    params: &mut Parameters,
    output_params: &DisplayParams,
    run_params: &RunParams,
) -> TractResult<()> {
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

    let mut all_values: HashMap<String, Vec<TractResult<TValue>>> = HashMap::new();
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
    run_params,
    ))
}

pub fn handle_npz(
    cumulative: bool,
    npz: &str,
    params: &Parameters,
    output_params: &DisplayParams,
    run_params: &RunParams,
) -> TractResult<()> {
    use tract_libcli::tensor::for_npz;
    let mut npz = ndarray_npy::NpzReader::new(fs_err::File::open(npz)?)?;
    let mut values: HashMap<String, Vec<TractResult<TValue>>> = HashMap::new();
    let multiturn = npz.names()?.iter().any(|n| n.starts_with("turn_0/"));
    for name in npz.names()? {
        if let Ok(value) = for_npz(&mut npz, &name) {
            if multiturn {
                let name = name
                    .split('/')
                    .nth(1)
                    .with_context(|| {
                        format!(
                            "npy filenames should be turn_XX/... in multiturn mode, got `{name}'"
                        )
                    })?
                    .trim_end_matches(".npy");
                values.entry(name.to_string()).or_default().push(Ok(value.into_tvalue()));
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
        params,
        output_params,
        run_params,
    ))
}

#[cfg(not(feature = "onnx"))]
pub fn handle_pbdir(
    _cumulative: bool,
    _pbdir: &str,
    _params: &Parameters,
    _output_params: &DisplayParams,
    _run_params: &RunParams,
) -> TractResult<()> {
    bail!("`onnx` feature is required for this to work");
}

#[cfg(feature = "onnx")]
pub fn handle_pbdir(
    cumulative: bool,
    pbdir: &str,
    params: &Parameters,
    output_params: &DisplayParams,
    run_params: &RunParams,
) -> TractResult<()> {
    use tract_onnx::data_resolver::FopenDataResolver;
    use tract_onnx::tensor::load_tensor;

    let mut values: HashMap<String, Vec<TractResult<TValue>>> = HashMap::new();
    for entry in fs::read_dir(pbdir)? {
        let entry = entry?;
        let file = fs::File::open(entry.path())?;
        let tensor = tract_onnx::tensor::proto_from_reader(file)?;
        let name = tensor.name.to_string();
        let value: Tensor = load_tensor(&FopenDataResolver, &tensor, None)?;
        values.insert(name, vec![Ok(value.into_tvalue())]);
    }
    dispatch_model_no_pulse!(params.tract_model, |m| compare(
        cumulative,
        m,
        &values,
        params,
        output_params,
        run_params,
    ))
}

pub fn handle_twice(
    cumulative: bool,
    params: &Parameters,
    output_params: &DisplayParams,
    run_params: &RunParams,
) -> TractResult<()> {
    let reference_model =
        params.tract_model.downcast_ref::<TypedModel>().context("Only work with a typed model")?;
    handle_with_model(cumulative, params, output_params, reference_model, run_params)
}

pub fn handle_reference_stage(
    cumulative: bool,
    params: &Parameters,
    output_params: &DisplayParams,
    run_params: &RunParams,
) -> TractResult<()> {
    info!("Computing results for reference stage");
    let reference_model =
        params.reference_model.as_ref().context("No reference model. need --with ?")?;
    let reference_model = reference_model
        .downcast_ref::<TypedModel>()
        .context("Only work with a typed reference model")?;
    handle_with_model(cumulative, params, output_params, reference_model, run_params)
}

pub fn handle_with_model(
    cumulative: bool,
    params: &Parameters,
    output_params: &DisplayParams,
    reference_model: &TypedModel,
    run_params: &RunParams,
) -> TractResult<()> {
    let mut values: HashMap<String, Vec<TractResult<TValue>>> = HashMap::new();

    let plan = SimplePlan::new(reference_model)?;
    let mut state = SimpleState::new(plan)?;
    for inputs in tract_libcli::tensor::retrieve_or_make_inputs(reference_model, run_params)? {
        state.run_plan_with_eval(inputs, |session, state, node, input| -> TractResult<_> {
            let result = tract_core::plan::eval(session, state, node, input)?;
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
        output_params,
        run_params,
    ))
}

pub fn compare<F, O>(
    cumulative: bool,
    tract: &Graph<F, O>,
    all_values: &HashMap<String, Vec<TractResult<TValue>>>,
    params: &Parameters,
    output_params: &DisplayParams,
    run_params: &RunParams,
) -> TractResult<()>
where
    F: Fact + Clone + for<'a> From<&'a Arc<Tensor>> + Hash,
    O: AsRef<dyn Op> + AsMut<dyn Op> + Display + Debug + Clone,
    Graph<F, O>: Model,
{
    info!("Obtained reference data, starting test run");
    // Execute the model step-by-step on tract.
    let plan = SimplePlan::new(tract)?;
    let mut state = SimpleState::new(plan)?;

    let mut annotations = Annotations::from_model(tract as &dyn Model)?;
    annotate_with_graph_def(&mut annotations, tract, &params.graph)?;

    let mut failing = std::collections::HashSet::new();
    let mut unchecked = std::collections::HashSet::new();
    let mut ok = 0;
    fn canonic(s: &str) -> String {
        s.replace(['.', '-'], "_")
    }
    let all_values: HashMap<String, &Vec<TractResult<TValue>>> =
        all_values.iter().map(|(k, v)| (canonic(k), v)).collect();
    let model_inputs = tract_libcli::tensor::retrieve_or_make_inputs(tract, run_params)?;
    for (turn, inputs) in model_inputs.into_iter().enumerate() {
        state.run_plan_with_eval(
            inputs,
            |session_state, state, node, input| -> TractResult<TVec<TValue>> {
                let tags = annotations.node_mut(node.id.into());
                let reference: Option<TVec<TValue>> = (0..node.outputs.len())
                    .map(|ix| {
                        let get_value = |label: &str| {
                            all_values
                                .get(&canonic(label))
                                .and_then(|v| v.get(turn))
                                .and_then(|r| r.as_ref().ok())
                                .cloned()
                        };
                        tract
                            .outlet_label((node.id, ix).into())
                            .and_then(get_value)
                            .or_else(|| get_value(&node.name).filter(|_| ix == 0))
                            .map(|t| {
                                let needed_fact = crate::utils::clarify_typed_fact(node.outputs[ix].fact.to_typed_fact().unwrap());
                                let needed_type = needed_fact.datum_type;
                                let needed_shape = needed_fact.shape.as_concrete();
                                let t = if needed_shape.is_some_and(|it| it != t.shape()) {

                                    t.into_tensor().into_shape(needed_shape.unwrap())
                                        .expect("Comparing incompatible shapes")
                                        .into_tvalue()
                                } else {
                                    t.clone()
                                };

                                if needed_type == t.datum_type() {
                                    t
                                } else if needed_type.unquantized() == t.datum_type().unquantized()
                                {
                                    let mut t = t.into_tensor();
                                    unsafe { t.set_datum_type(needed_type) };
                                    t.into_tvalue()
                                } else if needed_type.is_float() && t.datum_type().is_float() {
                                    t.cast_to_dt(needed_type).unwrap().into_owned().into_tvalue()
                                } else {
                                    panic!("Comparing incompatible types")
                                }
                            })
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
                    log::info!("Running {node}");
                    let obtained = tract_core::plan::eval(session_state, state, node, input);
                    match obtained {
                        Err(e) => {
                            error = Some(format!("{}: {}", Red.bold().paint("ERROR"), e));
                            dbg!(&e);
                        }
                        Ok(obtained) => {
                            let clear_obtained = crate::utils::clarify_tvalues(&obtained)?;
                            tested = Some(obtained);
                            if let Some(reference) = &reference {
                                if reference.len() != clear_obtained.len() {
                                    error = Some("Output number mismatch".to_string());
                                } else {
                                    for ix in 0..node.outputs.len() {

                                        if let Err(e) =
                                            clear_obtained[ix].close_enough(&reference[ix], true)
                                        {
                                            error = Some("Mismatch value".to_string());
                                            let mut msg = vec![Red
                                                .bold()
                                                .paint(format!(
                                                    "At turn {turn}, wrong value for output {ix}, {e}"
                                                ))
                                                .to_string()];
                                            msg.push(format!("got     : {:?}", clear_obtained[ix]));
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
                result.with_context(|| {
                    format!("Failure to compute and no reference value for {node}")
                })
            },
        )?;
    }

    for node in tract.nodes() {
        let color: nu_ansi_term::Style = if failing.contains(&node.id) {
            Red.into()
        } else if unchecked.contains(&node.id) {
            White.into()
        } else {
            Green.bold()
        };
        annotations.node_mut(node.id.into()).style = Some(color);
    }

    if log_enabled!(Info) {
        tract_libcli::terminal::render(tract, &annotations, output_params)?;
    } else {
        for f in failing.iter().sorted() {
            tract_libcli::terminal::render_node(tract, *f, &annotations, output_params)?;
        }
    }

    if failing.len() > 0 {
        bail!("{} error(s).", failing.len())
    } else {
        println!("{}", Green.paint(format!("{ok} node(s) passed the comparison.")));
    };
    Ok(())
}
