#![allow(dead_code)]
use std::fmt::{Debug, Display};
#[allow(unused_imports)]
use std::fs;

use ansi_term::Color::*;

use log::Level::Info;
use tract_core::internal::*;

use crate::display_params::DisplayParams;
use crate::*;

#[cfg(feature = "conform")]
pub fn handle_tensorflow(
    cumulative: bool,
    resilient: bool,
    params: &mut Parameters,
    output_params: DisplayParams,
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

    let mut all_values: HashMap<String, CliResult<Tensor>> = HashMap::new();
    if resilient {
        for name in wanted_outputs {
            all_values.insert(
                name.to_string(),
                tf.run(pairs.clone(), &name).map(|t| t[0].clone().into()).map_err(|e| e.into()),
            );
        }
    } else {
        tf.run_get_many(pairs, wanted_outputs)?.into_iter().for_each(|(k, v)| {
            all_values.insert(k.to_string(), Ok(v[0].clone().into()));
        });
    };

    for (ix, input) in tract.input_outlets().iter().enumerate() {
        let name = tract.node_name(input.node);
        all_values.insert(name.to_string(), Ok(generated[ix].clone()));
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
    let mut npz = ndarray_npy::NpzReader::new(std::fs::File::open(npz)?)?;
    let mut values = HashMap::new();
    for name in npz.names()? {
        if let Ok(value) = tensor::for_npz(&mut npz, &name) {
            let name = name.trim_end_matches(".npy");
            values.insert(name.to_string(), Ok(value.into()));
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

#[cfg(feature = "onnx")]
pub fn handle_pbdir(
    cumulative: bool,
    pbdir: &str,
    params: &Parameters,
    output_params: &DisplayParams,
) -> CliResult<()> {
    let mut values: HashMap<String, CliResult<Tensor>> = HashMap::new();
    for entry in fs::read_dir(pbdir)? {
        use std::convert::TryInto;
        let entry = entry?;
        let file = fs::File::open(entry.path())?;
        let tensor = tract_onnx::tensor::proto_from_reader(file)?;
        values.insert(tensor.name.to_string(), Ok(tensor.try_into()?));
    }
    dispatch_model_no_pulse!(params.tract_model, |m| compare(
        cumulative,
        m,
        &values,
        &params,
        output_params
    ))
}

pub fn handle_reference_stage(
    cumulative: bool,
    params: &Parameters,
    output_params: &DisplayParams,
) -> CliResult<()> {
    let reference_model =
        params.reference_model.as_ref().ok_or("No reference model. need --with ?")?;
    let reference_model = reference_model
        .downcast_ref::<TypedModel>()
        .ok_or("Only work with a typed reference model")?;
    let mut values: HashMap<String, CliResult<Tensor>> = HashMap::new();

    let plan = SimplePlan::new(reference_model)?;
    let mut state = SimpleState::new(plan)?;
    let input_facts = reference_model
        .input_outlets()?
        .iter()
        .map(|&i| reference_model.outlet_fact(i))
        .collect::<TractResult<Vec<_>>>()?;
    let generated = crate::tensor::make_inputs(&*input_facts)?;
    state.run_plan_with_eval(
        generated.clone(),
        |session, state, node, input| -> TractResult<_> {
            let result: TVec<Arc<Tensor>> = tract_core::plan::eval(session, state, node, input)?;
            values.insert(node.name.clone(), Ok(result[0].as_ref().clone()));
            Ok(result)
        },
    )?;
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
    all_values: &HashMap<String, CliResult<Tensor>>,
    params: &Parameters,
    output_params: &DisplayParams,
) -> CliResult<()>
where
    F: Fact + Clone + for<'a> From<&'a Tensor> + Hash,
    O: AsRef<dyn Op> + AsMut<dyn Op> + Display + Debug + Clone + Hash,
    Graph<F, O>: Model,
{
    let eval_order = ::tract_core::model::eval_order(&tract)?;

    // Execute the model step-by-step on tract.
    let plan = SimplePlan::new(tract)?;
    let mut state = SimpleState::new(plan)?;

    for (ix, input) in tract.input_outlets()?.iter().enumerate() {
        let name = &tract.node(input.node).name;
        let value = all_values[name].as_ref().unwrap();
        state.set_input(ix, value.clone())?;
    }

    let mut annotations = crate::annotations::Annotations::from_model(tract as &dyn Model)?
        .with_graph_def(tract, &params.graph)?;

    let mut failing = vec![];
    let mut ok = 0;

    for n in eval_order {
        let node = &tract.nodes()[n];
        let mut ok_node = true;

        let mut tags = annotations.node_mut(n.into());

        if tract.input_outlets()?.iter().any(|o| o.node == n) {
            tags.style = Some(Blue.into());
        } else if node.op().validation() == Validation::Random {
            tags.style = Some(Blue.into());
            tags.labels.push(Blue.paint("Random").to_string());
        } else if node.op_is::<tract_core::ops::unimpl::UnimplementedOp>() {
            tags.style = Some(Red.into());
            tags.labels.push(Red.paint("Unimplemented").to_string());
            failing.push(n);
        } else {
            debug!("Computing {} in tract", node);
            let error = state.compute_recursively(n).err();
            let inputs = tract.nodes()[n]
                .inputs
                .iter()
                .enumerate()
                .map(|(ix, o)| {
                    let tensor = &state.values[o.node].as_ref().and_then(|v| v.get(o.slot));
                    format!("input value #{}: {:?}", ix, tensor)
                })
                .collect::<Vec<_>>();
            if let Some(e) = error {
                failing.push(n);
                tags.style = Some(Red.into());
                tags.labels.push(format!("{}: {}", Red.bold().paint("ERROR"), e));
            } else {
                for ix in 0..node.outputs.len() {
                    let label = tract.outlet_label((n, ix).into()).unwrap_or(&node.name);
                    if let Some(ref_value) = all_values.get(label) {
                        match ref_value {
                            Ok(t) => {
                                let found = &state.values[n].as_ref().unwrap()[ix];
                                if let Err(e) = found
                                    .close_enough(t, node.op().validation() == Validation::Rounding)
                                {
                                    failing.push(n);
                                    ok_node = false;
                                    tags.style = Some(Red.bold());
                                    let mut msg = vec![Red
                                        .bold()
                                        .paint(format!("Wrong value for output {}, {}", ix, e))
                                        .to_string()];
                                    msg.push(format!("got     : {:?}", found));
                                    msg.push(format!("ref     : {:?}", t));
                                    msg.push(format!("check   : {:?}", node.op().validation()));
                                    tags.sections.push(msg);
                                }
                                if !cumulative {
                                    // Use the output from reference to keep tract from drifting.
                                    state.values[node.id].as_mut().unwrap()[ix] =
                                        t.to_owned().into_arc_tensor();
                                }
                            }
                            Err(e) => {
                                failing.push(n);
                                ok_node = false;
                                tags.style = Some(Red.bold());
                                tags.labels.push(format!("{}: {}", Red.bold().paint("ERROR"), e));
                            }
                        }
                    } else {
                        ok_node = false;
                        tags.style = Some(Yellow.bold());
                        tags.labels.push(Yellow.paint("Not matched against reference").to_string());
                    }
                }
                tags.sections.push(inputs);
                if ok_node {
                    tags.style = Some(Green.bold());
                    ok += 1;
                }
            }
        }
    }

    if log_enabled!(Info) {
        terminal::render(tract, &annotations, &output_params)?;
    } else {
        for f in &failing {
            terminal::render_node(tract, *f, &annotations, &output_params)?;
        }
    }

    if failing.len() > 0 {
        error_chain::bail!("{} error(s).", failing.len())
    } else {
        println!("{}", Green.paint(format!("{} node(s) passed the comparison.", ok)));
    };
    Ok(())
}
