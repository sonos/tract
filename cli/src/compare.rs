use std::fmt::{Debug, Display};
#[allow(unused_imports)]
use std::fs;

use ansi_term::Color::*;

use log::Level::Info;
use tract_core::internal::*;

use crate::display_graph::DisplayOptions;
use crate::*;

#[cfg(feature = "conform")]
pub fn handle_tensorflow(
    cumulative: bool,
    resilient: bool,
    params: &mut Parameters,
    output_params: DisplayOptions,
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
        output_params
    ))
}

pub fn handle_npz(
    cumulative: bool,
    npz: &str,
    params: &Parameters,
    output_params: DisplayOptions,
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
    output_params: DisplayOptions,
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

pub fn compare<TI, O>(
    cumulative: bool,
    tract: &ModelImpl<TI, O>,
    all_values: &HashMap<String, CliResult<Tensor>>,
    params: &Parameters,
    output_params: DisplayOptions,
) -> CliResult<()>
where
    TI: Fact + Clone + for<'a> From<&'a Tensor>,
    O: AsRef<dyn Op> + AsMut<dyn Op> + Display + Debug + Clone,
    ModelImpl<TI, O>: Model,
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

    let mut display_graph = crate::display_graph::DisplayGraph::from_model_and_options(
        tract as &dyn Model,
        output_params.into(),
    )?
    .with_graph_def(&params.graph)?;

    let mut failing = vec![];
    let mut ok = 0;

    for n in eval_order {
        let node = &tract.nodes()[n];
        let mut ok_node = true;

        if tract.input_outlets()?.iter().any(|o| o.node == n) {
            display_graph.set_node_color(n, Blue)?;
        } else if node.op().validation() == Validation::Random {
            display_graph.set_node_color(n, Blue)?;
            display_graph.add_node_label(&[n], Blue.paint("Random").to_string())?;
        } else if node.op_is::<tract_core::ops::unimpl::UnimplementedOp>() {
            display_graph.set_node_color(n, Red)?;
            display_graph.add_node_label(&[n], Red.paint("Unimplemented").to_string())?;
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
                display_graph.set_node_color(n, Red.bold())?;
                display_graph
                    .add_node_label(&[n], format!("{}: {}", Red.bold().paint("ERROR"), e))?;
            } else {
                for ix in 0..node.outputs.len() {
                    if let Some(ref_value) =
                        tract.outlet_label(OutletId::new(n, ix)).and_then(|lbl| all_values.get(lbl))
                    {
                        match ref_value {
                            Ok(t) => {
                                let found = &state.values[n].as_ref().unwrap()[ix];
                                if let Err(e) = found
                                    .close_enough(t, node.op().validation() == Validation::Rounding)
                                {
                                    failing.push(n);
                                    ok_node = false;
                                    display_graph.set_node_color(n, Red.bold())?;
                                    let mut msg = vec![Red
                                        .bold()
                                        .paint(format!("Wrong value for output {}, {}", ix, e))
                                        .to_string()];
                                    msg.push(format!("got     : {:?}", found));
                                    msg.push(format!("ref     : {:?}", t));
                                    msg.push(format!("check   : {:?}", node.op().validation()));
                                    display_graph.add_node_section(&[n], msg)?;
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
                                display_graph.set_node_color(n, Red.bold())?;
                                display_graph.add_node_label(
                                    &[n],
                                    format!("{}: {}", Red.bold().paint("ERROR"), e),
                                )?;
                            }
                        }
                    } else {
                        ok_node = false;
                        display_graph.set_node_color(n, Yellow.bold())?;
                        display_graph.add_node_label(
                            &[n],
                            Yellow.paint("Not matched against reference").to_string(),
                        )?;
                    }
                }
                display_graph.add_node_section(&[n], inputs)?;
                if ok_node {
                    display_graph.set_node_color(n, Green.bold())?;
                    ok += 1;
                }
            }
        }
    }

    if log_enabled!(Info) {
        display_graph.render()?;
    } else {
        for f in &failing {
            display_graph.render_node(*f)?;
        }
    }

    if failing.len() > 0 {
        bail!("{} error(s).", failing.len())
    } else {
        println!("{}", Green.paint(format!("{} node(s) passed the comparison.", ok)));
    };
    Ok(())
}
