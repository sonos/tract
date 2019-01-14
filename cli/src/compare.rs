#![allow(unused_imports)]
use log::Level::Info;
use tract_core::model::OutletId;
use tract_core::plan::{SimplePlan, SimpleState};
use tract_core::{SharedTensor, Tensor, TensorFact};

use display_graph::DisplayOptions;
use errors::*;
use format::*;
use utils::*;
use Parameters;

/// Handles the `compare` subcommand.
#[cfg(not(feature = "conform"))]
pub fn handle(_params: Parameters, _: DisplayOptions) -> CliResult<()> {
    bail!("Comparison requires the `tensorflow` feature.")
}

#[cfg(feature = "conform")]
pub fn handle(params: Parameters, output_params: DisplayOptions) -> CliResult<()> {
    use colored::Colorize;
    use format::Row;

    let tract = params.tract_model;
    let tf = params.tf_model;

    // First generate random values for the inputs.
    let generated = ::tensor::make_inputs(&[tract.input_fact()?.clone()])?;

    // Execute the model on tensorflow first.
    info!("Running the model on tensorflow.");
    trace!("Inject inputs in tensorflow graph.");
    let pairs = tract
        .inputs()
        .iter()
        .map(|s| &*tract.node(s[0].node).name)
        .zip(generated.iter().cloned())
        .collect();

    trace!("Execute the model on tensorflow.");
    let eval_order = ::tract_core::model::eval_order(&tract)?;
    let nodes = tract.nodes();
    let wanted_outputs: Vec<&str> = eval_order
        .iter()
        .filter(|&n| !tract.inputs().unwrap().contains(&OutletId::new(*n, 0)))
        .map(|&n| &*nodes[n].name)
        .collect();
    let mut tf_outputs = tf.unwrap().run_get_many(pairs, wanted_outputs)?;

    // Execute the model step-by-step on tract.
    let plan = SimplePlan::new(&tract)?;
    let mut state = SimpleState::new(plan)?;
    for (ix, input) in generated.clone().into_iter().enumerate() {
        state.set_input(ix, input)?;
    }

    let mut display_graph =
        ::display_graph::DisplayGraph::from_model_and_options(tract.clone(), output_params)?
            .with_graph_def(&params.graph)?;

    let mut failing = vec![];

    for n in eval_order {
        let node = &tract.nodes()[n];

        if node.op_is::<tract_core::ops::source::Source>() {
            continue;
        }

        if node.op_is::<tract_tensorflow::ops::logic::Switch>() {
            // observing Switch is not allowed in TF. so we'll just use tract
            // value
            state.compute_one(n)?;
            display_graph.add_node_label(n, "Unchecked".to_string())?;
            continue;
        }

        debug!("Computing {} in tensorflow", node.name);
        let tf_output = match tf_outputs.remove(&*node.name) {
            Some(it) => it,
            None => continue,
        };

        debug!("Computing {} in tract", node.name);
        match state.compute_recursively(n) {
            Err(e) => {
                failing.push(n);
                display_graph.add_node_label(n, format!("{}: {}", "ERROR".red(), e))?;
            }

            _ => {
                // how many outputs are actually required ? ignore ancilliary
                // training wires from tensorflow op (fused batch norm)
                let wanted = node
                    .outputs
                    .iter()
                    .enumerate()
                    .position(|(ix, o)| {
                        o.successors.len() == 0 && !tract
                            .outputs()
                            .unwrap()
                            .contains(&OutletId::new(node.id, ix))
                    }).unwrap_or(node.outputs.len());
                let expected: Vec<TensorFact> = tf_output
                    .iter()
                    .take(wanted)
                    .map(|m| m.clone().into())
                    .collect();
                let tract_output: &[SharedTensor] = &*state.values[n].as_ref().unwrap();
                match check_outputs(&tract_output, &expected) {
                    Err(e) => {
                        failing.push(n);
                        let header = format!("Output {}", n).yellow().bold();
                        let mismatches = tract_output
                            .iter()
                            .enumerate()
                            .map(|(n, data)| {
                                let reason = if n >= tf_output.len() {
                                    "Too many outputs".into()
                                } else if tf_output[n].shape() != data.shape() {
                                    "Wrong shape".into()
                                } else if !tf_output[n]
                                    .close_enough(data, node.op.rounding_errors())
                                {
                                    "Too far away".into()
                                } else {
                                    format!("Other error: {}", e)
                                };

                                Row::Double(
                                    format!("{} {}", header, reason).red().to_string(),
                                    format!(
                                        "TF    {:?}\ntract {:?}",
                                        tf_output[n],
                                        data.as_tensor()
                                    ),
                                )
                            }).collect::<Vec<_>>();
                        let inputs = tract.nodes()[n]
                            .inputs
                            .iter()
                            .enumerate()
                            .map(|(ix, o)| {
                                let tensor = &state.values[o.node].as_ref().unwrap()[o.slot];
                                Row::Double(format!("Input #{}", ix), format!("{:?}", tensor))
                            }).collect::<Vec<_>>();
                        display_graph.add_node_section(n, inputs)?;
                        display_graph.add_node_section(n, mismatches)?;
                        display_graph.add_node_label(n, "MISM.".red().to_string())?;
                    }

                    _ => {
                        display_graph.add_node_label(n, "OK".green().to_string())?;
                    }
                }
            }
        };

        // Use the output from tensorflow to keep tract from drifting.
        trace!(
            "copy tensorflow output in state: for node {}, {:?}",
            node.id,
            tf_output
        );
        state.set_values(node.id, tf_output.into())?;
    }

    if failing.len() > 0 {
        for f in &failing {
            display_graph.render_node(&tract.nodes()[*f])?;
        }
        bail!("{} error(s).", failing.len())
    } else if log_enabled!(Info) {
        display_graph.render()?;
    } else {
        println!("{}", "Each node passed the comparison.".bold().green());
    }
    Ok(())
}
