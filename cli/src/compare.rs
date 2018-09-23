#![allow(unused_imports)]
use simplelog::Level::{Error, Info, Trace};
use tfdeploy::plan::{SimplePlan, SimpleState};
use tfdeploy::{ Tensor, TensorFact };

use errors::*;
use format::*;
use utils::*;
use {OutputParameters, Parameters};

/// Handles the `compare` subcommand.
#[cfg(not(feature = "tensorflow"))]
pub fn handle(_params: Parameters, _: OutputParameters) -> CliResult<()> {
    bail!("Comparison requires the `tensorflow` feature.")
}

#[cfg(feature = "tensorflow")]
pub fn handle(params: Parameters, output_params: OutputParameters) -> CliResult<()> {
    use colored::Colorize;
    use format::Row;

    let tfd = params.tfd_model;
    let mut tf = params.tf_model;

    // First generate random values for the inputs.
    let generated = ::tensor::make_inputs(&params.inputs)?;

    // Execute the model on tensorflow first.
    info!("Running the model on tensorflow.");
    let pairs = params.input_nodes.iter().map(|s| &**s).zip(generated.iter().cloned()).collect();
    let mut tf_outputs = tf.unwrap().run_get_all(pairs)?;

    // Execute the model step-by-step on tfdeploy.
    let plan = SimplePlan::new(&tfd, &params.input_nodes, &[&params.output_node])?;
    let mut state = plan.state()?;
    for (ix, input) in generated.clone().into_iter().enumerate() {
        state.set_input(ix, input)?;
    }
    let plan = ::tfdeploy::model::eval_order_for_nodes(
        &tfd.nodes(),
        &[tfd.node_by_name(&params.output_node)?.id],
    )?;
    debug!("Using execution plan: {:?}", plan);

    let nodes: Vec<_> = tfd.nodes().iter().map(|a| &*a).collect();
    let mut display_graph =
        ::display_graph::DisplayGraph::from_nodes(&*nodes)?.with_graph_def(&params.graph)?;

    let mut failures = 0;

    let hidden = !log_enabled!(Info);

    for n in plan {
        let node = &tfd.nodes()[n];
        let dn = &mut display_graph.nodes[n];

        if node.op_name == "Placeholder" {
            dn.hidden = hidden;
            continue;
        }

        let tf_output = tf_outputs.remove(&node.name.to_string()).expect(
            format!(
                "No node with name {} was computed by tensorflow.",
                node.name
            ).as_str(),
        );

        match state.compute_one(n) {
            Err(e) => {
                failures += 1;
                dn.more_lines.push(format!("Error message: {:?}", e));
                dn.label = Some("ERROR".red().to_string());
                dn.hidden = false;
            }

            _ => {
                let tfd_output:Vec<Tensor> = state.values[n].as_ref().unwrap().iter().map(|v| v.as_tensor().to_owned()).collect();
                let expected:Vec<TensorFact> = tf_output.iter().map(|m| m.clone().into()).collect();
                match check_outputs(&tfd_output, &expected) {
                    Err(_) => {
                        failures += 1;
                        let mismatches = tfd_output
                            .iter()
                            .enumerate()
                            .map(|(n, data)| {
                                let header = format!("Output {} TFD", n).yellow().bold();

                                let reason = if n >= tf_output.len() {
                                    "Too many outputs"
                                } else if tf_output[n].shape() != data.shape() {
                                    "Wrong shape"
                                } else if !tf_output[n]
                                    .close_enough(data, node.op.rounding_errors())
                                {
                                    "Too far away"
                                } else {
                                    "Other error"
                                };

                                let infos = data.partial_dump(false).unwrap();

                                format!("{} {} {}", header, reason.red().bold().to_string(), infos)
                            })
                            .collect::<Vec<_>>();
                        dn.more_lines.extend(mismatches);
                        dn.label = Some("MISM.".red().to_string());
                        dn.hidden = false;
                    }

                    _ => {
                        dn.hidden = hidden;
                        dn.label = Some("OK".green().to_string());
                    }
                }
            }
        };

        // Use the output from tensorflow to keep tfdeploy from drifting.
        for (ix, out) in tf_output.iter().enumerate() {
            if dn.outputs.len() > ix {
                let edge = &mut display_graph.edges[dn.outputs[ix]];
                edge.label = Some(format!("{:?}", out));
            }
        }
        state.set_values(node.id, tf_output.into())?;
    }

    if failures > 0 {
        print_header(format!("There were {} errors:", failures), "red");
        display_graph.render(&output_params)?;
    } else if log_enabled!(Info) {
        display_graph.render(&output_params)?;
    } else {
        println!("{}", "Each node passed the comparison.".bold().green());
    }

    Ok(())
}
