#![allow(unused_imports)]
use simplelog::Level::{Error, Info, Trace};
use tfdeploy::Tensor;

use { OutputParameters, Parameters };
use errors::*;
use format::*;
use utils::*;

/// Handles the `compare` subcommand.
#[cfg(not(feature = "tensorflow"))]
pub fn handle(_params: Parameters, _: OutputParameters) -> Result<()> {
    bail!("Comparison requires the `tensorflow` feature.")
}

#[cfg(feature = "tensorflow")]
pub fn handle(params: Parameters, output_params: OutputParameters) -> Result<()> {
    use colored::Colorize;
    use format::Row;

    let tfd = params.tfd_model;
    let mut tf = params.tf_model;

    let output = tfd.get_node_by_id(params.output_node_id)?;
    let mut state = tfd.state();

    let input = params.input
        .ok_or("Exactly one of <size> or <data> must be specified.")?;

    let shape = input.shape.iter().cloned().collect::<Option<Vec<_>>>()
        .ok_or("The compare command doesn't support streaming dimensions.")?;

    // First generate random values for the inputs.
    let mut generated = Vec::new();
    for i in params.input_node_ids {
        let data = if input.data.is_some() {
            input.data.as_ref().unwrap().clone()
        } else {
            random_tensor(shape.clone(), input.datatype)
        };

        generated.push((
            tfd.get_node_by_id(i)?.name.as_str(),
            data,
        ));
    }

    // Execute the model on tensorflow first.
    info!("Running the model on tensorflow.");
    let mut tf_outputs = tf.run_get_all(generated.clone())?;

    // Execute the model step-by-step on tfdeploy.
    state.set_values(generated)?;
    let plan = output.eval_order(&tfd)?;
    debug!("Using execution plan: {:?}", plan);

    let nodes:Vec<_> = tfd.nodes.iter().map(|a| &*a).collect();
    let mut display_graph = ::display_graph::DisplayGraph::from_nodes(&*nodes)?.with_graph_def(&params.graph)?;

    let mut failures = 0;

    let hidden = !log_enabled!(Info);

    for n in plan {
        let node = tfd.get_node_by_id(n)?;
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
            },

            _ => {
                let tfd_output = state.outputs[n].as_ref().unwrap();
                let views = tfd_output.iter().map(|m| &**m).collect::<Vec<&Tensor>>();
                match compare_outputs(&tf_output, &views) {
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
                                } else if !tf_output[n].close_enough(data) {
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
        state.set_outputs(node.id, tf_output)?;
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

