#![allow(unused_imports)]
use simplelog::Level::{Error, Info, Trace};
use tfdeploy::Tensor;

use Parameters;
use errors::*;
use format::*;
use utils::*;

/// Handles the `compare` subcommand.
#[cfg(not(feature = "tensorflow"))]
pub fn handle(_params: Parameters) -> Result<()> {
    bail!("Comparison requires the `tensorflow` feature.")
}

#[cfg(feature = "tensorflow")]
pub fn handle(params: Parameters) -> Result<()> {
    use colored::Colorize;
    use format::Row;

    let tfd = params.tfd_model;
    let mut tf = params.tf_model;

    let output = tfd.get_node_by_id(params.output)?;
    let mut state = tfd.state();

    let input = params.input
        .ok_or("Exactly one of <size> or <data> must be specified.")?;

    let shape = input.shape.iter().cloned().collect::<Option<Vec<_>>>()
        .ok_or("The compare command doesn't support streaming dimensions.")?;

    // First generate random values for the inputs.
    let mut generated = Vec::new();
    for i in params.inputs {
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
    info!("Using execution plan: {:?}", plan);

    if log_enabled!(Info) {
        print_header(format!("Detailed comparison for {}:", params.name), "white");
    }

    let mut failures = vec![];

    for n in plan {
        let node = tfd.get_node_by_id(n)?;

        if node.op_name == "Placeholder" {
            if log_enabled!(Info) {
                print_node(
                    node,
                    &params.graph,
                    Some(&state),
                    vec!["SKIP".yellow().to_string()],
                    vec![],
                );
            }

            continue;
        }

        let tf_output = tf_outputs.remove(&node.name.to_string()).expect(
            format!(
                "No node with name {} was computed by tensorflow.",
                node.name
            ).as_str(),
        );

        let (failure, status, mismatches) = match state.compute_one(n) {
            Err(e) => (
                true,
                "ERROR".red(),
                vec![Row::Simple(format!("Error message: {:?}", e))],
            ),

            _ => {
                let tfd_output = state.outputs[n].as_ref().unwrap();
                let views = tfd_output.iter().map(|m| &**m).collect::<Vec<&Tensor>>();

                match compare_outputs(&tf_output, &views) {
                    Err(_) => {
                        let mismatches = tfd_output
                            .iter()
                            .enumerate()
                            .map(|(n, data)| {
                                let header = format!("{} (TFD):", format!("Output {}", n).bold(),);

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

                                Row::Double(header, format!("{}\n{}", reason.to_string(), infos))
                            })
                            .collect::<Vec<_>>();

                        (true, "MISM.".red(), mismatches)
                    }

                    _ => (false, "OK".green(), vec![]),
                }
            }
        };

        let outputs = tf_output
            .iter()
            .enumerate()
            .map(|(n, data)| {
                Row::Double(
                    format!("{} (TF):", format!("Output {}", n).bold()),
                    data.partial_dump(false).unwrap(),
                )
            })
            .collect::<Vec<_>>();

        if log_enabled!(Info) {
            // Print the results for the node.
            print_node(
                node,
                &params.graph,
                Some(&state),
                vec![status.to_string()],
                vec![outputs.clone(), mismatches.clone()],
            );
        }

        if failure {
            failures.push((node, status, outputs, mismatches));
        }

        // Re-use the output from tensorflow to keep tfdeploy from drifting.
        state.set_outputs(node.id, tf_output)?;
    }

    if log_enabled!(Info) {
        println!();
    }

    if failures.len() > 0 {
        print_header(format!("There were {} errors:", failures.len()), "red");

        for (node, status, outputs, mismatches) in failures {
            print_node(
                node,
                &params.graph,
                Some(&state),
                vec![status.to_string()],
                vec![outputs, mismatches],
            );
        }
    } else {
        println!("{}", "Each node passed the comparison.".bold().green());
    }

    Ok(())
}

