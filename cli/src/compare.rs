use ansi_term::Color::*;

use log::Level::Info;
use tract_core::internal::*;

use crate::display_graph::DisplayOptions;
use crate::utils::*;
use crate::*;

pub fn handle(mut params: Parameters, output_params: DisplayOptions) -> CliResult<()> {
    let tf = params.tf_model.take().unwrap();
    match &params.tract_model {
        SomeModel::Inference(m) => handle_t(m, tf, &params, output_params),
        SomeModel::Typed(m) => handle_t(m, tf, &params, output_params),
        SomeModel::Normalized(m) => handle_t(m, tf, &params, output_params),
        SomeModel::Pulsed(_, m) => handle_t(m, tf, &params, output_params),
    }
}

pub fn handle_t<TI: TensorInfo>(
    tract: &Model<TI>,
    mut tf: tract_tensorflow::conform::tf::Tensorflow,
    params: &Parameters,
    output_params: DisplayOptions,
) -> CliResult<()> {
    use crate::format::Row;

    // First generate random values for the inputs.
    let generated = crate::tensor::make_inputs(&[tract.input_fact(0)?.to_tensor_fact()])?;

    // Execute the model on tensorflow first.
    info!("Running the model on tensorflow.");
    trace!("Inject inputs in tensorflow graph.");
    let pairs = tract
        .input_outlets()
        .iter()
        .map(|s| &*tract.node(s[0].node).name)
        .zip(generated.iter().cloned())
        .collect();

    trace!("Execute the model on tensorflow.");
    let eval_order = ::tract_core::model::eval_order(&tract)?;
    let nodes = tract.nodes();
    let mut wanted_outputs: Vec<&str> = eval_order
        .iter()
        .filter(|&n| !tract.output_outlets().unwrap().contains(&OutletId::new(*n, 0)))
        .map(|&n| &*nodes[n].name)
        .collect();

    for o in tract.output_outlets()? {
        let name = &*tract.nodes()[o.node].name;
        if !wanted_outputs.contains(&name) {
            wanted_outputs.push(name);
        }
    }

    let mut tf_outputs = tf.run_get_many(pairs, wanted_outputs)?;

    // Execute the model step-by-step on tract.
    let plan = SimplePlan::new(tract)?;
    let mut state = SimpleState::new(plan)?;
    for (ix, input) in generated.clone().into_iter().enumerate() {
        state.set_input(ix, input)?;
    }

    let mut display_graph =
        crate::display_graph::DisplayGraph::from_model_and_options(tract.clone(), output_params)?
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

        debug!("Computing {} in tensorflow", node);
        let tf_output = match tf_outputs.remove(&*node.name) {
            Some(it) => it,
            None => continue,
        };

        debug!("Computing {} in tract", node);
        match state.compute_recursively(n) {
            Err(e) => {
                failing.push(n);
                display_graph.add_node_label(n, format!("{}: {}", Red.paint("ERROR"), e))?;
            }

            _ => {
                // how many outputs are actually required ? ignore ancilliary
                // training wires from tensorflow op (fused batch norm)
                let wanted = node
                    .outputs
                    .iter()
                    .enumerate()
                    .position(|(ix, o)| {
                        o.successors.len() == 0
                            && !tract.output_outlets().unwrap().contains(&OutletId::new(node.id, ix))
                    })
                    .unwrap_or(node.outputs.len());
                let expected: Vec<TensorFact> =
                    tf_output.iter().take(wanted).map(|m| m.clone().into()).collect();
                let tract_output: &[SharedTensor] = &*state.values[n].as_ref().unwrap();
                match check_outputs(&tract_output, &expected) {
                    Err(e) => {
                        failing.push(n);
                        let header = Yellow.bold().paint(format!("Output {}", n));
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
                                    Red.paint(format!("{} {}", header, reason)).to_string(),
                                    format!(
                                        "TF    {:?}\ntract {:?}",
                                        tf_output[n],
                                        data.as_tensor()
                                    ),
                                )
                            })
                            .collect::<Vec<_>>();
                        let inputs = tract.nodes()[n]
                            .inputs
                            .iter()
                            .enumerate()
                            .map(|(ix, o)| {
                                let tensor = &state.values[o.node].as_ref().unwrap()[o.slot];
                                Row::Double(format!("Input #{}", ix), format!("{:?}", tensor))
                            })
                            .collect::<Vec<_>>();
                        display_graph.add_node_section(n, inputs)?;
                        display_graph.add_node_section(n, mismatches)?;
                        display_graph.add_node_label(n, Red.paint("MISM.").to_string())?;
                    }

                    _ => {
                        display_graph.add_node_label(n, Green.paint("OK").to_string())?;
                    }
                }
            }
        };

        // Use the output from tensorflow to keep tract from drifting.
        trace!("copy tensorflow output in state: for node {}, {:?}", node.id, tf_output);
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
        println!("{}", Green.paint("Each node passed the comparison."));
    }
    Ok(())
}
