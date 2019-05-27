use std::fmt::{Debug, Display};

use ansi_term::Color::*;

use log::Level::Info;
use tract_core::internal::*;

use crate::display_graph::DisplayOptions;
use crate::utils::*;
use crate::*;

#[cfg(feature = "conform")]
pub fn handle_tensorflow(
    cumulative: bool,
    resilient: bool,
    mut params: Parameters,
    output_params: DisplayOptions,
) -> CliResult<()> {
    {
        let tf = params.tf_model.take().unwrap();
        return match &params.tract_model {
            SomeModel::Inference(m) => {
                handle_tensorflow_t(cumulative, resilient, m, tf, &params, output_params)
            }
            SomeModel::Typed(m) => {
                handle_tensorflow_t(cumulative, resilient, m, tf, &params, output_params)
            }
            SomeModel::Normalized(m) => {
                handle_tensorflow_t(cumulative, resilient, m, tf, &params, output_params)
            }
            SomeModel::Pulsed(_, _) => panic!("Compare unsupported in pulse mode"),
        };
    }
}

#[cfg(feature = "conform")]
fn handle_tensorflow_t<TI: TensorInfo, O>(
    cumulative: bool,
    resilient: bool,
    tract: &Model<TI, O>,
    mut tf: tract_tensorflow::conform::tf::Tensorflow,
    params: &Parameters,
    output_params: DisplayOptions,
) -> CliResult<()>
where
    TI: TensorInfo + Clone + for<'a> From<&'a Tensor>,
    O: AsRef<Op> + AsMut<Op> + Display + Debug + Clone,
{
    // First generate random values for the inputs.
    let input_facts = tract
        .input_outlets()?
        .iter()
        .map(|&i| Ok(tract.outlet_fact(i)?.to_tensor_fact()))
        .collect::<TractResult<Vec<_>>>()?;
    let generated = crate::tensor::make_inputs(&*input_facts)?;

    // Execute the model on tensorflow first.
    info!("Running the model on tensorflow.");
    trace!("Inject inputs in tensorflow graph.");
    let pairs: Vec<_> = tract
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

    let mut all_values: HashMap<String, TVec<Tensor>> = if resilient {
        /*
        wanted_outputs
            .iter()
            .filter_map(|name| {
                tf.run(pairs.clone(), name).ok().map(|tensors| (name.to_string(), tensors.into()))
            })
            .collect::<HashMap<String, TVec<Tensor>>>();
            */
        wanted_outputs
            .iter()
            .filter_map(|name| {
                tf.run(pairs.clone(), name).ok().map(|tensors| (name.to_string(), tensors.into()))
            })
            .collect()
    } else {
        tf.run_get_many(pairs, wanted_outputs)?
            .into_iter()
            .map(|(k, v)| (k.to_string(), v.into()))
            .collect()
    };

    for (ix, input) in tract.input_outlets()?.iter().enumerate() {
        let name = &tract.node(input.node).name;
        all_values.insert(name.to_string(), tvec!(generated[ix].clone()));
    }
    compare(cumulative, tract, &all_values, params, output_params)
}

pub fn handle_npz(
    cumulative: bool,
    npz: &str,
    params: Parameters,
    output_params: DisplayOptions,
) -> CliResult<()> {
    {
        return match &params.tract_model {
            SomeModel::Inference(m) => handle_npz_t(cumulative, m, npz, &params, output_params),
            SomeModel::Typed(m) => handle_npz_t(cumulative, m, npz, &params, output_params),
            SomeModel::Normalized(m) => handle_npz_t(cumulative, m, npz, &params, output_params),
            SomeModel::Pulsed(_, _) => panic!("Compare unsupported in pulse mode"),
        };
    }
}

pub fn handle_npz_t<TI, O>(
    cumulative: bool,
    tract: &Model<TI, O>,
    npz: &str,
    params: &Parameters,
    output_params: DisplayOptions,
) -> CliResult<()>
where
    TI: TensorInfo + Clone + for<'a> From<&'a Tensor>,
    O: AsRef<Op> + AsMut<Op> + Display + Debug + Clone,
{
    let mut npz = ndarray_npy::NpzReader::new(std::fs::File::open(npz)?)?;
    let mut values = HashMap::new();
    for name in npz.names()? {
        if let Ok(value) = npz.by_name::<ndarray::OwnedRepr<f32>, ndarray::IxDyn>(&name) {
            let name = name.trim_end_matches(".npy");
            values.insert(name.to_string(), tvec!(value.into()));
        }
    }
    compare(cumulative, tract, &values, params, output_params)
}

pub fn compare<TI, O>(
    cumulative: bool,
    tract: &Model<TI, O>,
    all_values: &HashMap<String, TVec<Tensor>>,
    params: &Parameters,
    output_params: DisplayOptions,
) -> CliResult<()>
where
    TI: TensorInfo + Clone + for<'a> From<&'a Tensor>,
    O: AsRef<Op> + AsMut<Op> + Display + Debug + Clone,
{
    let eval_order = ::tract_core::model::eval_order(&tract)?;

    // Execute the model step-by-step on tract.
    let plan = SimplePlan::new(tract)?;
    let mut state = SimpleState::new(plan)?;

    for (ix, input) in tract.input_outlets()?.iter().enumerate() {
        let name = &tract.node(input.node).name;
        let value = &all_values[name][0];
        state.set_input(ix, value.clone())?;
    }

    let mut display_graph =
        crate::display_graph::DisplayGraph::from_model_and_options(tract.clone(), output_params)?
            .with_graph_def(&params.graph)?;

    let mut failing = vec![];

    for n in eval_order {
        let node = &tract.nodes()[n];

        let tf_output = match all_values.get(&*node.name) {
            Some(it) => it,
            None => {
                display_graph.set_node_color(n, Yellow)?;
                display_graph.add_node_label(n, Yellow.paint("Not in reference").to_string())?;
                debug!("Skipping node (no reference) {}", node);
                continue;
            }
        };

        if node.op_is::<tract_core::ops::Source>() {
            display_graph.add_node_label(n, Blue.paint("Source").to_string())?;
        } else if node.op().validation() == Validation::Random {
            display_graph.add_node_label(n, Blue.paint("Random").to_string())?;
        } else if node.op_is::<tract_core::ops::unimpl::UnimplementedOp>() {
            display_graph.add_node_label(n, Red.paint("Unimplemented").to_string())?;
            failing.push(n);
        } else {
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
                                && !tract
                                    .output_outlets()
                                    .unwrap()
                                    .contains(&OutletId::new(node.id, ix))
                        })
                        .unwrap_or(node.outputs.len());
                    let expected: Vec<TensorFact> =
                        tf_output.iter().take(wanted).map(|m| m.clone().into()).collect();
                    let tract_output: &[Arc<Tensor>] = &*state.values[n].as_ref().unwrap();
                    match check_outputs(&tract_output, &expected) {
                        Err(_) => {
                            failing.push(n);
                            let mismatches = tract_output
                                .iter()
                                .enumerate()
                                .map(|(n, data)| {
                                    let (color, reason) = if n >= tf_output.len() {
                                        (Yellow, "Unexpected output".into())
                                    } else if tf_output[n].shape() != data.shape() {
                                        (Red, "Wrong shape".into())
                                    } else if !tf_output[n].close_enough(
                                        data,
                                        node.op().validation() == Validation::Rounding,
                                    ) {
                                        let msg = format!(
                                            "expected: {:?}\ngot     : {:?}",
                                            tf_output[n], data
                                        );
                                        (Red, msg)
                                    } else {
                                        (Green, "ok".into())
                                    };

                                    color.paint(format!("Output {}: {}", n, reason)).to_string()
                                })
                                .collect::<Vec<_>>();
                            let inputs = tract.nodes()[n]
                                .inputs
                                .iter()
                                .enumerate()
                                .map(|(ix, o)| {
                                    let tensor = &state.values[o.node].as_ref().unwrap()[o.slot];
                                    format!("Input #{}: {:?}", ix, tensor)
                                })
                                .collect::<Vec<_>>();
                            display_graph.add_node_section(n, inputs)?;
                            display_graph.add_node_section(n, mismatches)?;
                            display_graph.set_node_color(n, Red)?;
                        }
                        _ => {
                            display_graph.set_node_color(n, Green)?;
                        }
                    }
                }
            }
        }

        if !cumulative {
            // Use the output from tensorflow to keep tract from drifting.
            trace!("copy tensorflow output in state: for node {}, {:?}", node.id, tf_output);
            state.set_values(node.id, tf_output.clone().into())?;
        }
    }

    if log_enabled!(Info) {
        display_graph.render()?;
    } else {
        for f in &failing {
            display_graph.render_node(&tract.nodes()[*f])?;
        }
    }

    if failing.len() > 0 {
        bail!("{} error(s).", failing.len())
    } else {
        println!("{}", Green.paint("Each node passed the comparison."));
    };
    Ok(())
}
