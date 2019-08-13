#[allow(unused_imports)]
use std::fs;
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
            Model::Inference(m) => {
                handle_tensorflow_t(cumulative, resilient, m, tf, &params, output_params)
            }
            Model::Typed(m) => {
                handle_tensorflow_t(cumulative, resilient, m, tf, &params, output_params)
            }
            Model::Normalized(m) => {
                handle_tensorflow_t(cumulative, resilient, m, tf, &params, output_params)
            }
            Model::Pulsed(_, _) => panic!("Compare unsupported in pulse mode"),
        };
    }
}

#[cfg(feature = "conform")]
fn handle_tensorflow_t<TI: TensorInfo, O>(
    cumulative: bool,
    resilient: bool,
    tract: &ModelImpl<TI, O>,
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
        .input_outlets()?
        .iter()
        .map(|s| &*tract.node(s.node).name)
        .zip(generated.iter().cloned())
        .collect();

    trace!("Execute the model on tensorflow.");
    let eval_order = ::tract_core::model::eval_order(&tract)?;
    let nodes = tract.nodes();

    let mut wanted_outputs: Vec<&str> = eval_order
        .iter()
        .filter(|&n| !tract.input_outlets().unwrap().contains(&OutletId::new(*n, 0)))
        .map(|&n| &*nodes[n].name)
        .collect();

    for o in tract.output_outlets()? {
        let name = &*tract.nodes()[o.node].name;
        if !wanted_outputs.contains(&name) {
            wanted_outputs.push(name);
        }
    }

    let mut all_values:HashMap<String, CliResult<TVec<Tensor>>> = HashMap::new();
    if resilient {
        for name in wanted_outputs {
            all_values.insert(name.to_string(), tf.run(pairs.clone(), &name).map(|t| t.into()).map_err(|e| e.into()));
        }
    } else {
        tf.run_get_many(pairs, wanted_outputs)?.into_iter().for_each(|(k, v)| {
            all_values.insert(k.to_string(), Ok(v.into()));
        });
    };

    for (ix, input) in tract.input_outlets()?.iter().enumerate() {
        let name = &tract.node(input.node).name;
        all_values.insert(name.to_string(), Ok(tvec!(generated[ix].clone())));
    }
    compare(cumulative, tract, &all_values, params, output_params)
}

pub fn handle_npz(
    cumulative: bool,
    npz: &str,
    params: Parameters,
    output_params: DisplayOptions,
) -> CliResult<()> {
    let mut npz = ndarray_npy::NpzReader::new(std::fs::File::open(npz)?)?;
    let mut values = HashMap::new();
    for name in npz.names()? {
        if let Ok(value) = npz.by_name::<ndarray::OwnedRepr<f32>, ndarray::IxDyn>(&name) {
            let name = name.trim_end_matches(".npy");
            values.insert(name.to_string(), Ok(tvec!(value.into())));
        }
    }
    dispatch_model_no_pulse!(params.tract_model, |m| compare(cumulative, m, &values, &params, output_params))
}

#[cfg(feature = "onnx")]
pub fn handle_pbdir(
    cumulative: bool,
    pbdir: &str,
    params: Parameters,
    output_params: DisplayOptions,
) -> CliResult<()> {
    let mut values:HashMap<&str, TVec<Option<Tensor>>> = HashMap::new();
    let parsed_model = if let SomeGraphDef::Onnx(_, ref parsed) = params.graph {
        parsed
    } else {
        unreachable!("main must forcesGraphDef survival for pbdir to work")
    };
    for entry in fs::read_dir(pbdir)? {
        use std::convert::TryInto;
        let entry = entry?;
        let file = fs::File::open(entry.path())?;
        let tensor = tract_onnx::tensor::proto_from_reader(file)?;
        let outlet = parsed_model.outlets_by_name[tensor.get_name()];
        let node_name = &*parsed_model.model.nodes()[outlet.node].name;
        let output_tuple = values.entry(node_name).or_insert(tvec!());
        while output_tuple.len() <= outlet.slot {
            output_tuple.push(None)
        }
        output_tuple[outlet.slot] = Some(tensor.try_into()?);
    }
    let values = values.into_iter().filter(|(_,t)| t.iter().all(|t| t.is_some())).map(|(k,t)| (k.to_string(), Ok(t.into_iter().map(Option::unwrap).collect::<TVec<Tensor>>()))).collect();
    dispatch_model_no_pulse!(params.tract_model, |m| compare(cumulative, m, &values, &params, output_params))
}

pub fn compare<TI, O>(
    cumulative: bool,
    tract: &ModelImpl<TI, O>,
    all_values: &HashMap<String, CliResult<TVec<Tensor>>>,
    params: &Parameters,
    output_params: DisplayOptions,
) -> CliResult<()>
where
    TI: TensorInfo + Clone + for<'a> From<&'a Tensor>,
    O: AsRef<dyn Op> + AsMut<dyn Op> + Display + Debug + Clone,
    ModelImpl<TI, O>: Model,
{
    let eval_order = ::tract_core::model::eval_order(&tract)?;

    // Execute the model step-by-step on tract.
    let plan = SimplePlan::new(tract)?;
    let mut state = SimpleState::new(plan)?;

    for (ix, input) in tract.input_outlets()?.iter().enumerate() {
        let name = &tract.node(input.node).name;
        let value = &all_values[name].as_ref().unwrap()[0];
        state.set_input(ix, value.clone())?;
    }

    let mut display_graph =
        crate::display_graph::DisplayGraph::from_model_and_options(tract as &dyn Model, output_params.into())?
            .with_graph_def(&params.graph)?;

    let mut failing = vec![];

    for n in eval_order {
        let node = &tract.nodes()[n];

        let tf_output = match all_values.get(&*node.name) {
            Some(Ok(it)) => it,
            Some(Err(e)) => {
                display_graph.set_node_color(n, Yellow)?;
                display_graph.add_node_label(n, format!("{} {}", Yellow.paint("TF Error"), e))?;
                continue;
            },
            None => {
                display_graph.set_node_color(n, Yellow)?;
                display_graph.add_node_label(n, Yellow.paint("Not in reference").to_string())?;
                debug!("Skipping node (no reference) {}", node);
                continue;
            }
        };
        let expected: Vec<Option<Arc<Tensor>>> = tf_output.iter().map(|m| Some(m.clone().into_arc_tensor())).collect();
        let expected_outputs_section = expected.iter()
            .enumerate()
            .map(|(ix, o)| {
                format!("expected output #{}: {:?}", ix, o)
            })
            .collect::<Vec<_>>();
        display_graph.add_node_section(n, expected_outputs_section)?;

        if node.op_is::<tract_core::ops::Source>() {
            display_graph.set_node_color(n, Blue)?;
        } else if node.op().validation() == Validation::Random {
            display_graph.set_node_color(n, Blue)?;
            display_graph.add_node_label(n, Blue.paint("Random").to_string())?;
        } else if node.op_is::<tract_core::ops::unimpl::UnimplementedOp>() {
            display_graph.set_node_color(n, Red)?;
            display_graph.add_node_label(n, Red.paint("Unimplemented").to_string())?;
            failing.push(n);
        } else {
            debug!("Computing {} in tract", node);
            let error = state.compute_recursively(n).err();
            let inputs = tract.nodes()[n]
                .inputs
                .iter()
                .enumerate()
                .map(|(ix, o)| {
                    let tensor = &state.values[o.node].as_ref().unwrap()[o.slot];
                    format!("input value #{}: {:?}", ix, tensor)
                })
                .collect::<Vec<_>>();
            for (ix, f) in node.outputs.iter().enumerate() {
                if f.fact.to_tensor_fact().unify(&expected[ix].clone().unwrap().into()).is_err() {
                    failing.push(n);
                    display_graph.set_node_color(n, Red.bold())?;
                    display_graph.add_node_label(n, format!("{}: Could not reconcile infered fact for output #{} ({:?}) with reference.", Red.bold().paint("ERROR"), ix, f.fact))?;
                }
            }
            match error {
                Some(e) => {
                    failing.push(n);
                    display_graph.set_node_color(n, Red.bold())?;
                    display_graph.add_node_label(n, format!("{}: {}", Red.bold().paint("ERROR"), e))?;
                    display_graph.add_node_section(n, inputs)?;
                }
                _ => {
                    let tract_output: &[Arc<Tensor>] = &*state.values[n].as_ref().unwrap();
                    match check_outputs(&tract_output, &*expected) {
                        Err(e) => {
                            failing.push(n);
                            display_graph.add_node_section(n, inputs)?;
                            tract_output
                                .iter()
                                .enumerate()
                                .try_for_each(|(ix, data)| -> CliResult<()> {
                                    if ix >= tf_output.len() {
                                        display_graph.set_node_color(n, Yellow)?;
                                        display_graph.add_node_label(n, format!("Extra output (#{})", ix))?;
                                    } else if tf_output[ix].shape() != data.shape() {
                                        display_graph.set_node_color(n, Red.bold())?;
                                        display_graph.add_node_label(n, format!("Output {} has wrong shape. Expected {:?}, got {:?}", ix, tf_output[ix].shape(), data.shape()))?;
                                    } else if let Err(e) = tf_output[ix].close_enough(
                                        data,
                                        node.op().validation() == Validation::Rounding,
                                    ) {
                                        display_graph.set_node_color(n, Red.bold())?;
                                        let mut msg = vec!(Red.bold().paint(format!("Wrong value for output {}, {:?}", ix, e)).to_string());
                                        msg.push(format!("got     : {:?}", data));
                                        display_graph.add_node_section(n, msg)?;
                                    } else {
                                        display_graph.set_node_color(n, Red.bold())?;
                                        display_graph.add_node_label(n, format!("{:?}", e))?;
                                    };
                                    Ok(())
                                })?;
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
            display_graph.render_node(*f)?;
        }
    }

    if failing.len() > 0 {
        bail!("{} error(s).", failing.len())
    } else {
        println!("{}", Green.paint("Each node passed the comparison."));
    };
    Ok(())
}
