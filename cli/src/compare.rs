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
use tract_libcli::tensor::{RunParams, get_or_make_inputs};

pub fn handle(
    params: &mut Parameters,
    _matches: &clap::ArgMatches,
    sub_matches: &clap::ArgMatches,
    output_params: DisplayParams,
) -> TractResult<()> {
    let run_params = crate::tensor::run_params_from_subcommand(params, sub_matches)?;

    if sub_matches.is_present("stream") {
        return handle_stream(params, &output_params, &run_params);
    }

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
                vec![
                    tf.run(pairs.clone(), &name)
                        .map(|t| Arc::new(t[0].clone().into()))
                        .map_err(|e| e.into()),
                ],
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
    dispatch_model_no_pulse!(&params.tract_model, |m| compare(
        cumulative,
        &m,
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
    dispatch_model_no_pulse!(&params.tract_model, |m| compare(
        cumulative,
        &m,
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
    let reference_model = &params.tract_model;
    let reference_model: Arc<TypedModel> = Arc::downcast::<TypedModel>(reference_model.clone())
        .map_err(|_| anyhow!("Only works with a typed reference model"))?;
    handle_with_model(cumulative, params, output_params, &reference_model, run_params)
}

pub fn handle_reference_stage(
    cumulative: bool,
    params: &Parameters,
    output_params: &DisplayParams,
    run_params: &RunParams,
) -> TractResult<()> {
    info!("Computing results for reference stage");
    let reference_model: &Arc<dyn Model> =
        params.reference_model.as_ref().context("Missing reference model")?;
    let reference_model = Arc::downcast::<TypedModel>(reference_model.clone())
        .map_err(|_| anyhow!("Only works with a typed reference model"))?;
    handle_with_model(cumulative, params, output_params, &reference_model, run_params)
}

pub fn handle_with_model(
    cumulative: bool,
    params: &Parameters,
    output_params: &DisplayParams,
    reference_model: &Arc<TypedModel>,
    run_params: &RunParams,
) -> TractResult<()> {
    let mut values: HashMap<String, Vec<TractResult<TValue>>> = HashMap::new();
    let plan = reference_model.clone().into_runnable()?;
    let mut state = plan.spawn()?;
    let inputs = get_or_make_inputs(&(reference_model.clone() as _), run_params)?;
    state.init_states(&inputs.state_initializers)?;
    for input in inputs.sources {
        state.run_plan_with_eval(input, |session, state, node, input| -> TractResult<_> {
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
    dispatch_model_no_pulse!(&params.tract_model, |m| compare(
        cumulative,
        &m,
        &values,
        params,
        output_params,
        run_params,
    ))
}

pub fn compare<F, O>(
    cumulative: bool,
    tract: &Arc<Graph<F, O>>,
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
    let plan = Arc::clone(tract).into_runnable()?;
    let mut state = plan.spawn()?;

    let mut annotations = Annotations::from_model(tract.as_ref() as &dyn Model)?;
    annotate_with_graph_def(&mut annotations, tract.as_ref(), &params.graph)?;

    let mut failing = std::collections::HashSet::new();
    let mut unchecked = std::collections::HashSet::new();
    let mut ok = 0;
    fn canonic(s: &str) -> String {
        s.replace(['.', '-'], "_")
    }
    let all_values: HashMap<String, &Vec<TractResult<TValue>>> =
        all_values.iter().map(|(k, v)| (canonic(k), v)).collect();
    let inputs = get_or_make_inputs(&(tract.clone() as _), run_params)?;
    state.init_states(&inputs.state_initializers)?;
    for (turn, inputs) in inputs.sources.into_iter().enumerate() {
        state.run_plan_with_eval(
            inputs,
            |session_state, state, node, input| -> TractResult<TVec<TValue>> {
                let mut returning = tract_core::plan::eval(session_state, state, node, input)?;
                let tags = annotations.node_mut(node.id.into());
                let mut comparison_error = None;
                for slot in 0..returning.len() {
                    let get_value = |label: &str| {
                        all_values
                            .get(&canonic(label))
                            .and_then(|v| v.get(turn))
                            .and_then(|r| r.as_ref().ok())
                            .cloned()
                    };
                    let reference: Option<TValue> = tract
                        .outlet_label((node.id, slot).into())
                        .and_then(get_value)
                        .or_else(|| get_value(&node.name).filter(|_| slot == 0));

                    let Some(reference) = reference else {
                        tags.style = Some(Yellow.into());
                        unchecked.insert(node.id);
                        continue;
                    };

                    let mut reference = reference.into_tensor();
                    let clarified_fact = crate::utils::clarify_typed_fact(
                        node.outputs[slot].fact.to_typed_fact().unwrap(),
                    );
                    let needed_type = clarified_fact.datum_type;
                    let needed_shape = clarified_fact.shape.eval_to_usize(&session_state.resolved_symbols)?;

                    if **needed_shape != *reference.shape() {
                        let Ok(reshaped) = reference.clone().into_shape(&needed_shape) else {
                            comparison_error = Some(format!("Incompatible shape on output {slot} reference is {reference:?}, model expects {:?}.", needed_shape));
                            tags.style = Some(Red.into());
                            continue;
                        };
                        reference = reshaped;
                    }

                    if needed_type != reference.datum_type() {
                        if needed_type.unquantized() == reference.datum_type().unquantized() {
                            unsafe { reference.set_datum_type(needed_type) };
                        } else if needed_type.is_float() && reference.datum_type().is_float() {
                             reference = reference.cast_to_dt(needed_type)?.into_owned();
                        } else {
                            comparison_error = Some(format!("Incompatible type on output {slot} reference is {reference:?}, model expects {:?}.", needed_type));
                            tags.style = Some(Red.into());
                            continue;
                        }
                    }

                    let computed = crate::utils::clarify_tvalue(&returning[slot])?;

                    if let Err(e) =
                        computed.close_enough(&reference, params.assertions.approximation)
                    {
                        comparison_error = Some("Mismatch value".to_string());
                        let mut msg = vec![Red
                            .bold()
                            .paint(format!(
                                "At turn {turn}, wrong value for output {slot}, {e}"
                            ))
                            .to_string()];
                        msg.push(format!("got     : {:?}", computed));
                        msg.push(format!("ref     : {:?}", reference));
                        tags.sections.push(msg);
                    } else {
                        debug!(
                            "At turn {}, matching value for {:?}",
                            turn,
                            OutletId::new(node.id, slot)
                        )
                    }

                    if !cumulative && !returning[slot].datum_type().is_opaque() {
                        returning[slot] = reference.into_tvalue();
                    }
                }
                if let Some(e) = comparison_error {
                    annotations.node_mut(node.id.into()).style = Some(Red.into());
                    annotations.node_mut(node.id.into()).labels.push(e);
                    failing.insert(node.id);
                } else {
                    ok += 1;
                }
                Ok(returning)
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
        tract_libcli::terminal::render(tract.as_ref(), &annotations, output_params)?;
    } else {
        for f in failing.iter().sorted() {
            tract_libcli::terminal::render_node(tract.as_ref(), *f, &annotations, output_params)?;
        }
    }

    if failing.len() > 0 {
        bail!("{} error(s).", failing.len())
    } else {
        println!("{}", Green.paint(format!("{ok} node(s) passed the comparison.")));
    };
    Ok(())
}

// This does not reuse compare() above because the execution model is fundamentally different:
// compare() runs the model node-by-node via run_plan_with_eval and compares full outputs against
// reference values. Stream mode needs per-node stateful pulse-by-pulse execution with
// delay-aware valid-region slicing — each node has its own delay/axis/pulse and gets its own
// SimpleState. The input generation (NaN-padded pulse chunks with resolved_symbols on the last
// chunk) also doesn't fit get_or_make_inputs.
#[cfg(not(feature = "pulse"))]
pub fn handle_stream(
    _params: &Parameters,
    _output_params: &DisplayParams,
    _run_params: &RunParams,
) -> TractResult<()> {
    bail!("`pulse` feature is required for --stream to work");
}

#[cfg(feature = "pulse")]
pub fn handle_stream(
    params: &Parameters,
    output_params: &DisplayParams,
    _run_params: &RunParams,
) -> TractResult<()> {
    use tract_core::ndarray::{ArrayD, Axis};
    use tract_core::plan::SimpleState;
    use tract_pulse::internal::*;

    let reference: &TypedModel = params
        .reference_model
        .as_ref()
        .context("Reference (pre-pulse) model not available")?
        .downcast_ref::<TypedModel>()
        .context("Reference model is not a TypedModel")?;

    let model: Arc<TypedModel> =
        params.typed_model().context("Post-pulse TypedModel not available")?;

    // Find the streaming symbol and pulse size from the models
    let ref_input_fact = reference.input_fact(0)?;
    let model_input_fact = model.input_fact(0)?;

    // Find the streaming axis from the post-pulse model's properties
    let input_axis = model
        .properties
        .get("pulse.input_axes")
        .context("Expect pulse.input_axes property")?
        .cast_to::<i64>()?
        .as_slice::<i64>()?[0] as usize;

    // Find the stream symbol from the reference model's input fact
    let stream_symbol = ref_input_fact.shape[input_axis]
        .symbols()
        .into_iter()
        .next()
        .context("Could not find streaming symbol in reference model input")?;

    // Get pulse size from the post-pulse model's input fact on the streaming axis
    let input_pulse = model_input_fact.shape[input_axis]
        .to_usize()
        .context("Pulse dimension should be concrete")?;

    // Recreate PulsedModel for per-node metadata (delay, axis, pulse per node)
    let pulsed = PulsedModel::new(reference, stream_symbol.clone(), &input_pulse.to_dim())?;

    let mut annotations = Annotations::from_model(model.as_ref() as &dyn Model)?;
    annotate_with_graph_def(&mut annotations, model.as_ref(), &params.graph)?;

    let eval_order = reference.eval_order()?;
    let mut failing = std::collections::HashSet::new();
    let mut ok = 0usize;
    let mut skipped = 0usize;

    for &decl_node in eval_order.iter() {
        let pulsed_node = match pulsed.node_by_name(&*reference.node(decl_node).name) {
            Ok(node) => node.id,
            _ => {
                skipped += 1;
                continue;
            }
        };
        let node = match model.node_by_name(&*reference.node(decl_node).name) {
            Ok(node) => node.id,
            _ => {
                skipped += 1;
                continue;
            }
        };

        for output_slot in 0..reference.node(decl_node).outputs.len() {
            debug!("checking node: {} output: {}", reference.node(decl_node).name, output_slot);
            let decl_outlet = OutletId::new(decl_node, output_slot);
            let pulsed_outlet = OutletId::new(pulsed_node, output_slot);
            let outlet = OutletId::new(node, output_slot);

            let pulsed_output_fact = pulsed.outlet_fact(pulsed_outlet)?;

            let stream = match pulsed_output_fact.stream.as_ref() {
                Some(s) => s,
                None => continue,
            };
            let output_pulse =
                pulsed_output_fact.pulse().context("No pulse in pulsed fact")?.to_usize()?;
            let output_axis = stream.axis;
            let delay = stream.delay;

            let stream_dim = delay + 3 * input_pulse + input_pulse / 2;

            let concrete_sym_values = SymbolValues::default().with(&stream_symbol, stream_dim as _);
            let concrete_shape: Vec<usize> =
                ref_input_fact.shape.eval_to_usize(&concrete_sym_values)?.into_owned().into_vec();
            let concrete_input_fact = ref_input_fact.datum_type.fact(&*concrete_shape);
            let fixed_input =
                tract_libcli::tensor::tensor_for_fact(&concrete_input_fact, None, None)?;

            let fixed_result = reference
                .clone()
                .with_output_outlets(&[decl_outlet])?
                .concretize_dims(&concrete_sym_values)?
                .into_runnable()?
                .run(tvec!(fixed_input.clone().into_tvalue()))?
                .remove(output_slot);
            let fixed_output_len = fixed_result.shape()[output_axis];

            let plan = model.as_ref().clone().with_output_outlets(&[outlet])?.into_runnable()?;
            let plan = Arc::new(plan);
            let mut state = SimpleState::new(&plan)?;

            let mut node_ok = true;
            for i in 0.. {
                let input_shape = model_input_fact
                    .shape
                    .iter()
                    .map(|d| d.to_usize())
                    .collect::<TractResult<TVec<_>>>()?;
                let mut pulsed_input = ArrayD::from_elem(&*input_shape, f32::NAN);
                let offset = i * input_pulse;
                if offset < stream_dim {
                    let count = input_pulse.min(stream_dim - offset);
                    pulsed_input.slice_axis_mut(Axis(input_axis), (0..count).into()).assign(
                        &fixed_input
                            .to_array_view::<f32>()?
                            .slice_axis(Axis(input_axis), (offset..offset + count).into()),
                    );
                }
                if offset + input_pulse > stream_dim {
                    debug!("Set known_stream_len: {stream_dim}");
                    state.turn_state.resolved_symbols.set(&stream_symbol, stream_dim as _);
                }

                let output =
                    state.run(tvec!(pulsed_input.into_tensor().into()))?.remove(output_slot);

                let output_offset = (i + 1) * output_pulse;
                if output_offset <= delay {
                    continue;
                } else if output_offset - output_pulse >= delay + fixed_output_len {
                    break;
                }

                let (f_o, p_o, count) = if output_offset - output_pulse < delay {
                    // Beginning of signal: partial overlap
                    let count = output_offset - delay;
                    (0, output_pulse - count, count)
                } else if output_offset > delay + fixed_output_len {
                    // End of signal: partial overlap
                    let count = delay + fixed_output_len - (output_offset - output_pulse);
                    (output_offset - output_pulse - delay, 0, count)
                } else {
                    // Full pulse in valid region
                    (output_offset - output_pulse - delay, 0, output_pulse)
                };

                let valid_pulse_result = output.slice(output_axis, p_o, p_o + count)?;
                let valid_fixed_result = fixed_result.slice(output_axis, f_o, f_o + count)?;

                if valid_pulse_result
                    .close_enough(&valid_fixed_result, params.assertions.approximation)
                    .is_err()
                {
                    tract_libcli::terminal::render_node(
                        model.as_ref(),
                        node,
                        &annotations,
                        output_params,
                    )?;
                    println!("pulse: {} ({}..{})", i, i * output_pulse, (i + 1) * output_pulse);
                    println!("expected (ref): {:?}", valid_fixed_result);
                    println!("got (pulsed):   {:?}", valid_pulse_result);
                    failing.insert(node);
                    node_ok = false;
                    break;
                }
            }
            if node_ok {
                ok += 1;
            }
        }
    }

    for n in model.nodes() {
        let color: nu_ansi_term::Style =
            if failing.contains(&n.id) { Red.into() } else { Green.bold() };
        annotations.node_mut(n.id.into()).style = Some(color);
    }

    if log_enabled!(Info) {
        tract_libcli::terminal::render(model.as_ref(), &annotations, output_params)?;
    } else {
        for f in failing.iter().sorted() {
            tract_libcli::terminal::render_node(model.as_ref(), *f, &annotations, output_params)?;
        }
    }

    if !failing.is_empty() {
        bail!("{} node(s) failed stream comparison.", failing.len())
    } else {
        println!(
            "{}",
            Green.paint(format!(
                "{ok} node output(s) passed stream comparison ({skipped} skipped)."
            ))
        );
    }
    Ok(())
}
