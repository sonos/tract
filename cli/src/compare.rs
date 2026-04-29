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

    if sub_matches.get_flag("stream") {
        return handle_stream(params, &output_params, &run_params);
    }

    let cumulative = sub_matches.get_flag("cumulative");
    let resilent = sub_matches.get_flag("resilient");
    if sub_matches.get_one::<String>("stage").is_some() {
        // --with is by pipeline and put in params
        return handle_reference_stage(cumulative, params, &output_params, &run_params);
    } else if let Some(npz) = sub_matches.get_one::<String>("npz") {
        return handle_npz(cumulative, npz, params, &output_params, &run_params);
    } else if sub_matches.get_flag("twice") {
        return handle_twice(cumulative, params, &output_params, &run_params);
    }
    if let Some(pbdir) = sub_matches.get_one::<String>("pbdir") {
        return handle_pbdir(cumulative, pbdir, params, &output_params, &run_params);
    }
    if sub_matches.get_flag("tf") {
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
        &output_params,
        run_params,
        ("tract", "tf"),
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
        ("tract", "npz"),
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
        ("tract", "pbdir"),
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
        ("tract", "reference"),
    ))
}

// handle_stream runs the pulsed model pulse-by-pulse, stitches per-node valid output regions
// (accounting for each node's delay), then delegates to compare() which runs the concretized
// reference model and handles all comparison, annotation, and rendering.
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
    run_params: &RunParams,
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

    let ref_input_fact = reference.input_fact(0)?;
    let model_input_fact = model.input_fact(0)?;

    let input_axis = model
        .properties
        .get("pulse.input_axes")
        .context("Expect pulse.input_axes property")?
        .cast_to::<i64>()?
        .try_as_plain()?
        .as_slice::<i64>()?[0] as usize;

    let stream_symbol = ref_input_fact.shape[input_axis]
        .symbols()
        .into_iter()
        .next()
        .context("Could not find streaming symbol in reference model input")?;

    let input_pulse = model_input_fact.shape[input_axis]
        .to_usize()
        .context("Pulse dimension should be concrete")?;

    // Recreate PulsedModel to get per-node delay/axis/pulse metadata
    let pulsed = PulsedModel::new(reference, stream_symbol.clone(), &input_pulse.to_dim())?;

    // Pre-build per-node pulse metadata (keyed by node name, slot 0 only)
    struct PulseInfo {
        delay: usize,
        output_axis: usize,
        output_pulse: usize,
        fixed_output_len: usize,
    }
    let mut max_delay: usize = 0;
    let mut pulse_meta: HashMap<String, PulseInfo> = HashMap::new();

    // We need stream_dim to compute fixed_output_len, but stream_dim depends on max_delay.
    // First pass: find max_delay.
    for pulsed_node in pulsed.nodes() {
        if let Ok(fact) = pulsed.outlet_fact(OutletId::new(pulsed_node.id, 0)) {
            if let Some(stream) = &fact.stream {
                max_delay = max_delay.max(stream.delay);
            }
        }
    }

    let stream_dim = max_delay + 3 * input_pulse + input_pulse / 2;
    let concrete_sym_values = SymbolValues::default().with(&stream_symbol, stream_dim as _);

    // Second pass: build full metadata with fixed_output_len
    for pulsed_node in pulsed.nodes() {
        if let Ok(fact) = pulsed.outlet_fact(OutletId::new(pulsed_node.id, 0)) {
            if let Some(stream) = &fact.stream {
                let output_pulse = fact.pulse().context("no pulse")?.to_usize()?;
                let fixed_output_len = stream.dim.eval_to_i64(&concrete_sym_values)? as usize;
                pulse_meta.insert(
                    pulsed_node.name.clone(),
                    PulseInfo {
                        delay: stream.delay,
                        output_axis: stream.axis,
                        output_pulse,
                        fixed_output_len,
                    },
                );
            }
        }
    }

    // Generate the fixed input (same seed as compare()'s get_or_make_inputs will use)
    let concrete_shape: Vec<usize> =
        ref_input_fact.shape.eval_to_usize(&concrete_sym_values)?.into_owned().into_vec();
    let concrete_input_fact = ref_input_fact.datum_type.fact(&*concrete_shape);
    let fixed_input = tract_libcli::tensor::tensor_for_fact(&concrete_input_fact, None, None)?;

    // Run the pulsed model pulse-by-pulse, collecting per-node valid output slices
    let plan = Arc::new(model.as_ref().clone().into_runnable()?);
    let mut state = SimpleState::new(&plan)?;
    let input_shape: TVec<usize> =
        model_input_fact.shape.iter().map(|d| d.to_usize()).collect::<TractResult<TVec<_>>>()?;

    let mut node_slices: HashMap<String, Vec<Tensor>> = HashMap::new();
    let mut node_axes: HashMap<String, usize> = HashMap::new();
    let num_pulses = (stream_dim + max_delay) / input_pulse + 2;

    for i in 0..num_pulses {
        let mut pulsed_input = ArrayD::from_elem(&*input_shape, f32::NAN);
        let offset = i * input_pulse;
        if offset < stream_dim {
            let count = input_pulse.min(stream_dim - offset);
            pulsed_input.slice_axis_mut(Axis(input_axis), (0..count).into()).assign(
                &fixed_input
                    .to_plain_array_view::<f32>()?
                    .slice_axis(Axis(input_axis), (offset..offset + count).into()),
            );
        }
        if offset + input_pulse > stream_dim {
            state.turn_state.resolved_symbols.set(&stream_symbol, stream_dim as _);
        }

        state.run_plan_with_eval(
            tvec!(pulsed_input.into_tensor().into()),
            |session, op_state, node, input| -> TractResult<TVec<TValue>> {
                let result = tract_core::plan::eval(session, op_state, node, input)?;

                if let Some(info) = pulse_meta.get(&node.name) {
                    // Skip nodes where the optimizer removed the streaming axis (e.g.
                    // ChangeAxes squeezing a singleton batch dim from an intermediate
                    // EinSum). The optimised model is still correct end-to-end; we just
                    // cannot slice along the expected axis any more.
                    let streaming_axis_ok = result[0].rank() > info.output_axis
                        && result[0].shape()[info.output_axis] == info.output_pulse;
                    if streaming_axis_ok {
                        let output_offset = (i + 1) * info.output_pulse;
                        // Check if this pulse has valid output for this node
                        if output_offset > info.delay
                            && output_offset - info.output_pulse
                                < info.delay + info.fixed_output_len
                        {
                            let (p_o, count) = if output_offset - info.output_pulse < info.delay {
                                // Beginning of signal: partial overlap
                                let count = output_offset - info.delay;
                                (info.output_pulse - count, count)
                            } else if output_offset > info.delay + info.fixed_output_len {
                                // End of signal: partial overlap
                                let count = info.delay + info.fixed_output_len
                                    - (output_offset - info.output_pulse);
                                (0, count)
                            } else {
                                // Full pulse in valid region
                                (0, info.output_pulse)
                            };
                            let valid = result[0].slice(info.output_axis, p_o, p_o + count)?;
                            node_slices
                                .entry(node.name.clone())
                                .or_default()
                                .push(valid.into_tensor());
                            node_axes.insert(node.name.clone(), info.output_axis);
                        }
                    }
                } else if i == 0 && node.outputs.len() == 1 {
                    // Non-streaming node: capture on first pulse only
                    node_slices
                        .entry(node.name.clone())
                        .or_default()
                        .push(result[0].clone().into_tensor());
                }

                Ok(result)
            },
        )?;
    }

    // Stitch: concatenate valid slices per node into full-length tensors
    let mut all_values: HashMap<String, Vec<TractResult<TValue>>> = HashMap::new();
    for (name, slices) in &node_slices {
        if let Some(&axis) = node_axes.get(name) {
            let stitched = Tensor::stack_tensors(axis, slices)?;
            all_values.insert(name.clone(), vec![Ok(stitched.into_tvalue())]);
        } else {
            all_values.insert(name.clone(), vec![Ok(slices[0].clone().into_tvalue())]);
        }
    }

    // Concretize the reference model and delegate to compare()
    let concrete_ref =
        Arc::new(reference.clone().substitute_symbols(&concrete_sym_values.to_dim_map())?);
    compare(
        false,
        &concrete_ref,
        &all_values,
        params,
        output_params,
        run_params,
        ("reference", "pulsed"),
    )
}

pub fn compare<F, O>(
    cumulative: bool,
    tract: &Arc<Graph<F, O>>,
    all_values: &HashMap<String, Vec<TractResult<TValue>>>,
    params: &Parameters,
    output_params: &DisplayParams,
    run_params: &RunParams,
    labels: (&str, &str),
) -> TractResult<()>
where
    F: Fact + Clone + Hash,
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
                    let needed_shape =
                        clarified_fact.shape.eval_to_usize(&session_state.resolved_symbols)?;

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
                        msg.push(format!("{:<8}: {:?}", labels.0, computed));
                        msg.push(format!("{:<8}: {:?}", labels.1, reference));
                        tags.sections.push(msg);
                    } else {
                        debug!(
                            "At turn {}, matching value for {:?}",
                            turn,
                            OutletId::new(node.id, slot)
                        )
                    }

                    if !cumulative && returning[slot].is_plain() {
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
