//! End-to-end test of PulseV2 against the hey_snips WaveNet model.
//!
//! Loads the cached .pb, brings it to typed/decluttered form, runs both the
//! batch model and the pulsified v2 model on the same random input, stitches
//! the per-pulse outputs back together, and compares numerically against
//! batch (modulo the pulsified model's startup delay).

use std::path::PathBuf;

use tract_core::ops::cnn::Conv;
use tract_core::plan::SimplePlan;
use tract_core::prelude::*;
use tract_pulse::v2::PulseV2Model;
use tract_pulse::v2_buffer::PulseV2Buffer;
use tract_pulse::v2_slice::PulseV2Slice;
use tract_tensorflow::prelude::*;

fn cached_hey_snips() -> Option<PathBuf> {
    for candidate in [
        std::env::var("MODELS").ok().map(|m| PathBuf::from(m).join("hey_snips_v4_model17.pb")),
        Some(PathBuf::from(".cached/hey_snips_v4_model17.pb")),
        Some(PathBuf::from("../../.cached/hey_snips_v4_model17.pb")),
    ]
    .into_iter()
    .flatten()
    {
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}

#[test]
fn hey_snips_pulsify_v2() -> TractResult<()> {
    let Some(path) = cached_hey_snips() else {
        eprintln!("hey_snips_v4_model17.pb not cached — skipping");
        return Ok(());
    };

    let mut model = tensorflow().model_for_path(&path)?;
    let s = model.symbols.sym("S");
    model.set_input_fact(0, f32::fact([s.to_dim(), 20.to_dim()]).into())?;
    // Pulsification operates on the typed model. Don't call `into_optimized()`
    // here — that decomposes Conv into Im2col + OptMatMul and inserts AxisOp
    // wrappers that pulse-v2's RegionTransform inventory doesn't handle.
    let model = model.into_typed()?.into_decluttered()?;

    // Concrete stream length for the comparison run. Wavenet's receptive
    // field is ~180 samples; pick something well past it so steady-state
    // dominates.
    let stream_len: usize = 256;
    let pulse: usize = 8;
    let stream_axis = 0;

    // Random-but-deterministic input.
    let input_shape = [stream_len, 20];
    let mut input_vec: Vec<f32> = Vec::with_capacity(stream_len * 20);
    for i in 0..stream_len * 20 {
        input_vec.push((i as f32 * 0.137).sin());
    }
    let input = tract_ndarray::Array::from_shape_vec(input_shape, input_vec)?.into_tensor();

    // Run batch reference.
    let concrete_input_fact = f32::fact(&input_shape);
    let mut batch_model = model.clone();
    batch_model.set_input_fact(0, concrete_input_fact.into())?;
    let batch_model = batch_model.into_decluttered()?;
    let batch_outputs = batch_model.into_runnable()?.run(tvec!(input.clone().into_tvalue()))?;
    let batch_output = batch_outputs[0].clone().into_tensor();

    // Pulsify and run pulse-by-pulse.
    let pv2 = PulseV2Model::new(&model, s.clone())?;
    let t_sym = pv2.symbols.pulse_id.clone();
    let p_sym = pv2.symbols.pulse.clone();
    let typed = pv2.into_typed()?;

    let output_axis = typed
        .output_fact(0)?
        .shape
        .iter()
        .position(|d| d.symbols().contains(&p_sym))
        .expect("Pulsed output should have a streaming axis");

    // Total delay = sum of buffer lookbacks (divided by total stride) +
    // sum of slice starts on the streaming axis.
    // Quick instrumentation of inserted ops, for debugging.
    let mut buf_count = 0usize;
    let mut conv_count = 0usize;
    let mut slice_count = 0usize;
    for node in typed.nodes() {
        if node.op.downcast_ref::<PulseV2Buffer>().is_some() {
            buf_count += 1;
        }
        if node.op.downcast_ref::<Conv>().is_some() {
            conv_count += 1;
        }
        if node.op.downcast_ref::<PulseV2Slice>().is_some() {
            slice_count += 1;
        }
    }
    eprintln!(
        "PulseV2 wavenet inventory: buffers={buf_count} convs={conv_count} slices={slice_count}"
    );

    let plan = SimplePlan::new(typed)?;
    let mut state = plan.spawn()?;

    let batch_len = batch_output.shape()[output_axis];
    let mut output_chunks: Vec<Tensor> = vec![];
    let mut written = 0usize;
    let mut pulse_idx: i64 = 0;
    // Run exactly as many pulses as needed to consume the input stream.
    // Per-pulse output is constant (= pulse, mod stride). After
    // `num_pulses * out_per_pulse` total output samples, the steady-state
    // tail of the pulsed output corresponds to the BATCH output's tail.
    let num_input_pulses = stream_len.div_ceil(pulse);
    while pulse_idx < num_input_pulses as i64 {
        let chunk_len = pulse.min(stream_len.saturating_sub(written));

        state.turn_state.resolved_symbols.set(&t_sym, pulse_idx);
        state.turn_state.resolved_symbols.set(&p_sym, pulse as i64);
        state.turn_state.resolved_symbols.set(&s, stream_len as i64);

        let chunk = if chunk_len == 0 {
            let mut shape = input.shape().to_vec();
            shape[stream_axis] = pulse;
            Tensor::zero_dt(input.datum_type(), &shape)?
        } else if chunk_len < pulse {
            let chunk = input.slice(stream_axis, written, written + chunk_len)?;
            let mut padded_shape = input.shape().to_vec();
            padded_shape[stream_axis] = pulse;
            let mut padded = Tensor::zero_dt(input.datum_type(), &padded_shape)?;
            padded.assign_slice(0..chunk_len, &chunk, 0..chunk_len, stream_axis)?;
            padded
        } else {
            input.slice(stream_axis, written, written + chunk_len)?.into_tensor()
        };

        let outputs = state.run(tvec!(chunk.into_tvalue()))?;
        let out = outputs[0].clone().into_tensor();
        if out.shape()[output_axis] > 0 {
            output_chunks.push(out);
        }
        written += pulse;
        pulse_idx += 1;
    }

    let pulsed_output = Tensor::stack_tensors(output_axis, &output_chunks)?;
    let pulsed_len = pulsed_output.shape()[output_axis];
    assert!(pulsed_len >= batch_len, "pulsed_len={pulsed_len} batch_len={batch_len}");
    // Steady-state tail: last `batch_len` samples of pulsed output line up
    // with the full batch output (assuming the model has run long enough
    // for warmup garbage to wash out, which `stream_len > 2 × receptive_field`
    // ensures).
    let compare_len = batch_len;
    let pulsed_slice = pulsed_output.slice(output_axis, pulsed_len - compare_len, pulsed_len)?;
    let batch_slice = batch_output.slice(output_axis, 0, compare_len)?;

    println!(
        "stream_len={stream_len} pulse={pulse} batch_len={batch_len} pulsed_len={pulsed_len} \
         compare_len={compare_len}"
    );
    if let Err(e) = pulsed_slice.close_enough(&batch_slice, true) {
        panic!("Mismatch:\nbatch:  {batch_slice:?}\npulsed: {pulsed_slice:?}\n{e}");
    }
    println!("PulseV2 hey_snips numerical match: {compare_len} samples on output axis");
    Ok(())
}
