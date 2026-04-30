use proptest::prelude::*;
use proptest::test_runner::TestCaseResult;
use tract_core::ops::cnn::Conv;
use tract_core::plan::SimplePlan;
use tract_core::prelude::*;
use tract_pulse::v2::PulseV2Model;
use tract_pulse::v2_buffer::PulseV2Buffer;
use tract_pulse::v2_slice::PulseV2Slice;

/// Pulsify a model via PulseV2, run pulse by pulse, stitch output,
/// and compare against batch reference.
pub fn run_and_compare_v2(
    model: TypedModel,
    pulse: usize,
    input: &Tensor,
    axis: usize,
) -> TestCaseResult {
    let stream_sym = model
        .input_fact(0)
        .unwrap()
        .shape
        .iter()
        .flat_map(|d| d.symbols())
        .next()
        .expect("No streaming symbol found in model input");

    let batch =
        model.clone().into_runnable().unwrap().run(tvec!(input.clone().into_tvalue())).unwrap();

    let pv2 = PulseV2Model::new(&model, stream_sym.clone()).unwrap();
    let t_sym = pv2.symbols.pulse_id.clone();
    let p_sym = pv2.symbols.pulse.clone();

    let batch_output_fact = model.output_fact(0).unwrap();
    let output_axis = batch_output_fact
        .shape
        .iter()
        .position(|d| d.symbols().contains(&stream_sym))
        .unwrap_or(axis);

    let typed = pv2.into_typed().unwrap();
    // Output-axis delay: each PulseV2Buffer pre-fills `lookback` zeros on the
    // input axis; the corresponding garbage shows up at the OUTPUT axis after
    // being divided by every stride applied between the buffer and the output.
    // For `Source → Buffer(L) → Conv(stride=s) → output`, output garbage =
    // L/s. For chained ops, the stride compounds. Approximation here: walk
    // the model once, accumulate `total_lookback` from all buffers and
    // `total_stride` from all Conv nodes that touch the streaming axis,
    // delay = total_lookback / total_stride. Holds for the proptest models
    // (single linear streaming path) — real graphs would need per-buffer
    // accumulated downstream stride.
    let mut total_lookback = 0usize;
    let mut total_stride = 1usize;
    let mut total_slice_start: i64 = 0;
    for node in typed.nodes() {
        if let Some(buf) = node.op.downcast_ref::<PulseV2Buffer>() {
            total_lookback += buf.lookback.iter().copied().max().unwrap_or(0);
        }
        if let Some(conv) = node.op.downcast_ref::<Conv>() {
            let h_axis = conv.pool_spec.data_format.h_axis();
            if axis >= h_axis {
                let geo_ix = axis - h_axis;
                let s = conv
                    .pool_spec
                    .strides
                    .as_ref()
                    .and_then(|st| st.get(geo_ix).copied())
                    .unwrap_or(1);
                total_stride *= s;
            }
        }
        if let Some(slice) = node.op.downcast_ref::<PulseV2Slice>() {
            if slice.axis == axis {
                if let Ok(s) = slice.start.to_i64() {
                    total_slice_start += s;
                }
            }
        }
    }
    let total_delay = total_lookback / total_stride.max(1) + total_slice_start as usize;
    let plan = SimplePlan::new(typed).unwrap();
    let mut state = plan.spawn().unwrap();

    let input_len = input.shape()[axis];
    let mut output_chunks: Vec<Tensor> = vec![];
    let mut written = 0;
    let mut pulse_idx: i64 = 0;
    let batch_len = batch[0].shape()[output_axis];

    loop {
        let chunk_len = pulse.min(input_len.saturating_sub(written));

        state.turn_state.resolved_symbols.set(&t_sym, pulse_idx);
        state.turn_state.resolved_symbols.set(&p_sym, pulse as i64);
        state.turn_state.resolved_symbols.set(&stream_sym, input_len as i64);

        let chunk = if chunk_len == 0 {
            let mut shape = input.shape().to_vec();
            shape[axis] = pulse;
            Tensor::zero_dt(input.datum_type(), &shape).unwrap()
        } else if chunk_len < pulse {
            let chunk = input.slice(axis, written, written + chunk_len).unwrap();
            let mut padded_shape = input.shape().to_vec();
            padded_shape[axis] = pulse;
            let mut padded = Tensor::zero_dt(input.datum_type(), &padded_shape).unwrap();
            padded.assign_slice(0..chunk_len, &chunk, 0..chunk_len, axis).unwrap();
            padded
        } else {
            input.slice(axis, written, written + chunk_len).unwrap().into_tensor()
        };

        let outputs = state.run(tvec!(chunk.into_tvalue())).unwrap();
        let out = outputs[0].clone().into_tensor();
        if out.shape()[output_axis] > 0 {
            output_chunks.push(out);
        }
        written += pulse;
        pulse_idx += 1;

        let total_out: usize = output_chunks.iter().map(|t| t.shape()[output_axis]).sum();
        if total_out >= batch_len + total_delay {
            break;
        }
        if pulse_idx > 1000 {
            panic!("Pulsed run exceeded 1000 pulses");
        }
    }

    let pulsed_output = Tensor::stack_tensors(output_axis, &output_chunks).unwrap();

    let batch_output = &batch[0];
    let pulsed_len = pulsed_output.shape()[output_axis];
    prop_assert!(
        pulsed_len >= total_delay,
        "Pulsed output ({pulsed_len}) shorter than total_delay ({total_delay})"
    );
    let pulsed_valid = pulsed_output.slice(output_axis, total_delay, pulsed_len).unwrap();
    let pulsed_valid_len = pulsed_valid.shape()[output_axis];
    let compare_len = batch_len.min(pulsed_valid_len);
    prop_assert!(compare_len > 0, "No output produced");
    let batch_slice = batch_output.slice(output_axis, 0, compare_len).unwrap();
    let pulsed_slice = pulsed_valid.slice(output_axis, 0, compare_len).unwrap();
    prop_assert!(
        pulsed_slice.close_enough(&batch_slice, true).is_ok(),
        "Mismatch:\nbatch:  {:?}\npulsed: {:?}",
        batch_slice,
        pulsed_slice
    );
    Ok(())
}

// ── Proptests using V1 problem generators ──────────────────────────────

use super::conv_plus_conv::ConvPlusConvProblem;
use super::deconv::DeconvProblem;
use super::delay_plus_downsample::DelayPlusDownsampleProblem;
use super::delay_plus_pool::DelayPlusPoolProblem;
use super::pad_plus_conv::PadPlusConvProblem;

#[test]
#[ignore]
fn proptest_v2_conv_full() {
    use proptest::test_runner::{Config, TestRunner};
    let mut runner = TestRunner::new(Config::default());
    runner.run(&ConvPlusConvProblem::arbitrary(), |pb| pb.run_v2()).unwrap();
}

#[test]
#[ignore]
fn proptest_v2_pad_plus_conv_full() {
    use proptest::test_runner::{Config, TestRunner};
    let mut runner = TestRunner::new(Config::default());
    runner.run(&PadPlusConvProblem::arbitrary(), |pb| pb.run_v2()).unwrap();
}

#[test]
#[ignore]
fn proptest_v2_delay_plus_pool_full() {
    use proptest::test_runner::{Config, TestRunner};
    let mut runner = TestRunner::new(Config::default());
    runner.run(&DelayPlusPoolProblem::arbitrary(), |pb| pb.run_v2()).unwrap();
}

#[test]
#[ignore]
fn proptest_v2_deconv_full() {
    use proptest::test_runner::{Config, TestRunner};
    let mut runner = TestRunner::new(Config::default());
    runner.run(&DeconvProblem::arbitrary(), |pb| pb.run_v2()).unwrap();
}

#[test]
#[ignore]
fn proptest_v2_delay_plus_downsample_full() {
    use proptest::test_runner::{Config, TestRunner};
    let mut runner = TestRunner::new(Config::default());
    runner.run(&DelayPlusDownsampleProblem::arbitrary(), |pb| pb.run_v2()).unwrap();
}

// ── Focused proptests for supported subset ─────────────────────────────

use tract_core::dims;
use tract_core::ops::cnn::*;
use tract_core::ops::nn::DataFormat;

fn wire_conv(
    model: &mut TypedModel,
    name: &str,
    input: OutletId,
    kernel_size: usize,
    dilation: usize,
    stride: usize,
) -> TVec<OutletId> {
    let ker_data: Vec<f32> = (0..kernel_size).map(|i| (i + 1) as f32).collect();
    let ker_tensor = tensor1(&ker_data).into_shape(&[1, 1, kernel_size]).unwrap();
    let k = model.add_const(format!("{name}.k"), ker_tensor).unwrap();
    let b = model.add_const(format!("{name}.b"), tensor0(0f32)).unwrap();
    model
        .wire_node(
            name,
            Conv {
                pool_spec: PoolSpec {
                    data_format: DataFormat::NCHW,
                    kernel_shape: tvec!(kernel_size),
                    padding: PaddingSpec::Valid,
                    dilations: if dilation > 1 { Some(tvec!(dilation)) } else { None },
                    strides: if stride > 1 { Some(tvec!(stride)) } else { None },
                    input_channels: 1,
                    output_channels: 1,
                },
                kernel_fmt: KernelFormat::OIHW,
                group: 1,
                q_params: None,
            },
            &[input, k, b],
        )
        .unwrap()
}

fn nchw_model_and_input(
    ops: impl FnOnce(&mut TypedModel, OutletId) -> TVec<OutletId>,
    input_len: usize,
) -> (TypedModel, Tensor) {
    let mut model = TypedModel::default();
    let s = model.symbols.sym("S");
    let a = model.add_source("a", f32::fact(dims!(1, 1, s))).unwrap();
    let out = ops(&mut model, a);
    model.select_output_outlets(&out).unwrap();
    let input_data: Vec<f32> = (0..input_len).map(|i| i as f32).collect();
    let input = tensor1(&input_data).into_shape(&[1, 1, input_len]).unwrap();
    (model, input)
}

proptest! {
    #[test]
    fn proptest_v2_single_conv(kernel in 1usize..5, pulse_extra in 0usize..5, input_extra in 0usize..8) {
        let pulse = kernel + pulse_extra;
        let (model, input) = nchw_model_and_input(
            |m, a| wire_conv(m, "conv", a, kernel, 1, 1),
            kernel + pulse + input_extra,
        );
        run_and_compare_v2(model, pulse, &input, 2)?;
    }

    #[test]
    fn proptest_v2_conv_chain(k1 in 1usize..4, k2 in 1usize..4, pulse_extra in 0usize..4, input_extra in 0usize..6) {
        let lookback = (k1 - 1) + (k2 - 1);
        let pulse = lookback.max(1) + pulse_extra;
        let (model, input) = nchw_model_and_input(
            |m, a| { let c1 = wire_conv(m, "c1", a, k1, 1, 1); wire_conv(m, "c2", c1[0], k2, 1, 1) },
            lookback + pulse + input_extra + 1,
        );
        run_and_compare_v2(model, pulse, &input, 2)?;
    }

    #[test]
    fn proptest_v2_conv_dilation(kernel in 1usize..4, dilation in 1usize..4, pulse_extra in 0usize..4, input_extra in 0usize..6) {
        let receptive = (kernel - 1) * dilation;
        let pulse = receptive.max(1) + pulse_extra;
        let (model, input) = nchw_model_and_input(
            |m, a| wire_conv(m, "conv", a, kernel, dilation, 1),
            receptive + pulse + input_extra + 1,
        );
        run_and_compare_v2(model, pulse, &input, 2)?;
    }
}

proptest! {
    #[test]
    fn proptest_v2_conv_stride(kernel in 1usize..4, conv_stride in 1usize..4, pulse_extra in 0usize..4, input_extra in 0usize..6) {
        let min_factors = (kernel + conv_stride - 1) / conv_stride;
        let pulse = conv_stride * (min_factors + pulse_extra);
        let (model, input) = nchw_model_and_input(
            |m, a| wire_conv(m, "conv", a, kernel, 1, conv_stride),
            kernel + pulse + input_extra,
        );
        run_and_compare_v2(model, pulse, &input, 2)?;
    }
}

proptest! {
    #[test]
    fn proptest_v2_crop(pulse in 1usize..5, input_len in 1usize..10, begin in 0usize..3, end_margin in 0usize..3) {
        use tract_core::ops::array::Slice;
        let full_len = input_len + begin + end_margin;
        let mut model = TypedModel::default();
        let s = model.symbols.sym("S");
        let a = model.add_source("a", f32::fact(&[TDim::from(s.clone())])).unwrap();
        let slice = model.wire_node("slice", Slice::new(0, begin, begin + input_len), &[a]).unwrap();
        model.select_output_outlets(&slice).unwrap();
        let input = tensor1(&(1..=full_len).map(|i| i as f32).collect::<Vec<_>>());
        run_and_compare_v2(model, pulse, &input, 0)?;
    }
}

proptest! {
    #[test]
    fn proptest_v2_pad(pulse in 1usize..5, input_len in 1usize..10, before in 0usize..4, after in 0usize..4) {
        use tract_core::ops::array::{Pad, PadMode};
        let mut model = TypedModel::default();
        let s = model.symbols.sym("S");
        let a = model.add_source("a", f32::fact(&[TDim::from(s.clone())])).unwrap();
        let pad = model.wire_node("pad", Pad::new(vec![(before, after)], PadMode::Constant(Arc::new(Tensor::from(-1f32)))), &[a]).unwrap();
        model.select_output_outlets(&pad).unwrap();
        let input = tensor1(&(1..=input_len).map(|i| i as f32).collect::<Vec<_>>());
        run_and_compare_v2(model, pulse, &input, 0)?;
    }
}

proptest! {
    #[test]
    fn proptest_v2_pad_plus_conv(kernel in 1usize..4, pad_before in 0usize..4, pad_after in 0usize..4, pulse_extra in 0usize..4, input_extra in 0usize..6) {
        use tract_core::ops::array::{Pad, PadMode};
        let pulse = kernel.max(1) + pulse_extra;
        let input_len = kernel + pulse + input_extra;
        let mut model = TypedModel::default();
        let s = model.symbols.sym("S");
        let a = model.add_source("a", f32::fact(dims!(1, 1, s))).unwrap();
        let wire = model.wire_node("pad", Pad::new(vec![(0, 0), (0, 0), (pad_before, pad_after)], PadMode::Constant(Arc::new(Tensor::from(-1f32)))), &[a]).unwrap()[0];
        let conv = wire_conv(&mut model, "conv", wire, kernel, 1, 1);
        model.select_output_outlets(&conv).unwrap();
        let input_data: Vec<f32> = (0..input_len).map(|i| i as f32).collect();
        let input = tensor1(&input_data).into_shape(&[1, 1, input_len]).unwrap();
        run_and_compare_v2(model, pulse, &input, 2)?;
    }
}
