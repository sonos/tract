use proptest::prelude::*;
use proptest::test_runner::TestCaseResult;
use tract_core::dims;
use tract_core::plan::SimplePlan;
use tract_core::prelude::*;
use tract_pulse::v2::PulseV2Model;

/// Pulsify a model via PulseV2, run pulse by pulse, stitch output,
/// and compare against batch reference. Returns TestCaseResult for proptest.
pub fn run_and_compare_v2(
    model: TypedModel,
    pulse: usize,
    input: &Tensor,
    axis: usize,
) -> TestCaseResult {
    // Find the streaming symbol (first symbol in any input shape).
    let stream_sym = model
        .input_fact(0)
        .unwrap()
        .shape
        .iter()
        .flat_map(|d| d.symbols())
        .next()
        .expect("No streaming symbol found in model input");

    // Batch reference
    let batch =
        model.clone().into_runnable().unwrap().run(tvec!(input.clone().into_tvalue())).unwrap();

    // PulseV2: pulsify → lower to typed → run pulse by pulse
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
    let plan = SimplePlan::new(typed).unwrap();
    let mut state = plan.spawn().unwrap();

    let input_len = input.shape()[axis];
    let mut output_chunks: Vec<Tensor> = vec![];
    let mut written = 0;
    let mut pulse_idx: i64 = 0;

    while written < input_len {
        let chunk_len = pulse.min(input_len - written);
        let chunk = input.slice(axis, written, written + chunk_len).unwrap();

        state.turn_state.resolved_symbols.set(&t_sym, pulse_idx);
        state.turn_state.resolved_symbols.set(&p_sym, pulse as i64);

        let chunk = if chunk_len < pulse {
            let mut padded_shape = input.shape().to_vec();
            padded_shape[axis] = pulse;
            let mut padded = Tensor::zero_dt(input.datum_type(), &padded_shape).unwrap();
            padded.assign_slice(0..chunk_len, &chunk, 0..chunk_len, axis).unwrap();
            state.turn_state.resolved_symbols.set(&stream_sym, input_len as i64);
            padded
        } else {
            chunk.into_tensor()
        };

        let outputs = state.run(tvec!(chunk.into_tvalue())).unwrap();
        output_chunks.push(outputs[0].clone().into_tensor());
        written += pulse;
        pulse_idx += 1;
    }

    let pulsed_output = Tensor::stack_tensors(output_axis, &output_chunks).unwrap();

    let batch_output = &batch[0];
    let batch_len = batch_output.shape()[output_axis];
    let pulsed_len = pulsed_output.shape()[output_axis];
    let compare_len = batch_len.min(pulsed_len);
    prop_assert!(compare_len > 0, "No output produced");
    let batch_slice = batch_output.slice(output_axis, 0, compare_len).unwrap();
    let pulsed_slice = pulsed_output.slice(output_axis, 0, compare_len).unwrap();
    prop_assert!(
        pulsed_slice.close_enough(&batch_slice, true).is_ok(),
        "Mismatch:\nbatch:  {:?}\npulsed: {:?}",
        batch_slice,
        pulsed_slice
    );
    Ok(())
}

use super::conv_plus_conv::ConvPlusConvProblem;

/// Full proptest over all conv combinations (stride, dilation, padding).
/// TODO: support stride > 1, explicit padding, dilation in PulseV2.
#[test]
#[ignore]
fn proptest_v2_conv_full() {
    use proptest::test_runner::{Config, TestRunner};
    let mut runner = TestRunner::new(Config::default());
    runner.run(&ConvPlusConvProblem::arbitrary(), |pb| pb.run_v2()).unwrap();
}

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

proptest! {
    #[test]
    fn proptest_v2_single_conv(
        kernel in 1usize..5,
        pulse_extra in 0usize..5,
        input_extra in 0usize..8,
    ) {
        let pulse = kernel + pulse_extra;
        let input_len = kernel + pulse + input_extra;
        let mut model = TypedModel::default();
        let s = model.symbols.sym("S");
        let a = model.add_source("a", f32::fact(dims!(1, 1, s))).unwrap();
        let conv = wire_conv(&mut model, "conv", a, kernel, 1, 1);
        model.select_output_outlets(&conv).unwrap();

        let input_data: Vec<f32> = (0..input_len).map(|i| i as f32).collect();
        let input = tensor1(&input_data).into_shape(&[1, 1, input_len]).unwrap();
        run_and_compare_v2(model, pulse, &input, 2)?;
    }

    #[test]
    fn proptest_v2_conv_chain(
        k1 in 1usize..4,
        k2 in 1usize..4,
        pulse_extra in 0usize..4,
        input_extra in 0usize..6,
    ) {
        let total_lookback = (k1 - 1) + (k2 - 1);
        let pulse = total_lookback.max(1) + pulse_extra;
        let input_len = total_lookback + pulse + input_extra + 1;
        let mut model = TypedModel::default();
        let s = model.symbols.sym("S");
        let a = model.add_source("a", f32::fact(dims!(1, 1, s))).unwrap();
        let c1 = wire_conv(&mut model, "conv1", a, k1, 1, 1);
        let c2 = wire_conv(&mut model, "conv2", c1[0], k2, 1, 1);
        model.select_output_outlets(&c2).unwrap();

        let input_data: Vec<f32> = (0..input_len).map(|i| i as f32).collect();
        let input = tensor1(&input_data).into_shape(&[1, 1, input_len]).unwrap();
        run_and_compare_v2(model, pulse, &input, 2)?;
    }

    #[test]
    fn proptest_v2_conv_dilation(
        kernel in 1usize..4,
        dilation in 1usize..4,
        pulse_extra in 0usize..4,
        input_extra in 0usize..6,
    ) {
        let receptive = (kernel - 1) * dilation;
        let pulse = receptive.max(1) + pulse_extra;
        let input_len = receptive + pulse + input_extra + 1;
        let mut model = TypedModel::default();
        let s = model.symbols.sym("S");
        let a = model.add_source("a", f32::fact(dims!(1, 1, s))).unwrap();
        let conv = wire_conv(&mut model, "conv", a, kernel, dilation, 1);
        model.select_output_outlets(&conv).unwrap();

        let input_data: Vec<f32> = (0..input_len).map(|i| i as f32).collect();
        let input = tensor1(&input_data).into_shape(&[1, 1, input_len]).unwrap();
        run_and_compare_v2(model, pulse, &input, 2)?;
    }

}

proptest! {
    #[test]
    fn proptest_v2_conv_stride(
        kernel in 1usize..4,
        conv_stride in 1usize..4,
        pulse_extra in 0usize..4,
        input_extra in 0usize..6,
    ) {
        let receptive = kernel;
        // pulse must be >= kernel and a multiple of stride
        let min_pulse_factors = (kernel + conv_stride - 1) / conv_stride;
        let pulse = conv_stride * (min_pulse_factors + pulse_extra);
        let input_len = receptive + pulse + input_extra;
        let mut model = TypedModel::default();
        let s = model.symbols.sym("S");
        let a = model.add_source("a", f32::fact(dims!(1, 1, s))).unwrap();
        let conv = wire_conv(&mut model, "conv", a, kernel, 1, conv_stride);
        model.select_output_outlets(&conv).unwrap();

        let input_data: Vec<f32> = (0..input_len).map(|i| i as f32).collect();
        let input = tensor1(&input_data).into_shape(&[1, 1, input_len]).unwrap();
        run_and_compare_v2(model, pulse, &input, 2)?;
    }
}

proptest! {
    #[test]
    fn proptest_v2_crop(
        pulse in 1usize..5,
        input_len in 1usize..10,
        begin in 0usize..3,
        end_margin in 0usize..3,
    ) {
        use tract_core::ops::array::Slice;
        let full_len = input_len + begin + end_margin;
        let slice_end = begin + input_len;
        let mut model = TypedModel::default();
        let s = model.symbols.sym("S");
        let a = model.add_source("a", f32::fact(&[TDim::from(s.clone())])).unwrap();
        let slice = model.wire_node("slice", Slice::new(0, begin, slice_end), &[a]).unwrap();
        model.select_output_outlets(&slice).unwrap();

        let input_data: Vec<f32> = (1..=full_len).map(|i| i as f32).collect();
        let input = tensor1(&input_data);
        run_and_compare_v2(model, pulse, &input, 0)?;
    }
}
