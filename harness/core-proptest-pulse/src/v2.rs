use tract_core::dims;
use tract_core::ndarray::*;
use tract_core::ops::cnn::{Conv, KernelFormat, PaddingSpec, PoolSpec};
use tract_core::ops::nn::DataFormat;
use tract_core::plan::SimplePlan;
use tract_core::prelude::*;
use tract_pulse::v2::PulseV2Model;

/// Pulsify a model via PulseV2, run pulse by pulse, stitch output,
/// and compare against batch reference.
fn run_and_compare(
    model: TypedModel,
    stream_sym: &Symbol,
    pulse: usize,
    input: &Tensor,
    axis: usize,
) {
    // Batch reference — feed the full input directly, S resolves from shape
    let batch =
        model.clone().into_runnable().unwrap().run(tvec!(input.clone().into_tvalue())).unwrap();

    // PulseV2: pulsify → lower to typed → run pulse by pulse
    let pv2 = PulseV2Model::new(&model, stream_sym.clone(), pulse).unwrap();
    let skip = pv2.startup_skip();

    // Determine output streaming axis from the batch model
    let batch_output_fact = model.output_fact(0).unwrap();
    let output_axis = batch_output_fact
        .shape
        .iter()
        .position(|d| d.symbols().contains(stream_sym))
        .unwrap_or(axis);

    let typed = pv2.into_typed().unwrap();
    let plan = SimplePlan::new(typed).unwrap();
    let mut state = plan.spawn().unwrap();

    let input_len = input.shape()[axis];
    let mut output_chunks: Vec<Tensor> = vec![];
    let mut written = 0;

    while written < input_len {
        let chunk_len = pulse.min(input_len - written);
        let chunk = input.slice(axis, written, written + chunk_len).unwrap();

        let chunk = if chunk_len < pulse {
            let mut padded_shape = input.shape().to_vec();
            padded_shape[axis] = pulse;
            let mut padded = Tensor::zero_dt(input.datum_type(), &padded_shape).unwrap();
            padded.assign_slice(0..chunk_len, &chunk, 0..chunk_len, axis).unwrap();
            state.turn_state.resolved_symbols.set(stream_sym, input_len as i64);
            padded
        } else {
            chunk.into_tensor()
        };

        let outputs = state.run(tvec!(chunk.into_tvalue())).unwrap();
        output_chunks.push(outputs[0].clone().into_tensor());
        written += pulse;
    }

    // Stitch — no delay to skip, every frame is valid
    let pulsed_output = Tensor::stack_tensors(output_axis, &output_chunks).unwrap();

    // Skip startup frames, then compare against batch
    let batch_output = &batch[0];
    let batch_len = batch_output.shape()[output_axis];
    let pulsed_len = pulsed_output.shape()[output_axis];
    let available = pulsed_len.saturating_sub(skip);
    let compare_len = batch_len.min(available);
    assert!(compare_len > 0, "No valid output (skip={skip}, pulsed_len={pulsed_len})");
    let batch_slice = batch_output.slice(output_axis, 0, compare_len).unwrap();
    let pulsed_slice = pulsed_output.slice(output_axis, skip, skip + compare_len).unwrap();
    pulsed_slice.close_enough(&batch_slice, true).unwrap_or_else(|e| {
        panic!("Mismatch: {e}\nbatch:  {batch_slice:?}\npulsed: {pulsed_slice:?}")
    });
}

/// Identity: Source → output
#[test]
fn test_v2_identity() {
    let mut model = TypedModel::default();
    let s = model.symbols.sym("S");
    let a = model.add_source("a", f32::fact(dims!(1, s, 4))).unwrap();
    model.select_output_outlets(&[a]).unwrap();

    let input =
        Array3::<f32>::from_shape_fn((1, 8, 4), |(_, t, c)| (t * 10 + c) as f32).into_tensor();
    run_and_compare(model, &s, 3, &input, 1);
}

/// Simple conv: Source → Conv (kernel=3, valid padding) → output
/// The conv needs 2 past samples (kernel-1) so pulse-by-pulse without
/// buffering will produce wrong results for now — this test documents
/// the current behavior.
#[test]
fn test_v2_simple_conv() {
    let mut model = TypedModel::default();
    let s = model.symbols.sym("S");
    let a = model.add_source("a", f32::fact(dims!(1, 1, s))).unwrap();
    let kernel = model.add_const("kernel", rctensor3(&[[[0.5f32, 1.0, -0.1]]])).unwrap();
    let bias = model.add_const("bias", tensor0(0f32)).unwrap();
    let conv = model
        .wire_node(
            "conv",
            Conv {
                pool_spec: PoolSpec {
                    data_format: DataFormat::NCHW,
                    kernel_shape: tvec!(3),
                    padding: PaddingSpec::Valid,
                    dilations: None,
                    strides: None,
                    input_channels: 1,
                    output_channels: 1,
                },
                kernel_fmt: KernelFormat::OIHW,
                group: 1,
                q_params: None,
            },
            &[a, kernel, bias],
        )
        .unwrap();
    model.select_output_outlets(&conv).unwrap();

    let input = arr3(&[[[1.0f32, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0]]]).into_tensor();
    // pulse=4, input_len=8, kernel=3 → batch output_len=6
    // Without buffering this will fail — each pulse sees only its own 4 samples
    // and the conv at pulse boundaries misses the overlap.
    run_and_compare(model, &s, 4, &input, 2);
}
