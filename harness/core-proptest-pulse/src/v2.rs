use tract_core::dims;
use tract_core::ndarray::*;
use tract_core::ops::cnn::{Conv, KernelFormat, PaddingSpec, PoolSpec};
use tract_core::ops::nn::DataFormat;
use tract_core::plan::SimplePlan;
use tract_core::prelude::*;

/// Run a batch model pulse by pulse (naive: no buffering) and return
/// the stitched output. This is a test utility, not the real pulsifier.
fn run_pulse_v2_naive(
    batch_model: &TypedModel,
    stream_sym: &Symbol,
    pulse: usize,
    input_array: &Tensor,
    input_axis: usize,
) -> TractResult<Tensor> {
    let input_len = input_array.shape()[input_axis];

    // Create the pulsed typed model: same graph but with S → pulse in shapes.
    let sv = SymbolValues::default().with(stream_sym, pulse as i64);
    let pulsed = batch_model.clone().concretize_dims(&sv)?;
    let plan = SimplePlan::new(pulsed)?;
    let mut state = plan.spawn()?;

    // Determine output axis
    let batch_output_fact = batch_model.output_fact(0)?;
    let output_axis = batch_output_fact
        .shape
        .iter()
        .position(|d| d.symbols().contains(stream_sym))
        .unwrap_or(input_axis);

    let mut output_chunks: Vec<Tensor> = vec![];
    let mut written = 0;

    while written < input_len {
        let chunk_len = pulse.min(input_len - written);
        let chunk = input_array.slice(input_axis, written, written + chunk_len)?;

        let chunk = if chunk_len < pulse {
            let mut padded_shape = input_array.shape().to_vec();
            padded_shape[input_axis] = pulse;
            let mut padded = Tensor::zero_dt(input_array.datum_type(), &padded_shape)?;
            padded.assign_slice(0..chunk_len, &chunk, 0..chunk_len, input_axis)?;
            state.turn_state.resolved_symbols.set(stream_sym, input_len as i64);
            padded
        } else {
            chunk.into_tensor()
        };

        let outputs = state.run(tvec!(chunk.into_tvalue()))?;
        output_chunks.push(outputs[0].clone().into_tensor());
        written += pulse;
    }

    Tensor::stack_tensors(output_axis, &output_chunks)
}

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

    // PulseV2
    let pulsed_output = run_pulse_v2_naive(&model, stream_sym, pulse, input, axis).unwrap();

    // Compare: trim both to the shorter length on the streaming axis
    let batch_output = &batch[0];
    let batch_len = batch_output.shape()[axis];
    let pulsed_len = pulsed_output.shape()[axis];
    let compare_len = batch_len.min(pulsed_len);
    let batch_slice = batch_output.slice(axis, 0, compare_len).unwrap();
    let pulsed_slice = pulsed_output.slice(axis, 0, compare_len).unwrap();
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
