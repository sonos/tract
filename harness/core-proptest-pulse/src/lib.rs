#![cfg(test)]

#[macro_use]
extern crate log;

use proptest::prelude::*;
use proptest::proptest;
use proptest::test_runner::TestCaseResult;
use proptest::*;
use tract_core::ndarray::arr3;
use tract_core::num_traits::Zero;
use tract_core::ops::array::Pad;
use tract_core::ops::array::PadMode;
use tract_core::ops::array::Slice;
use tract_core::ops::cnn::Conv;
use tract_core::ops::cnn::KernelFormat;
use tract_core::ops::cnn::PaddingSpec;
use tract_core::ops::cnn::PoolSpec;
use tract_core::ops::nn::DataFormat;
use tract_ndarray::prelude::*;
use tract_pulse::internal::*;

mod conv_plus_conv;
mod deconv;
mod delay_plus_downsample;
mod delay_plus_pool;
mod einsum;
mod pad_plus_conv;

#[allow(dead_code)]
fn setup_test_logger() {
    let _ = env_logger::Builder::from_env("TRACT_LOG").try_init();
}

fn proptest_regular_against_pulse(
    model: TypedModel,
    pulse: usize,
    input_array: tract_ndarray::ArrayD<f32>,
    axis: usize,
) -> TestCaseResult {
    setup_test_logger();

    let len = input_array.shape()[axis];
    let model = model.into_decluttered().unwrap();
    // dbg!(&model);
    let s = model.symbols.sym("S");
    let symbols = SymbolValues::default().with(&s, len as i64);

    let concrete = model.clone().concretize_dims(&symbols).unwrap();
    if concrete.nodes.iter().any(|n| n.outputs.iter().any(|o| o.fact.shape.volume().is_zero())) {
        return Err(TestCaseError::reject("too short input"));
    }
    let runnable = concrete.into_runnable().unwrap();

    // dbg!(&runnable);
    let outputs = runnable.run(tvec!(input_array.clone().into_tvalue())).unwrap();
    // dbg!(&outputs);
    debug!("Build pulsing model");
    // dbg!(&model);
    let pulsed = PulsedModel::new(&model, s.clone(), &pulse.to_dim()).unwrap();
    // dbg!(&pulsed);
    let output_fact = pulsed.output_fact(0).unwrap().clone();

    let stream_info = output_fact.stream.as_ref().unwrap();
    prop_assert!(stream_info.dim.eval(&symbols) == outputs[0].shape()[stream_info.axis].to_dim());
    let output_stream_axis = stream_info.axis;
    let delay = stream_info.delay;
    let mut initial_output_shape = output_fact.shape.clone();
    initial_output_shape.set(output_stream_axis, 0.to_dim());
    let initial_output_shape: TVec<usize> =
        initial_output_shape.iter().map(|d| d.to_usize().unwrap()).collect();

    let pulsed_plan = SimplePlan::new(pulsed).unwrap();
    let mut state = SimpleState::new(&pulsed_plan).unwrap();

    let mut got: ArrayD<f32> = ArrayD::zeros(&*initial_output_shape);
    let mut output_len = None;

    debug!("Run pulsing model");
    //dbg!(pulsed_plan.model());
    let mut written = 0;
    loop {
        let to_write_in_chunk = pulse.min(input_array.shape()[axis].saturating_sub(written));
        let mut chunk: ArrayD<f32> = input_array
            .slice_axis(Axis(axis), (written..written + to_write_in_chunk).into())
            .to_owned();
        written += to_write_in_chunk;
        if to_write_in_chunk < pulse {
            let mut filler_shape = input_array.shape().to_vec();
            filler_shape[axis] = pulse - to_write_in_chunk;
            chunk = tract_ndarray::concatenate(
                Axis(axis),
                &[chunk.view(), ArrayD::from_elem(filler_shape, f32::NAN).view()],
            )
            .unwrap();
            state.session_state.resolved_symbols.set(&s, written as i64);
            output_len = stream_info
                .dim
                .eval(&state.session_state.resolved_symbols)
                .to_isize()
                .ok()
                .map(|n| n.max(0) as usize);
        }
        let mut outputs = state.run(tvec!(chunk.into_tensor().into_tvalue())).unwrap();
        got = tract_ndarray::concatenate(
            Axis(output_stream_axis),
            &[got.view(), outputs.remove(0).to_array_view::<f32>().unwrap()],
        )
        .unwrap();
        eprintln!("GOT: {got}");
        if let Some(output_len) = output_len {
            if got.shape()[output_stream_axis] >= output_len.max(0) + delay {
                break;
            }
        }
        eprintln!();
    }

    let pulsed_output = got
        .slice_axis(
            Axis(output_stream_axis),
            (stream_info.delay..stream_info.delay + output_len.unwrap().max(0)).into(),
        )
        .to_owned()
        .into_tensor();

    prop_assert!(
        &pulsed_output.close_enough(&outputs[0], true).is_ok(),
        "{:?} == {:?}",
        pulsed_output,
        outputs[0]
    );
    Ok(())
}

proptest! {
    #[test]
    fn proptest_crop(pulse in 1i32..3, input_len in 0i32..10, begin in 0i32..3, end in 0i32..3) {
        let full_len = input_len + begin + end;
        let mut model = TypedModel::default();
        let s = model.symbols.sym("S");
        let a = model.add_source("a", f32::fact(&[s])).unwrap();
        let slice = model.wire_node("slice", Slice::new(0, begin as usize, (input_len + begin) as usize), &[a]).unwrap();
        model.set_output_outlets(&slice).unwrap();

        let input = Array1::range(1.0f32, full_len as f32 + 1.0, 1.0);
        proptest_regular_against_pulse(model, pulse as _, input.into_dyn(), 0)?;
    }

    #[test]
    fn proptest_pad(pulse in 1i32..3, input_len in 0i32..10, begin in 0i32..3, end in 0i32..3) {
        let mut model = TypedModel::default();
        let s = model.symbols.sym("S");
        let a = model.add_source("a", f32::fact(&[s])).unwrap();
        let pad = model.wire_node("pad", Pad::new(vec![(begin as _, end as _)],
        PadMode::Constant(Arc::new(Tensor::from(-1f32)))), &[a]).unwrap();
        model.set_output_outlets(&pad).unwrap();

        let input = Array1::range(1.0f32, input_len as f32 + 1.0, 1.0);
        proptest_regular_against_pulse(model, pulse as _, input.into_dyn(), 0)?;
    }
}

fn vec(len: impl Strategy<Value = usize>) -> impl Strategy<Value = Vec<f32>> {
    len.prop_flat_map(|l| proptest::collection::vec(-5..5, l..=l))
        .prop_map(|v| v.into_iter().map(|f| f as f32).collect())
}

#[test]
fn test_simple_conv() {
    let mut model = TypedModel::default();
    let kernel = rctensor3(&[[[0.5f32, 1.0, -0.1]]]);
    let s = model.symbols.sym("S");
    let a = model.add_source("a", f32::fact(dims!(1, 1, s))).unwrap();
    let kernel = model.add_const("kernel", kernel).unwrap();
    let bias = model.add_const("bias", tensor0(0f32)).unwrap();

    model
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
    model.auto_outputs().unwrap();

    let input = arr3(&[[[1.0f32, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0]]]);
    proptest_regular_against_pulse(model, 4, input.into_dyn(), 2).unwrap();
}

#[test]
fn test_pad_before_1() {
    let mut model = TypedModel::default();
    let s = model.symbols.sym("S");
    let a = model.add_source("a", f32::fact(&[s])).unwrap();
    model
        .wire_node(
            "pad",
            Pad::new(vec![(1, 0)], PadMode::Constant(Arc::new(Tensor::from(-1f32)))),
            &[a],
        )
        .unwrap();
    model.auto_outputs().unwrap();

    let input = arr1(&[1.0]);
    proptest_regular_against_pulse(model, 1, input.into_dyn(), 0).unwrap();
}

#[test]
fn test_pad_before_2() {
    let mut model = TypedModel::default();
    let s = model.symbols.sym("S");
    let a = model.add_source("a", f32::fact(&[s])).unwrap();
    model
        .wire_node(
            "pad",
            Pad::new(vec![(1, 0)], PadMode::Constant(Arc::new(Tensor::from(-1f32)))),
            &[a],
        )
        .unwrap();
    model.auto_outputs().unwrap();

    let input = arr1(&[1.0, 2.0]);
    proptest_regular_against_pulse(model, 2, input.into_dyn(), 0).unwrap();
}
