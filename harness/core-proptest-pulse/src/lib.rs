#![cfg(test)]

use proptest::prelude::*;
use proptest::proptest;
use proptest::test_runner::TestCaseResult;
use proptest::*;
use tract_core::dimfact;
use tract_core::internal::*;
use tract_core::ndarray::*;
use tract_core::shapefact;

mod conv_plus_conv;
mod pad_plus_conv;

fn proptest_regular_against_pulse(
    model: InferenceModel,
    pulse: usize,
    input_array: ArrayD<f32>,
    axis: usize,
) -> TestCaseResult {
    let mut ref_model = model.clone();
    ref_model.set_input_fact(0, TensorFact::dt_shape(f32::datum_type(), input_array.shape()))?;
    let input = Tensor::from(input_array.clone());
    let plan = SimplePlan::new(&ref_model).unwrap();
    let outputs = plan.run(tvec!(input.clone())).unwrap();

    let model = model.into_normalized().unwrap();

    let pulsed = PulsedModel::new(&model, pulse).unwrap();
    let output_fact = pulsed.output_fact(0).unwrap().clone();

    let output_stream_axis = output_fact.axis;
    let delay = output_fact.delay;
    let mut initial_output_shape = output_fact.shape.clone();
    initial_output_shape[output_stream_axis] = 0;

    let pulsed_plan = SimplePlan::new(pulsed).unwrap();
    let mut state = SimpleState::new(&pulsed_plan).unwrap();

    let mut got: ArrayD<f32> = ArrayD::zeros(&*initial_output_shape);
    let mut output_len = None;

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
            chunk = stack(
                Axis(axis),
                &[chunk.view(), ArrayD::from_elem(filler_shape, std::f32::NAN).view()],
            )
            .unwrap();
            output_len = output_fact.dim.eval(written as _);
            state.session_state.known_stream_len = Some(written)
        }
        let mut outputs = state.run(tvec!(Tensor::from(chunk.to_owned()).into())).unwrap();
        got = stack(
            Axis(output_stream_axis),
            &[got.view(), outputs.remove(0).to_array_view::<f32>().unwrap()],
        )
        .unwrap();
        if let Some(output_len) = output_len {
            if got.shape()[output_stream_axis] >= output_len as usize + delay {
                break;
            }
        }
    }

    let pulsed_output = got.slice_axis(
        Axis(output_stream_axis),
        (output_fact.delay..output_fact.delay + output_len.unwrap() as usize).into(),
    );

    prop_assert_eq!(pulsed_output, outputs[0].to_array_view::<f32>().unwrap());
    Ok(())
}

proptest! {
    #[test]
    fn proptest_crop(pulse in 1i32..3, input_len in 0i32..10, begin in 0i32..3, end in 0i32..3) {
        use tract_core::ops::array::Slice;
        let full_len = input_len + begin + end;
        let mut model = InferenceModel::default();
        let _ = model
            .add_source("a", TensorFact::dt_shape(f32::datum_type(), shapefact!(S)))
            .unwrap();
        model.chain_default("slice", Slice::new(0, begin as usize, (input_len + begin) as usize)).unwrap();
        model.auto_outputs().unwrap();

        let input = Array1::range(1.0f32, full_len as f32 + 1.0, 1.0);
        proptest_regular_against_pulse(model, pulse as _, input.into_dyn(), 0)?;
    }

    #[test]
    fn proptest_pad(pulse in 1i32..3, input_len in 0i32..10, begin in 0i32..3, end in 0i32..3) {
        use tract_core::ops::array::{ Pad, PadMode };
        let mut model = InferenceModel::default();
        let _ = model
            .add_source("a", TensorFact::dt_shape(f32::datum_type(), shapefact!(S)))
            .unwrap();
        model.chain_default("pad", Pad::new(vec![(begin as _, end as _)],
            PadMode::Constant(Arc::new(Tensor::from(-1f32))))).unwrap();
        model.auto_outputs().unwrap();

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
    use tract_core::ops::cnn::*;

    let mut model = InferenceModel::default();
    let ker = model.add_const("kernel", tensor3(&[[[0.5f32, 1.0, -0.1]]])).unwrap();
    let _ = model
        .add_source("a", TensorFact::dt_shape(f32::datum_type(), shapefact!(1, 1, S))) // NCT
        .unwrap();

    let conv = model.chain_default("conv", Conv::default()).unwrap();
    model.add_edge(ker, InletId::new(conv, 1)).unwrap();
    model.auto_outputs().unwrap();

    let input = arr3(&[[[1.0f32, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0]]]);
    proptest_regular_against_pulse(model, 4, input.into_dyn(), 2).unwrap();
}

#[test]
fn test_crop_after_1() {
    use tract_core::ops::array::Slice;
    let mut model = InferenceModel::default();
    let _ = model.add_source("a", TensorFact::dt_shape(f32::datum_type(), shapefact!(S))).unwrap();
    model.chain_default("slice", Slice::new(0, 0, 0)).unwrap();
    model.auto_outputs().unwrap();

    let input = arr1(&[1.0]);
    proptest_regular_against_pulse(model, 1, input.into_dyn(), 0).unwrap();
}

#[test]
fn test_pad_after_1() {
    use tract_core::ops::array::{Pad, PadMode};
    let mut model = InferenceModel::default();
    let _ = model.add_source("a", TensorFact::dt_shape(f32::datum_type(), shapefact!(S))).unwrap();
    model
        .chain_default(
            "pad",
            Pad::new(vec![(0, 1)], PadMode::Constant(Arc::new(Tensor::from(-1f32)))),
        )
        .unwrap();
    model.auto_outputs().unwrap();

    let input = arr1(&[]);
    proptest_regular_against_pulse(model, 1, input.into_dyn(), 0).unwrap();
}

#[test]
fn test_pad_before_1() {
    use tract_core::ops::array::{Pad, PadMode};
    let mut model = InferenceModel::default();
    let _ = model.add_source("a", TensorFact::dt_shape(f32::datum_type(), shapefact!(S))).unwrap();
    model
        .chain_default(
            "pad",
            Pad::new(vec![(1, 0)], PadMode::Constant(Arc::new(Tensor::from(-1f32)))),
        )
        .unwrap();
    model.auto_outputs().unwrap();

    let input = arr1(&[1.0]);
    proptest_regular_against_pulse(model, 1, input.into_dyn(), 0).unwrap();
}

#[test]
fn test_pad_before_2() {
    use tract_core::ops::array::{Pad, PadMode};
    let mut model = InferenceModel::default();
    let _ = model.add_source("a", TensorFact::dt_shape(f32::datum_type(), shapefact!(S))).unwrap();
    model
        .chain_default(
            "pad",
            Pad::new(vec![(1, 0)], PadMode::Constant(Arc::new(Tensor::from(-1f32)))),
        )
        .unwrap();
    model.auto_outputs().unwrap();

    let input = arr1(&[1.0, 2.0]);
    proptest_regular_against_pulse(model, 2, input.into_dyn(), 0).unwrap();
}
