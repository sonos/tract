#![cfg(test)]

#[macro_use]
extern crate log;

use proptest::prelude::*;
use proptest::proptest;
use proptest::test_runner::TestCaseResult;
use proptest::*;
use tract_hir::internal::*;
use tract_ndarray::*;
use tract_pulse::internal::*;

mod conv_plus_conv;
mod delay_plus_pool;
mod pad_plus_conv;

#[allow(dead_code)]
fn setup_test_logger() {
    let _ = env_logger::Builder::from_env("TRACT_LOG").try_init();
}

fn proptest_regular_against_pulse(
    model: InferenceModel,
    pulse: usize,
    input_array: ArrayD<f32>,
    axis: usize,
) -> TestCaseResult {
    setup_test_logger();
    let mut ref_model = model.clone();
    let s = stream_symbol();
    debug!("Run reference");
    ref_model
        .set_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), input_array.shape()))
        .unwrap();
    let input = Tensor::from(input_array.clone());
    let plan = SimplePlan::new(&ref_model).unwrap();
    let outputs = plan.run(tvec!(input.clone())).unwrap();

    debug!("Build pulsing model");
    let model = model.into_typed().unwrap();
    dbg!(&model);
    let pulsed = PulsedModel::new(&model, pulse).unwrap();
    dbg!(&pulsed);
    let output_fact = pulsed.output_fact(0).unwrap().clone();

    let output_stream_axis = output_fact.axis;
    let delay = output_fact.delay;
    let mut initial_output_shape = output_fact.shape.clone();
    initial_output_shape[output_stream_axis] = 0.to_dim();
    let initial_output_shape: TVec<usize> =
        initial_output_shape.iter().map(|d| d.to_usize().unwrap()).collect();

    let pulsed_plan = SimplePlan::new(pulsed).unwrap();
    let mut state = SimpleState::new(&pulsed_plan).unwrap();

    let mut got: ArrayD<f32> = ArrayD::zeros(&*initial_output_shape);
    let mut output_len = None;

    debug!("Run pulsing model");
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
            state.session_state.resolved_symbols[s] = Some(written as i64);
            output_len = output_fact
                .dim
                .eval(&state.session_state.resolved_symbols)
                .to_isize()
                .ok()
                .map(|n| n.max(0) as usize);
        }
        let mut outputs = state.run(tvec!(Tensor::from(chunk.to_owned()).into())).unwrap();
        got = stack(
            Axis(output_stream_axis),
            &[got.view(), outputs.remove(0).to_array_view::<f32>().unwrap()],
        )
        .unwrap();
        if let Some(output_len) = output_len {
            if got.shape()[output_stream_axis] >= output_len.max(0) as usize + delay {
                break;
            }
        }
    }

    let pulsed_output = got
        .slice_axis(
            Axis(output_stream_axis),
            (output_fact.delay..output_fact.delay + output_len.unwrap().max(0) as usize).into(),
        )
        .to_owned()
        .into_tensor();

    prop_assert!(
        &pulsed_output.close_enough(&*outputs[0], true).is_ok(),
        "{:?} == {:?}",
        pulsed_output,
        outputs[0]
    );
    Ok(())
}

proptest! {
    #[test]
    fn proptest_crop(pulse in 1i32..3, input_len in 0i32..10, begin in 0i32..3, end in 0i32..3) {
        use tract_hir::ops::array::Slice;
        let full_len = input_len + begin + end;
        let mut model = InferenceModel::default();
        let a = model
            .add_source("a", InferenceFact::dt_shape(f32::datum_type(), shapefactoid!(S)))
            .unwrap();
        let slice = model.wire_node("slice", Slice::new(0, begin as usize, (input_len + begin) as usize), &[a]).unwrap();
        model.set_output_outlets(&slice).unwrap();

        let input = Array1::range(1.0f32, full_len as f32 + 1.0, 1.0);
        proptest_regular_against_pulse(model, pulse as _, input.into_dyn(), 0)?;
    }

    #[test]
    fn proptest_pad(pulse in 1i32..3, input_len in 0i32..10, begin in 0i32..3, end in 0i32..3) {
        use tract_hir::ops::array::{ Pad, PadMode };
        let mut model = InferenceModel::default();
        let a = model
            .add_source("a", InferenceFact::dt_shape(f32::datum_type(), shapefactoid!(S)))
            .unwrap();
        let pad = model.wire_node("pad",Pad::new(vec![(begin as _, end as _)],
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
    use tract_hir::ops::cnn::*;

    let mut model = InferenceModel::default();
    let ker = model.add_const("kernel", tensor3(&[[[0.5f32, 1.0, -0.1]]])).unwrap();
    let a = model
        .add_source("a", InferenceFact::dt_shape(f32::datum_type(), shapefactoid!(1, 1, S))) // NCT
        .unwrap();

    model.wire_node("conv", expand(Conv::default()), &[a, ker]).unwrap();
    model.auto_outputs().unwrap();

    let input = arr3(&[[[1.0f32, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0]]]);
    proptest_regular_against_pulse(model, 4, input.into_dyn(), 2).unwrap();
}

#[test]
fn test_crop_after_1() {
    use tract_hir::ops::array::Slice;
    let mut model = InferenceModel::default();
    let a = model
        .add_source("a", InferenceFact::dt_shape(f32::datum_type(), shapefactoid!(S)))
        .unwrap();
    model.wire_node("slice", Slice::new(0, 0, 0), &[a]).unwrap();
    model.auto_outputs().unwrap();

    let input = arr1(&[1.0]);
    proptest_regular_against_pulse(model, 1, input.into_dyn(), 0).unwrap();
}

#[test]
fn test_pad_after_1() {
    use tract_hir::ops::array::{Pad, PadMode};
    let mut model = InferenceModel::default();
    let a = model
        .add_source("a", InferenceFact::dt_shape(f32::datum_type(), shapefactoid!(S)))
        .unwrap();
    model
        .wire_node(
            "pad",
            Pad::new(vec![(0, 1)], PadMode::Constant(Arc::new(Tensor::from(-1f32)))),
            &[a],
        )
        .unwrap();
    model.auto_outputs().unwrap();

    let input = arr1(&[]);
    proptest_regular_against_pulse(model, 1, input.into_dyn(), 0).unwrap();
}

#[test]
fn test_pad_before_1() {
    use tract_hir::ops::array::{Pad, PadMode};
    let mut model = InferenceModel::default();
    let a = model
        .add_source("a", InferenceFact::dt_shape(f32::datum_type(), shapefactoid!(S)))
        .unwrap();
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
    use tract_hir::ops::array::{Pad, PadMode};
    let mut model = InferenceModel::default();
    let a = model
        .add_source("a", InferenceFact::dt_shape(f32::datum_type(), shapefactoid!(S)))
        .unwrap();
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
