use proptest::proptest;
use proptest::test_runner::TestCaseResult;
use tract_core::dimfact;
use tract_core::hir::array;
use tract_core::infer::*;
use tract_core::internal::*;
use tract_core::ndarray::*;
use tract_core::ops::{cnn, nn};
use tract_core::shapefactoid;

use super::*;

#[derive(Debug, Clone)]
struct DelayPlusPoolProblem {
    input: Vec<f32>,
    pulse: usize,
    delay: usize,
    stride: usize,
    pool_window: usize,
}

impl Arbitrary for DelayPlusPoolProblem {
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with(_: Self::Parameters) -> BoxedStrategy<Self> {
        (1usize..4, 1usize..4, 0usize..5, 1usize..4)
            .prop_flat_map(|(pool_window, factor, delay, stride)| {
                let min_input = delay + pool_window;
                (
                    Just(pool_window),
                    Just(factor),
                    Just(delay),
                    Just(stride),
                    vec(min_input..min_input + 10),
                )
            })
            .prop_map(|(pool_window, factor, delay, stride, input)| {
                let pulse = factor * stride;
                DelayPlusPoolProblem { input, pulse, delay, stride, pool_window }
            })
            .boxed()
    }
}

impl DelayPlusPoolProblem {
    pub fn run(&self) -> TestCaseResult {
        let mut model = InferenceModel::default();
        let a = model
            .add_source("a", InferenceFact::dt_shape(f32::datum_type(), shapefactoid!(1, S, 1)))
            .unwrap();
        let crop = model.wire_node("crop", array::Crop::new(1, self.delay, 0), &[a]).unwrap();
        let pool_spec = cnn::PoolSpec::new(
            nn::DataFormat::NHWC,
            tvec!(self.pool_window),
            cnn::PaddingSpec::Valid,
            None,
            Some(tvec!(self.stride)),
            None,
        );
        let pool = model.wire_node("pool", cnn::MaxPool::new(pool_spec, None), &crop).unwrap();
        model.set_output_outlets(&pool).unwrap();
        let input = arr1(&self.input).into_shape((1, self.input.len(), 1)).unwrap().into_dyn();
        proptest_regular_against_pulse(model, self.pulse as _, input, 1)
    }
}

proptest! {
    #[test]
    fn proptest(pb in DelayPlusPoolProblem::arbitrary()) { pb.run().unwrap() }
}

#[test]
fn test_basic() {
    DelayPlusPoolProblem {
        input: vec![0.0, 0.0, 0.0, 0.0, 1.0],
        pulse: 2,
        delay: 0,
        stride: 1,
        pool_window: 2,
    }
    .run()
    .unwrap()
}

#[test]
fn test_stride() {
    DelayPlusPoolProblem {
        input: vec![0.0, 0.0, 0.0],
        pulse: 2,
        delay: 0,
        stride: 2,
        pool_window: 1,
    }
    .run()
    .unwrap()
}

#[test]
fn test_misaligned_stride() {
    DelayPlusPoolProblem { input: vec![0.0, 1.0], pulse: 2, delay: 1, stride: 2, pool_window: 1 }
        .run()
        .unwrap()
}

#[test]
fn test_overlap() {
    DelayPlusPoolProblem { input: vec![0.0, 1.0], pulse: 1, delay: 0, stride: 1, pool_window: 2 }
        .run()
        .unwrap()
}

#[test]
fn test_overlap_realign() {
    DelayPlusPoolProblem {
        input: vec![4.0, 0.0, 0.0, 0.0],
        pulse: 2,
        delay: 1,
        stride: 2,
        pool_window: 3,
    }
    .run()
    .unwrap()
}

#[test]
fn test_long_overlap_1() {
    DelayPlusPoolProblem {
        input: vec![0.0, 0.0, 0.0],
        pulse: 1,
        delay: 0,
        stride: 1,
        pool_window: 3,
    }
    .run()
    .unwrap()
}

#[test]
fn test_long_overlap_2() {
    DelayPlusPoolProblem {
        input: vec![0.0, 0.0, 0.0, 0.0],
        pulse: 1,
        delay: 2,
        stride: 1,
        pool_window: 2,
    }
    .run()
    .unwrap()
}

#[test]
fn test_long_overlap_3() {
    DelayPlusPoolProblem {
        input: vec![-1.0, -1.0, 0.0],
        pulse: 2,
        delay: 0,
        stride: 2,
        pool_window: 3,
    }
    .run()
    .unwrap()
}
