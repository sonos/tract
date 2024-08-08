use proptest::proptest;
use proptest::test_runner::TestCaseResult;
use tract_core::ops::cnn::MaxPool;

use super::*;

#[derive(Debug, Clone)]
struct DelayPlusPoolProblem {
    input: Vec<f32>,
    pulse: usize,
    delay: usize,
    stride: usize,
    pool_window: usize,
    padding: PaddingSpec,
}

impl Arbitrary for DelayPlusPoolProblem {
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with(_: Self::Parameters) -> BoxedStrategy<Self> {
        (1usize..4, 1usize..4, 0usize..5, 1usize..4)
            .prop_flat_map(|(pool_window, factor, delay, stride)| {
                let padding = pool_window - 1;
                let explicit = (0..=padding).prop_map(move |right| {
                    PaddingSpec::ExplicitOnnxPool(tvec!(padding - right), tvec!(right), false)
                });
                let min_input = delay + pool_window;
                (
                    Just(pool_window),
                    Just(factor),
                    Just(delay),
                    Just(stride),
                    vec(min_input..min_input + 10),
                    prop_oneof![Just(PaddingSpec::Valid), explicit],
                )
            })
            .prop_map(|(pool_window, factor, delay, stride, input, padding)| {
                let pulse = factor * stride;
                DelayPlusPoolProblem { input, pulse, delay, stride, pool_window, padding }
            })
            .boxed()
    }
}

impl DelayPlusPoolProblem {
    pub fn run(&self) -> TestCaseResult {
        let mut model = TypedModel::default();
        let s = model.symbols.sym("S");
        let a = model.add_source("a", f32::fact(dims!(1, s, 1))).unwrap();
        let crop = model.wire_node("delay", Slice::new(1, self.delay, s), &[a]).unwrap();
        let pool_spec = PoolSpec::new(
            DataFormat::NHWC,
            tvec!(self.pool_window),
            self.padding.clone(),
            None,
            Some(tvec!(self.stride)),
            1,
            1,
        );
        let pool = model.wire_node("pool", MaxPool::new(pool_spec, None), &crop).unwrap();
        model.set_output_outlets(&pool).unwrap();
        let input = arr1(&self.input).into_shape_with_order((1, self.input.len(), 1)).unwrap().into_dyn();
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
        padding: PaddingSpec::Valid,
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
        padding: PaddingSpec::Valid,
    }
    .run()
    .unwrap()
}

#[test]
fn test_misaligned_stride() {
    DelayPlusPoolProblem {
        input: vec![0.0, 1.0],
        pulse: 2,
        delay: 1,
        stride: 2,
        pool_window: 1,
        padding: PaddingSpec::Valid,
    }
    .run()
    .unwrap()
}

#[test]
fn test_overlap() {
    DelayPlusPoolProblem {
        input: vec![0.0, 1.0],
        pulse: 1,
        delay: 0,
        stride: 1,
        pool_window: 2,
        padding: PaddingSpec::Valid,
    }
    .run()
    .unwrap()
}

#[test]
fn test_overlap_realign() {
    DelayPlusPoolProblem {
        input: vec![f32::NAN, 2.0, 3.0, 4.0, 5.0, 6.0],
        pulse: 2,
        delay: 1,
        stride: 2,
        pool_window: 3,
        padding: PaddingSpec::Valid,
    }
    .run()
    .unwrap();
}

#[test]
fn test_long_overlap_1() {
    DelayPlusPoolProblem {
        input: vec![0.0, 0.0, 0.0],
        pulse: 1,
        delay: 0,
        stride: 1,
        pool_window: 3,
        padding: PaddingSpec::Valid,
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
        padding: PaddingSpec::Valid,
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
        padding: PaddingSpec::Valid,
    }
    .run()
    .unwrap()
}

#[test]
fn test_pad_right() {
    DelayPlusPoolProblem {
        input: vec![0.0, 0.0],
        pulse: 1,
        delay: 0,
        stride: 1,
        pool_window: 2,
        padding: PaddingSpec::ExplicitOnnxPool(tvec!(0), tvec!(1), false),
    }
    .run()
    .unwrap()
}

#[test]
fn test_pad_right_2() {
    DelayPlusPoolProblem {
        input: vec![f32::NAN, 0.0, 1.0],
        pulse: 2,
        delay: 1,
        stride: 2,
        pool_window: 2,
        padding: PaddingSpec::ExplicitOnnxPool(tvec!(0), tvec!(1), false),
    }
    .run()
    .unwrap()
}
