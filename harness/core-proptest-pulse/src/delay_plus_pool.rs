use proptest::proptest;
use proptest::test_runner::TestCaseResult;
use tract_core::dimfact;
use tract_core::internal::*;
use tract_core::ndarray::*;
use tract_core::ops::{cnn, nn};
use tract_core::shapefact;

use super::*;

#[derive(Debug, Clone)]
struct DelayPlusPoolProblem {
    input: Vec<f32>,
    pulse: usize,
    delay: usize,
    pool_window: usize,
}

impl Arbitrary for DelayPlusPoolProblem {
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with(_: Self::Parameters) -> BoxedStrategy<Self> {
        (1usize..4, 1usize..4, 0usize..5, vec(5usize..25))
            .prop_map(|(pool_window, factor, delay, input)| {
                let pulse = factor * pool_window;
                DelayPlusPoolProblem { input, pulse, delay, pool_window }
            })
            .boxed()
    }
}

impl DelayPlusPoolProblem {
    pub fn run(&self) -> TestCaseResult {
        let mut model = InferenceModel::default();
        let a = model
            .add_source("a", InferenceFact::dt_shape(f32::datum_type(), shapefact!(1, S, 1)))
            .unwrap();
        let pool_spec = cnn::PoolSpec::new(
            nn::DataFormat::NHWC,
            tvec!(self.pool_window),
            cnn::PaddingSpec::Valid,
            None,
        );
        let pool = model.wire_node("pool", cnn::MaxPool::new(pool_spec, None), &[a]);
        model.auto_outputs().unwrap();
        let input = arr1(&self.input).into_shape((1, self.input.len(), 1)).unwrap().into_dyn();
        proptest_regular_against_pulse(model, self.pulse as _, input, 1)
    }
}

proptest! {
    #[test]
    fn proptest(pb in DelayPlusPoolProblem::arbitrary()) { pb.run().unwrap() }
}

#[test]
fn basic() {
    DelayPlusPoolProblem {
        input: vec![0.0, 0.0, 0.0, 0.0, 1.0],
        pulse: 2,
        delay: 0,
        pool_window: 2,
    }
    .run()
    .unwrap()
}
