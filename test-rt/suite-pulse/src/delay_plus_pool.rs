use infra::{Test, TestSuite};
use proptest::prelude::*;
use tract_core::internal::*;
use tract_core::ndarray::arr1;
use tract_core::ops::array::Slice;
use tract_core::ops::cnn::{MaxPool, PaddingSpec, PoolSpec};
use tract_core::ops::nn::DataFormat;
use tract_core::runtime::Runtime;

use crate::pulse_and_compare;

#[derive(Debug, Clone)]
pub struct DelayPlusPoolProblem {
    pub input: Vec<f32>,
    pub pulse: usize,
    pub delay: usize,
    pub stride: usize,
    pub pool_window: usize,
    pub padding: PaddingSpec,
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
                    crate::values(min_input..min_input + 10),
                    prop_oneof![Just(PaddingSpec::Valid), explicit],
                )
            })
            .prop_map(|(pool_window, factor, delay, stride, input, padding)| DelayPlusPoolProblem {
                input,
                pulse: factor * stride,
                delay,
                stride,
                pool_window,
                padding,
            })
            .boxed()
    }
}

impl Test for DelayPlusPoolProblem {
    fn run_with_approx(
        &self,
        _id: &str,
        runtime: &dyn Runtime,
        approx: Approximation,
    ) -> infra::TestResult {
        let mut model = TypedModel::default();
        let s = model.symbols.sym("S");
        let a = model.add_source("a", f32::fact(dims!(1, s, 1)))?;
        let crop = model.wire_node("delay", Slice::new(1, self.delay, s), &[a])?;
        let pool_spec = PoolSpec::new(
            DataFormat::NHWC,
            tvec!(self.pool_window),
            self.padding.clone(),
            None,
            Some(tvec!(self.stride)),
            1,
            1,
        );
        let pool = model.wire_node("pool", MaxPool::new(pool_spec, None), &crop)?;
        model.select_output_outlets(&pool)?;
        let input = arr1(&self.input).into_shape_with_order((1, self.input.len(), 1))?.into_dyn();
        pulse_and_compare(runtime, approx, model, self.pulse, input, 1)
    }
}

pub fn suite() -> TractResult<TestSuite> {
    let mut suite = TestSuite::default();
    suite.add_arbitrary::<DelayPlusPoolProblem>("proptest", ());
    let case = |input: &[f32], pulse, delay, stride, pool_window, padding| DelayPlusPoolProblem {
        input: input.to_vec(),
        pulse,
        delay,
        stride,
        pool_window,
        padding,
    };
    let valid = PaddingSpec::Valid;
    let pad_right = PaddingSpec::ExplicitOnnxPool(tvec!(0), tvec!(1), false);
    suite.add_test("basic", case(&[0., 0., 0., 0., 1.], 2, 0, 1, 2, valid.clone()));
    suite.add_test("stride", case(&[0., 0., 0.], 2, 0, 2, 1, valid.clone()));
    suite.add_test("misaligned_stride", case(&[0., 1.], 2, 1, 2, 1, valid.clone()));
    suite.add_test("overlap", case(&[0., 1.], 1, 0, 1, 2, valid.clone()));
    suite.add_test(
        "overlap_realign",
        case(&[f32::NAN, 2., 3., 4., 5., 6.], 2, 1, 2, 3, valid.clone()),
    );
    suite.add_test("long_overlap_1", case(&[0., 0., 0.], 1, 0, 1, 3, valid.clone()));
    suite.add_test("long_overlap_2", case(&[0., 0., 0., 0.], 1, 2, 1, 2, valid.clone()));
    suite.add_test("long_overlap_3", case(&[-1., -1., 0.], 2, 0, 2, 3, valid));
    suite.add_test("pad_right", case(&[0., 0.], 1, 0, 1, 2, pad_right.clone()));
    suite.add_test("pad_right_2", case(&[f32::NAN, 0., 1.], 2, 1, 2, 2, pad_right));
    Ok(suite)
}
