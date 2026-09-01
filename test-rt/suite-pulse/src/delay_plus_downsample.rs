use infra::{Test, TestSuite};
use proptest::prelude::*;
use tract_core::internal::*;
use tract_core::ndarray::{ArrayD, arr1};
use tract_core::ops::Downsample;
use tract_core::ops::array::Slice;
use tract_core::runtime::Runtime;
use tract_core::tract_data::itertools::Itertools;

use crate::pulse_and_compare;

#[derive(Debug, Clone, Default)]
pub struct DelayPlusDownsampleProblem {
    pub input: usize,
    pub pulse: usize,
    pub delay: usize,
    pub stride: usize,
    pub modulo: usize,
}

impl Arbitrary for DelayPlusDownsampleProblem {
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with(_: Self::Parameters) -> BoxedStrategy<Self> {
        (1usize..100, 1usize..4, 0usize..100, 1usize..4)
            .prop_flat_map(|(input, pulse_mul, delay, stride)| {
                (
                    Just(input + stride + delay),
                    Just(pulse_mul * stride),
                    Just(delay),
                    Just(stride),
                    0..stride,
                )
            })
            .prop_map(|(input, pulse, delay, stride, modulo)| DelayPlusDownsampleProblem {
                input,
                pulse,
                delay,
                stride,
                modulo,
            })
            .boxed()
    }
}

fn ramp(n: usize) -> ArrayD<f32> {
    arr1(&(0..n).map(|x| x as f32).collect_vec()).into_shape_with_order(vec![1, n, 1]).unwrap()
}

impl Test for DelayPlusDownsampleProblem {
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
        let ds = model.wire_node(
            "ds",
            Downsample { axis: 1, stride: self.stride as isize, modulo: self.modulo },
            &crop,
        )?;
        model.select_output_outlets(&ds)?;
        pulse_and_compare(runtime, approx, model, self.pulse, ramp(self.input), 1)
    }
}

pub fn suite() -> TractResult<TestSuite> {
    let mut suite = TestSuite::default();
    suite.add_arbitrary::<DelayPlusDownsampleProblem>("proptest", ());
    let case = |input, pulse, delay, stride, modulo| DelayPlusDownsampleProblem {
        input,
        pulse,
        delay,
        stride,
        modulo,
    };
    suite.add_test("modulo", case(3, 2, 0, 2, 1));
    suite.add_test("delay", case(3, 2, 1, 2, 0));
    suite.add_test("from_convs", case(5, 2, 1, 2, 0));
    suite.add_test("delayed_stride", case(9, 2, 1, 2, 0));
    suite.add_test("big_delay", case(6, 1, 4, 1, 0));
    suite.add_test("huge_delay", case(4, 2, 1, 2, 0));
    Ok(suite)
}
