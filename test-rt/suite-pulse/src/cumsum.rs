use infra::{Test, TestSuite};
use proptest::prelude::*;
use tract_core::internal::*;
use tract_core::ndarray::ArrayD;
use tract_core::ops::array::CumSum;
use tract_core::ops::array::Slice;
use tract_core::runtime::Runtime;

use crate::{pulse_and_compare, values};

#[derive(Debug, Clone, Default)]
pub struct CumSumProblem {
    pub input: Vec<f32>,
    pub channels: usize,
    pub pulse: usize,
    pub delay: usize,
    pub axis: usize,
    pub exclusive: bool,
}

impl Arbitrary for CumSumProblem {
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with(_: Self::Parameters) -> BoxedStrategy<Self> {
        (1usize..30, 1usize..4, 1usize..6, 0usize..8, 1usize..3, any::<bool>())
            .prop_flat_map(|(frames, channels, pulse, delay, axis, exclusive)| {
                (
                    values(Just((frames + delay) * channels)),
                    Just(channels),
                    Just(pulse),
                    Just(delay),
                    Just(axis),
                    Just(exclusive),
                )
            })
            .prop_map(|(input, channels, pulse, delay, axis, exclusive)| CumSumProblem {
                input,
                channels,
                pulse,
                delay,
                axis,
                exclusive,
            })
            .boxed()
    }
}

impl Test for CumSumProblem {
    fn run_with_approx(
        &self,
        _id: &str,
        runtime: &dyn Runtime,
        approx: Approximation,
    ) -> infra::TestResult {
        let mut model = TypedModel::default();
        let s = model.symbols.sym("S");
        let a = model.add_source("a", f32::fact(dims!(1, s, self.channels)))?;
        let crop = model.wire_node("delay", Slice::new(1, self.delay, s), &[a])?;
        let cumsum = model.wire_node(
            "cumsum",
            CumSum { axis: self.axis, exclusive: self.exclusive, reverse: false },
            &crop,
        )?;
        model.select_output_outlets(&cumsum)?;
        let frames = self.input.len() / self.channels;
        let input = ArrayD::from_shape_vec(vec![1, frames, self.channels], self.input.clone())?;
        pulse_and_compare(runtime, approx, model, self.pulse, input, 1)
    }
}

pub fn suite() -> TractResult<TestSuite> {
    let mut suite = TestSuite::default();
    suite.add_arbitrary::<CumSumProblem>("proptest", ());
    let case = |frames: usize, channels: usize, pulse, delay, axis, exclusive| CumSumProblem {
        input: (0..frames * channels).map(|x| (x % 7) as f32 - 3.0).collect(),
        channels,
        pulse,
        delay,
        axis,
        exclusive,
    };
    suite.add_test("stream_axis", case(9, 2, 4, 0, 1, false));
    suite.add_test("stream_axis_delayed", case(9, 2, 4, 3, 1, false));
    suite.add_test("stream_axis_exclusive_delayed", case(9, 1, 2, 5, 1, true));
    suite.add_test("delay_beyond_first_pulse", case(10, 1, 2, 7, 1, false));
    suite.add_test("channel_axis", case(9, 3, 4, 2, 2, false));
    Ok(suite)
}
