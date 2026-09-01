use infra::{Test, TestSuite};
use proptest::prelude::*;
use tract_core::internal::*;
use tract_core::ndarray::Array1;
use tract_core::ops::array::{Pad, PadMode};
use tract_core::runtime::Runtime;

use crate::pulse_and_compare;

/// A Pad on the streaming axis: `before` frames of constant become output
/// delay, `after` frames extend the stream past its end.
#[derive(Clone, Debug)]
pub struct PadProblem {
    pub pulse: usize,
    pub input_len: usize,
    pub before: usize,
    pub after: usize,
}

impl Arbitrary for PadProblem {
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with(_: Self::Parameters) -> BoxedStrategy<Self> {
        (1usize..3, 0usize..10, 0usize..3, 0usize..3)
            .prop_map(|(pulse, input_len, before, after)| PadProblem {
                pulse,
                input_len,
                before,
                after,
            })
            .boxed()
    }
}

impl Test for PadProblem {
    fn run_with_approx(
        &self,
        _id: &str,
        runtime: &dyn Runtime,
        approx: Approximation,
    ) -> infra::TestResult {
        let mut model = TypedModel::default();
        let s = model.symbols.sym("S");
        let a = model.add_source("a", f32::fact(&[s.to_dim()]))?;
        let pad = model.wire_node(
            "pad",
            Pad::new(
                vec![(self.before, self.after)],
                PadMode::Constant(Arc::new(Tensor::from(-1f32))),
            ),
            &[a],
        )?;
        model.select_output_outlets(&pad)?;

        let input = Array1::range(1.0f32, self.input_len as f32 + 1.0, 1.0);
        pulse_and_compare(runtime, approx, model, self.pulse, input.into_dyn(), 0)
    }
}

pub fn suite() -> TractResult<TestSuite> {
    let mut suite = TestSuite::default();
    suite.add_arbitrary::<PadProblem>("proptest", ());
    let case = |pulse, input_len, before| PadProblem { pulse, input_len, before, after: 0 };
    suite.add_test("before_1", case(1, 1, 1));
    suite.add_test("before_2", case(2, 2, 1));
    suite.add_test("shrunk_before_2", PadProblem { pulse: 1, input_len: 2, before: 2, after: 0 });
    Ok(suite)
}
