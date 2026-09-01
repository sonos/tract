use infra::{Test, TestSuite};
use proptest::prelude::*;
use tract_core::internal::*;
use tract_core::ndarray::Array1;
use tract_core::ops::array::Slice;
use tract_core::runtime::Runtime;

use crate::pulse_and_compare;

/// A Slice on the streaming axis, dropping `begin` frames at the head and
/// `end` at the tail: the head becomes output delay, the tail shortens the
/// stream.
#[derive(Clone, Debug)]
pub struct SliceProblem {
    pub pulse: usize,
    pub input_len: usize,
    pub begin: usize,
    pub end: usize,
}

impl Arbitrary for SliceProblem {
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with(_: Self::Parameters) -> BoxedStrategy<Self> {
        (1usize..3, 0usize..10, 0usize..3, 0usize..3)
            .prop_map(|(pulse, input_len, begin, end)| SliceProblem {
                pulse,
                input_len,
                begin,
                end,
            })
            .boxed()
    }
}

impl Test for SliceProblem {
    fn run_with_approx(
        &self,
        _id: &str,
        runtime: &dyn Runtime,
        approx: Approximation,
    ) -> infra::TestResult {
        let mut model = TypedModel::default();
        let s = model.symbols.sym("S");
        let a = model.add_source("a", f32::fact(&[s.to_dim()]))?;
        let slice = model.wire_node(
            "slice",
            Slice::new(0, self.begin, self.input_len + self.begin),
            &[a],
        )?;
        model.select_output_outlets(&slice)?;

        let full_len = self.input_len + self.begin + self.end;
        let input = Array1::range(1.0f32, full_len as f32 + 1.0, 1.0);
        pulse_and_compare(runtime, approx, model, self.pulse, input.into_dyn(), 0)
    }
}

pub fn suite() -> TractResult<TestSuite> {
    let mut suite = TestSuite::default();
    suite.add_arbitrary::<SliceProblem>("proptest", ());
    Ok(suite)
}
