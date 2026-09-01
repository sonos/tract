use infra::{Test, TestSuite};
use tract_core::internal::*;
use tract_core::ops::einsum::EinSum;
use tract_core::runtime::Runtime;

use crate::pulse_and_compare;

/// An EinSum contracting a streaming input against a constant weight: the
/// streaming axis is a free axis of the expression, so one pulse is one matmul.
#[derive(Clone, Debug)]
pub struct StreamingEinSum;

impl Test for StreamingEinSum {
    fn run_with_approx(
        &self,
        _id: &str,
        runtime: &dyn Runtime,
        approx: Approximation,
    ) -> infra::TestResult {
        let mut model = TypedModel::default();
        let s = model.symbols.sym("S");
        let x = model.add_source("x", f32::fact(dims!(s, 8, 2)))?;
        let w = model.add_const("w", Tensor::zero::<f32>(&[8, 2, 4])?)?;
        let einsum = EinSum {
            axes: "sij,ijk->sik".parse()?,
            operating_dt: f32::datum_type(),
            q_params: None,
        };
        let einsum = model.wire_node("einsum", einsum, &[x, w])?;
        model.select_output_outlets(&einsum)?;

        let mut input = Tensor::zero::<f32>(&[5, 8, 2])?;
        input
            .try_as_plain_mut()?
            .as_slice_mut::<f32>()?
            .iter_mut()
            .enumerate()
            .for_each(|(ix, x)| *x = ix as f32);
        pulse_and_compare(runtime, approx, model, 1, input.into_plain_array()?, 0)
    }
}

pub fn suite() -> TractResult<TestSuite> {
    let mut suite = TestSuite::default();
    suite.add_test("pulsed_matmul", StreamingEinSum);
    Ok(suite)
}
