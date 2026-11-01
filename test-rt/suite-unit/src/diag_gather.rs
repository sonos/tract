use infra::Test;
use infra::TestResult;
use infra::TestSuite;
use proptest::collection::vec;
use proptest::prelude::*;
use tract_core::internal::*;
use tract_core::ndarray::ArrayD;
use tract_core::ndarray::Dimension;
use tract_core::num_traits::Float;
use tract_transformers::ops::diag_gather::DiagGather;

use crate::tensor;

/// `output[…, i, k] = input[…, i, offset + k − i]`, zero where that falls outside the input row.
#[derive(Debug, Clone)]
pub struct DiagGatherProblem<F>
where
    F: Datum + Float,
{
    input: ArrayD<F>,
    offset: i64,
    out_len: usize,
}

impl<F> Arbitrary for DiagGatherProblem<F>
where
    F: Datum + Float,
{
    type Parameters = ();
    type Strategy = BoxedStrategy<DiagGatherProblem<F>>;

    fn arbitrary_with(_params: Self::Parameters) -> Self::Strategy {
        (vec(1usize..4, 0..3), 1usize..8, 1usize..16, 1usize..16)
            .prop_flat_map(|(batch, t, r, out_len)| {
                let mut shape = batch;
                shape.extend([t, r]);
                (tensor::<F>(&shape), -(t as i64)..(r + t) as i64, Just(out_len))
            })
            .prop_map(|(input, offset, out_len)| Self { input, offset, out_len })
            .boxed()
    }
}

impl<F> DiagGatherProblem<F>
where
    F: Datum + Float,
{
    fn tract(&self) -> TractResult<TypedModel> {
        let mut model = TypedModel::default();
        let input = self.input.clone().into_tensor();
        let input = model.add_source("input", TypedFact::shape_and_dt_of(&input))?;
        let op = DiagGather { offset: self.offset.to_dim(), out_len: self.out_len.to_dim() };
        let output = model.wire_node("diag_gather", op, &[input])?;
        model.select_output_outlets(&output)?;
        Ok(model)
    }

    fn reference(&self) -> ArrayD<F> {
        let rank = self.input.ndim();
        let r = self.input.shape()[rank - 1] as i64;
        let mut shape = self.input.shape().to_vec();
        shape[rank - 1] = self.out_len;
        ArrayD::from_shape_fn(shape, |coords| {
            let (i, k) = (coords[rank - 2] as i64, coords[rank - 1] as i64);
            let source = self.offset + k - i;
            if (0..r).contains(&source) {
                let mut coords = coords.as_array_view().to_vec();
                coords[rank - 1] = source as usize;
                self.input[&*coords]
            } else {
                F::zero()
            }
        })
    }
}

impl<F> Test for DiagGatherProblem<F>
where
    F: Datum + Float,
{
    fn run_with_approx(
        &self,
        id: &str,
        runtime: &dyn Runtime,
        approx: Approximation,
    ) -> TestResult {
        let reference = self.reference().into_tensor();
        let mut model = self.tract()?;
        model.properties.insert("tract-rt-test.id".to_string(), rctensor0(id.to_string()));
        let mut output = runtime.prepare(model)?.run(tvec!(self.input.clone().into_tvalue()))?;
        output.remove(0).into_tensor().close_enough(&reference, approx)
    }
}

pub fn suite() -> TractResult<TestSuite> {
    let mut suite = TestSuite::default();
    suite.add_arbitrary::<DiagGatherProblem<f32>>("proptest_f32", ());
    suite.add_arbitrary::<DiagGatherProblem<f16>>("proptest_f16", ());
    Ok(suite)
}
