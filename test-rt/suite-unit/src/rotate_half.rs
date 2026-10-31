use infra::Test;
use infra::TestResult;
use infra::TestSuite;
use proptest::collection::vec;
use proptest::prelude::*;
use tract_core::internal::*;
use tract_core::ndarray::ArrayD;
use tract_core::ndarray::concatenate;
use tract_core::num_traits::Float;
use tract_transformers::ops::apply_rope::RotateHalf;

use crate::tensor;

/// `RotateHalf` over the last axis, which must have an even length: the second half negated,
/// then the first half.
#[derive(Debug, Clone)]
pub struct RotateHalfProblem<F>
where
    F: Datum + Float,
{
    input: ArrayD<F>,
}

impl<F> Arbitrary for RotateHalfProblem<F>
where
    F: Datum + Float,
{
    type Parameters = ();
    type Strategy = BoxedStrategy<RotateHalfProblem<F>>;

    fn arbitrary_with(_params: Self::Parameters) -> Self::Strategy {
        (vec(1usize..6, 0..4), 1usize..10)
            .prop_flat_map(|(mut shape, half)| {
                shape.push(2 * half);
                tensor::<F>(&shape).prop_map(|input| Self { input })
            })
            .boxed()
    }
}

impl<F> RotateHalfProblem<F>
where
    F: Datum + Float,
{
    fn tract(&self) -> TractResult<TypedModel> {
        let mut model = TypedModel::default();
        let input = model
            .add_source("input", TypedFact::shape_and_dt_of(&self.input.clone().into_tensor()))?;
        let output = model.wire_node("rotate_half", RotateHalf, &[input])?;
        model.select_output_outlets(&output)?;
        model.into_decluttered()
    }

    fn reference(&self) -> ArrayD<F> {
        let axis = tract_ndarray::Axis(self.input.ndim() - 1);
        let mid = self.input.shape()[axis.0] / 2;
        let first_half = self.input.slice_axis(axis, (..mid).into());
        let second_half = self.input.slice_axis(axis, (mid..).into()).mapv(|x| -x);
        concatenate(axis, &[second_half.view(), first_half]).unwrap()
    }
}

impl<F> Test for RotateHalfProblem<F>
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
        let mut output = runtime.prepare(model)?.run(tvec![self.input.clone().into_tvalue()])?;
        let output = output.remove(0).into_tensor();
        output.close_enough(&reference, approx)
    }
}

pub fn suite() -> TractResult<TestSuite> {
    let mut suite = TestSuite::default();

    suite.add_arbitrary::<RotateHalfProblem<f32>>("proptest_f32", ());
    suite.add_arbitrary::<RotateHalfProblem<f16>>("proptest_f16", ());

    suite.add(
        "trivial_f32_0",
        RotateHalfProblem { input: tensor1(&[0f32, 1., 2., 3.]).into_plain_array::<f32>()? },
    );
    suite.add(
        "heads_f32",
        RotateHalfProblem {
            input: Tensor::from_shape(
                &[2, 3, 64],
                &(0..2 * 3 * 64).map(|i| i as f32).collect::<Vec<_>>(),
            )?
            .into_plain_array::<f32>()?,
        },
    );
    Ok(suite)
}
