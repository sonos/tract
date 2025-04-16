use infra::Test;
use infra::TestResult;
use infra::TestSuite;
use proptest::collection::vec;
use proptest::prelude::*;
use tract_core::internal::*;
use tract_core::ndarray::concatenate;
use tract_core::ndarray::ArrayD;
use tract_core::num_traits::Float;
use tract_transformers::ops::apply_rope::ApplyRope;

use crate::tensor;

#[derive(Debug, Clone)]
pub struct ApplyRopeProblem<F>
where
    F: Datum + Float,
{
    input: ArrayD<F>,
    cos: ArrayD<F>,
    sin: ArrayD<F>,
}

impl<F> Arbitrary for ApplyRopeProblem<F>
where
    F: Datum + Float,
{
    type Parameters = ();
    type Strategy = BoxedStrategy<ApplyRopeProblem<F>>;

    fn arbitrary_with(_params: Self::Parameters) -> Self::Strategy {
        (2usize..5)
            .prop_flat_map(|rank| {
                let dim = 1usize..10;
                (vec(dim.clone(), 2..=2), vec(dim, (rank - 2)..=(rank - 2)))
            })
            .prop_flat_map(|(mut cos_sin_shape, extra_shape)| {
                cos_sin_shape[1] *= 2; // Ensure inner axis dim is multiple of 2
                (
                    tensor::<F>(&[extra_shape.clone(), cos_sin_shape.clone()].concat()),
                    tensor::<F>(&[extra_shape.clone(), cos_sin_shape.clone()].concat()),
                    tensor::<F>(&[extra_shape, cos_sin_shape.clone()].concat()),
                )
                    .prop_map(|(input, cos, sin)| Self { input, cos, sin })
            })
            .boxed()
    }
}

impl<F> ApplyRopeProblem<F>
where
    F: Datum + Float,
    f32: From<F>,
{
    fn tract(&self) -> TractResult<TypedModel> {
        let mut model = TypedModel::default();

        let input = model
            .add_source("input", TypedFact::shape_and_dt_of(&self.input.clone().into_tensor()))?;
        let cos =
            model.add_source("cos", TypedFact::shape_and_dt_of(&self.cos.clone().into_tensor()))?;
        let sin =
            model.add_source("sin", TypedFact::shape_and_dt_of(&self.sin.clone().into_tensor()))?;

        let output = model.wire_node("apply_rope", ApplyRope, &[input, cos, sin])?;
        model.set_output_outlets(&output)?;

        model = model.into_decluttered()?;
        Ok(model)
    }

    fn reference(&self) -> ArrayD<F> {
        let input = self.input.clone();

        let inner_axis = input.shape().len() - 1;
        let axis = tract_ndarray::Axis(inner_axis);
        let len = input.shape()[inner_axis];
        assert!(len % 2 == 0, "Length of the last axis must be even");

        let mid = len / 2;

        // Slice and clone the two halves
        let first_half = input.slice_axis(axis, (..mid).into()).to_owned();
        let mut second_half = input.slice_axis(axis, (mid..).into()).to_owned();
        second_half.mapv_inplace(|x| -x);

        // Concatenate in reverse order
        let rotated_input = concatenate(axis, &[second_half.view(), first_half.view()]).unwrap();

        let brd_cos = self.cos.broadcast(input.raw_dim()).unwrap();
        let brd_sin = self.sin.broadcast(input.raw_dim()).unwrap();

        (input * brd_cos) + (rotated_input * brd_sin)
    }
}

impl<F> Test for ApplyRopeProblem<F>
where
    F: Datum + Float,
    f32: From<F>,
{
    fn run_with_approx(
        &self,
        _suite: &str,
        id: &str,
        runtime: &dyn Runtime,
        approx: Approximation,
    ) -> TestResult {
        let reference = self.reference().into_tensor();
        let mut model = self.tract()?;

        model.properties.insert("tract-rt-test.id".to_string(), rctensor0(id.to_string()));

        let mut output = runtime.prepare(model)?.run(tvec![
            self.input.clone().into_tvalue(),
            self.cos.clone().into_tvalue(),
            self.sin.clone().into_tvalue()
        ])?;
        let output = output.remove(0).into_tensor();

        output.close_enough(&reference, approx)
    }
}

pub fn suite() -> TractResult<TestSuite> {
    let mut suite = TestSuite::default();

    suite.add_arbitrary::<ApplyRopeProblem<f32>>("proptest_f32", ());
    suite.add_arbitrary::<ApplyRopeProblem<f16>>("proptest_f16", ());

    suite.add(
        "trivial_f32_0",
        ApplyRopeProblem {
            input: tensor2(&[[0f32, 1f32]]).into_array::<f32>().unwrap(),
            cos: tensor2(&[[0f32, 0f32]]).into_array::<f32>().unwrap(),
            sin: tensor2(&[[0f32, 0f32]]).into_array::<f32>().unwrap(),
        },
    );
    Ok(suite)
}
