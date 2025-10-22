use std::fmt;

use infra::Test;
use infra::TestResult;
use infra::TestSuite;
use proptest::collection::vec;
use proptest::prelude::*;
use tract_core::internal::*;
use tract_core::ndarray::ArrayD;
use tract_core::num_traits::Float;
use tract_core::num_traits::FromPrimitive;
use tract_transformers::ops::rms_norm::RmsNorm;

use crate::tensor;

#[derive(Clone)]
pub struct RmsNormProblem<F>
where
    F: Datum + Float,
{
    input: Tensor,
    axis: usize,
    eps: F,
}

impl<F> std::fmt::Debug for RmsNormProblem<F>
where
    F: Datum + Float,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Input:{:?} Axis:{:?} Epsilon:{:?}", self.input, self.axis, self.eps)
    }
}

impl<F> Arbitrary for RmsNormProblem<F>
where
    F: Datum + Float,
{
    type Parameters = ();
    type Strategy = BoxedStrategy<RmsNormProblem<F>>;

    fn arbitrary_with(_params: Self::Parameters) -> Self::Strategy {
        (0usize..3, 0usize..3)
            .prop_flat_map(|(left, right)| {
                let axis = left;
                let shape_len = usize::min(left + right, 4);
                let iter_ax_dim = 1usize..50;
                let other_dim = 1usize..5;
                (iter_ax_dim, vec(other_dim, shape_len..=shape_len), Just(axis))
            })
            .prop_flat_map(|(iter_dim, mut shape, axis)| {
                shape.insert(axis, iter_dim);
                let input = tensor::<F>(&shape);
                (input, Just(axis), 0f32..=1e6).prop_map(|(input, axis, eps)| Self {
                    input: input.into(),
                    axis,
                    eps: F::from(eps / 1e5).unwrap(),
                })
            })
            .boxed()
    }
}

impl<F> RmsNormProblem<F>
where
    F: Datum + Float + FromPrimitive,
{
    fn tract(&self) -> TractResult<TypedModel> {
        let mut model = TypedModel::default();
        let input = model.add_source("input", TypedFact::shape_and_dt_of(&self.input))?;

        let output = model.wire_node(
            "rms_norm",
            RmsNorm { axis: self.axis, eps: tensor0(self.eps).into_arc_tensor() },
            &[input],
        )?;
        model.set_output_outlets(&output)?;

        model = model.into_decluttered()?;
        Ok(model)
    }

    fn reference(&self) -> ArrayD<F> {
        let input = self.input.cast_to::<F>().unwrap();

        let a = input.to_array_view::<F>().unwrap().to_owned();
        let mean_square = a.pow2().mean_axis(tract_ndarray::Axis(self.axis)).unwrap();

        let norm = mean_square
            .mapv(|ms| (ms + self.eps).sqrt())
            .insert_axis(tract_ndarray::Axis(self.axis));
        let broadcasted_norm = norm.broadcast(a.raw_dim()).unwrap();

        a / broadcasted_norm
    }
}

impl<F> Test for RmsNormProblem<F>
where
    F: Datum + Float + FromPrimitive,
{
    fn run_with_approx(
        &self,
        _suite: &str,
        id: &str,
        runtime: &dyn Runtime,
        approx: Approximation,
    ) -> TestResult {
        let reference = self.reference().into_tensor();
        //dbg!(&reference);
        let mut model = self.tract()?;

        model.properties.insert("tract-rt-test.id".to_string(), rctensor0(id.to_string()));

        let mut output = runtime.prepare(model)?.run(tvec!(self.input.clone().into()))?;
        let output = output.remove(0).into_tensor();
        output.close_enough(&reference, approx)
    }
}

pub fn suite() -> TractResult<TestSuite> {
    let mut suite = TestSuite::default();

    suite.add_arbitrary::<RmsNormProblem<f32>>("proptest_f32", ());
    suite.add_arbitrary::<RmsNormProblem<f16>>("proptest_f16", ());

    suite.add("trivial_f32_0", RmsNormProblem { input: tensor1(&[0f32]), axis: 0, eps: 0f32 });

    Ok(suite)
}
