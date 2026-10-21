use infra::Test;
use infra::TestResult;
use infra::TestSuite;
use proptest::collection::vec;
use proptest::prelude::*;
use tract_core::internal::*;
use tract_core::ndarray::ArrayD;
use tract_core::num_traits::Float;
use tract_core::ops::nn::gelu_exact::gelu_exact;
use tract_core::tract_linalg::routines::Func;

use crate::tensor;

#[derive(Debug, Clone)]
pub struct GeluExactProblem<F>
where
    F: Datum + Float,
{
    input: ArrayD<F>,
}

impl<F> Arbitrary for GeluExactProblem<F>
where
    F: Datum + Float,
{
    type Parameters = ();
    type Strategy = BoxedStrategy<GeluExactProblem<F>>;

    fn arbitrary_with(_params: Self::Parameters) -> Self::Strategy {
        (0usize..5)
            .prop_flat_map(|rank| vec(1usize..10, rank..=rank))
            .prop_flat_map(|shape| tensor::<F>(&shape).prop_map(|input| Self { input }))
            .boxed()
    }
}

impl<F> GeluExactProblem<F>
where
    F: Datum + Float,
{
    fn tract(&self) -> TractResult<TypedModel> {
        let mut model = TypedModel::default();
        let input = self.input.clone().into_tensor();
        let input = model.add_source("input", TypedFact::shape_and_dt_of(&input))?;
        let output = model.wire_node("gelu", gelu_exact(), &[input])?;
        model.select_output_outlets(&output)?;
        model.into_decluttered()
    }

    fn reference(&self) -> TractResult<ArrayD<F>> {
        let x = self.input.mapv(|v| v.to_f32().unwrap());
        let mut e: Vec<f32> = x.iter().map(|v| v * std::f32::consts::FRAC_1_SQRT_2).collect();
        Func::Erf.ew_f32()?.run(&mut e)?;
        let out: Vec<F> =
            x.iter().zip(e).map(|(x, e)| F::from(0.5 * x * (1.0 + e)).unwrap()).collect();
        Ok(ArrayD::from_shape_vec(x.shape(), out)?)
    }
}

impl<F> Test for GeluExactProblem<F>
where
    F: Datum + Float,
{
    fn run_with_approx(
        &self,
        id: &str,
        runtime: &dyn Runtime,
        approx: Approximation,
    ) -> TestResult {
        let reference = self.reference()?.into_tensor();
        let mut model = self.tract()?;
        model.properties.insert("tract-rt-test.id".to_string(), rctensor0(id.to_string()));
        let mut output = runtime.prepare(model)?.run(tvec!(self.input.clone().into_tvalue()))?;
        let output = output.remove(0).into_tensor();
        output.close_enough(&reference, approx)
    }
}

pub fn suite() -> TractResult<TestSuite> {
    let mut suite = TestSuite::default();
    suite.add_arbitrary::<GeluExactProblem<f32>>("proptest_f32", ());
    suite.add_arbitrary::<GeluExactProblem<f16>>("proptest_f16", ());
    Ok(suite)
}
