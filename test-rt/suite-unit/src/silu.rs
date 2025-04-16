use infra::Test;
use infra::TestResult;
use infra::TestSuite;
use proptest::collection::vec;
use proptest::prelude::*;
use tract_core::internal::*;
use tract_core::ndarray::ArrayD;
use tract_core::num_traits::Float;
use tract_transformers::ops::silu::Silu;

use crate::tensor;

#[derive(Debug, Clone)]
pub struct SiluProblem<F>
where
    F: Datum + Float,
{
    input: ArrayD<F>,
}

impl<F> Arbitrary for SiluProblem<F>
where
    F: Datum + Float,
{
    type Parameters = ();
    type Strategy = BoxedStrategy<SiluProblem<F>>;

    fn arbitrary_with(_params: Self::Parameters) -> Self::Strategy {
        (0usize..5)
            .prop_flat_map(|rank| {
                let other_dim = 1usize..10;
                vec(other_dim, rank..=rank)
            })
            .prop_flat_map(|shape| tensor::<F>(&shape).prop_map(|input| Self { input }))
            .boxed()
    }
}

impl<F> SiluProblem<F>
where
    F: Datum + Float,
    f32: From<F>,
{
    fn tract(&self) -> TractResult<TypedModel> {
        let mut model = TypedModel::default();
        let input = self.input.clone().into_tensor();
        let input = model.add_source("input", TypedFact::shape_and_dt_of(&input))?;

        let output = model.wire_node("silu", Silu, &[input])?;
        model.set_output_outlets(&output)?;

        model = model.into_decluttered()?;
        Ok(model)
    }

    fn reference(&self) -> ArrayD<F> {
        let input = &self.input;
        input.mapv(|x| F::from(f32::from(x) / (1.0 + f32::from(-x).exp())).unwrap())
    }
}

impl<F> Test for SiluProblem<F>
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

        let mut output = runtime.prepare(model)?.run(tvec!(self.input.clone().into_tvalue()))?;
        let output = output.remove(0).into_tensor();

        output.close_enough(&reference, approx)
    }
}

pub fn suite() -> TractResult<TestSuite> {
    let mut suite = TestSuite::default();

    suite.add_arbitrary::<SiluProblem<f32>>("proptest_f32", ());
    suite.add_arbitrary::<SiluProblem<f16>>("proptest_f16", ());

    Ok(suite)
}
