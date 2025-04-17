use std::f32::consts::PI;

use infra::Test;
use infra::TestResult;
use infra::TestSuite;
use proptest::collection::vec;
use proptest::prelude::*;
use tract_core::internal::*;
use tract_core::ndarray::ArrayD;
use tract_core::num_traits::Float;
use tract_transformers::ops::gelu_approximate::GeluApproximate;

use crate::tensor;

#[derive(Debug, Clone)]
pub struct GeluApproximateProblem<F>
where
    F: Datum + Float,
{
    input: ArrayD<F>,
    fast_impl: bool,
}

impl<F> Arbitrary for GeluApproximateProblem<F>
where
    F: Datum + Float,
{
    type Parameters = ();
    type Strategy = BoxedStrategy<GeluApproximateProblem<F>>;

    fn arbitrary_with(_params: Self::Parameters) -> Self::Strategy {
        (0usize..5)
            .prop_flat_map(|rank| {
                let other_dim = 1usize..10;
                vec(other_dim, rank..=rank)
            })
            .prop_flat_map(|shape| {
                (tensor::<F>(&shape), any::<bool>())
                    .prop_map(move |(input, fast_impl)| Self { input, fast_impl })
            })
            .boxed()
    }
}

impl<F> GeluApproximateProblem<F>
where
    F: Datum + Float,
    f32: From<F>,
{
    fn tract(&self) -> TractResult<TypedModel> {
        let mut model = TypedModel::default();
        let input = self.input.clone().into_tensor();
        let input = model.add_source("input", TypedFact::shape_and_dt_of(&input))?;

        let output =
            model.wire_node("gelu", GeluApproximate { fast_impl: self.fast_impl }, &[input])?;
        model.set_output_outlets(&output)?;

        model = model.into_decluttered()?;
        Ok(model)
    }

    fn reference(&self) -> ArrayD<F> {
        let input = &self.input;
        //0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)));
        input.mapv(|x| {
            let x_f32 = f32::from(x);
            let pow = if self.fast_impl { 2 } else { 3 };
            F::from(
                0.5 * x_f32
                    * (1. + ((2. / PI).sqrt() * (x_f32 + 0.044715 * x_f32.powi(pow))).tanh()),
            )
            .unwrap()
        })
    }
}

impl<F> Test for GeluApproximateProblem<F>
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

    suite.add_arbitrary::<GeluApproximateProblem<f32>>("proptest_f32", ());
    suite.add_arbitrary::<GeluApproximateProblem<f16>>("proptest_f16", ());

    Ok(suite)
}
