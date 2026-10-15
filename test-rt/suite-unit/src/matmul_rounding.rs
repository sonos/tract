use infra::{Test, TestResult, TestSuite};
use tract_core::internal::*;
use tract_core::ops::cast::Cast;
use tract_core::ops::einsum::EinSum;
use tract_core::ops::math::add;

#[derive(Clone)]
struct RoundingCase {
    cast: bool,
}

impl Test for RoundingCase {
    fn run_with_approx(
        &self,
        _: &'static str,
        runtime: &dyn Runtime,
        _: Approximation,
    ) -> TestResult {
        let mut model = TypedModel::default();
        let mut x = vec![f16::ZERO; 16 * 32];
        for row in 0..16 {
            x[row * 32] = f16::from_f32(64.5);
        }
        let mut w = vec![f16::ZERO; 32 * 16];
        w[..16].fill(f16::from_f32(64.5));
        let x = Tensor::from_shape(&[16, 32], &x)?;
        let input = model.add_source("x", TypedFact::shape_and_dt_of(&x))?;
        let weight = model.add_const("weight", Tensor::from_shape(&[32, 16], &w)?)?;
        let product = model.wire_node(
            "product",
            EinSum::new("mk,kn->mn".parse()?, DatumType::F16),
            &[input, weight],
        )?[0];
        let (output, expected) = if self.cast {
            (
                model.wire_node("cast", Cast::new(DatumType::F32), &[product])?[0],
                Tensor::from_shape(&[16, 16], &[4160f32; 256])?,
            )
        } else {
            let bias =
                model.add_const("bias", Tensor::from_shape(&[1, 1], &[f16::from_f32(-4160.)])?)?;
            (model.wire_node("add", add(), &[product, bias])?[0], Tensor::zero::<f16>(&[16, 16])?)
        };
        model.select_output_outlets(&[output])?;
        let outputs = runtime.prepare(model)?.run(tvec![x.into_tvalue()])?;
        outputs[0].close_enough(&expected, Approximation::Exact)
    }
}

pub fn suite() -> TractResult<TestSuite> {
    let mut suite = TestSuite::default();
    suite.add("before_add", RoundingCase { cast: false });
    suite.add("before_cast", RoundingCase { cast: true });
    Ok(suite)
}
