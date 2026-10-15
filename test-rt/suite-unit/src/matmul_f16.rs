use infra::Test;
use infra::TestResult;
use infra::TestSuite;
use tract_core::internal::*;
use tract_core::ops::einsum::EinSum;

#[derive(Clone)]
struct WideProducts {
    k: usize,
    weight_first: bool,
}

impl Test for WideProducts {
    fn run_with_approx(
        &self,
        _: &'static str,
        runtime: &dyn Runtime,
        _: Approximation,
    ) -> TestResult {
        let weights = Tensor::from_shape(&[8, self.k], &vec![f16::from_f32(300.); 8 * self.k])?;
        let mut values = vec![f16::ZERO; self.k];
        if self.k == 130 {
            values[128] = f16::from_f32(300.);
            values[129] = f16::from_f32(-300.);
        } else {
            values[..self.k / 2].fill(f16::from_f32(300.));
            values[self.k / 2..].fill(f16::from_f32(-300.));
        }
        let input = Tensor::from_shape(&[1, self.k], &values)?;
        let (a, b, shape) =
            if self.weight_first { (weights, input, [8, 1]) } else { (input, weights, [1, 8]) };
        let mut model = TypedModel::default();
        let a_wire = model.add_source("a", TypedFact::shape_and_dt_of(&a))?;
        let b_wire = model.add_source("b", TypedFact::shape_and_dt_of(&b))?;
        let output = model.wire_node(
            "product",
            EinSum::new("mk,nk->mn".parse()?, DatumType::F16),
            &[a_wire, b_wire],
        )?;
        model.select_output_outlets(&output)?;
        let outputs = runtime.prepare(model)?.run(tvec![a.into_tvalue(), b.into_tvalue()])?;
        outputs[0].close_enough(&Tensor::zero::<f16>(&shape)?, Approximation::Exact)
    }
}

pub fn suite() -> TractResult<TestSuite> {
    let mut suite = TestSuite::default();
    for k in [64, 128, 130] {
        for weight_first in [false, true] {
            suite.add(
                format!("wide_products_k{k}_weight_first_{weight_first}"),
                WideProducts { k, weight_first },
            );
        }
    }
    Ok(suite)
}
