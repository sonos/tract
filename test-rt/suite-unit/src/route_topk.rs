use infra::Test;
use infra::TestResult;
use infra::TestSuite;
use tract_core::internal::*;
use tract_transformers::ops::moe_ffn::GateMode;
use tract_transformers::ops::moe_ffn::RouteTopK;

#[derive(Clone)]
struct RouterCase {
    tokens: usize,
    invalid_k: bool,
    large_weights: bool,
    dt: DatumType,
}

impl Test for RouterCase {
    fn run_with_approx(
        &self,
        _: &'static str,
        runtime: &dyn Runtime,
        _: Approximation,
    ) -> TestResult {
        let mut model = TypedModel::default();
        let x = model.add_source("x", self.dt.fact([self.tokens, 64]))?;
        let (low, high) =
            if self.large_weights { (70_000f32, 80_000f32) } else { (1.0001f32, 1.0002f32) };
        let mut weights = vec![low; 128];
        weights[64..].fill(high);
        let wg = model.add_const("wg", Tensor::from_shape(&[2, 64], &weights)?)?;
        let output = model.wire_node(
            "routes",
            RouteTopK { k: if self.invalid_k { 3 } else { 1 }, gate: GateMode::Raw },
            &[x, wg],
        )?;
        model.select_output_outlets(&output)?;
        let result = runtime.prepare(model).and_then(|plan| {
            plan.run(tvec![
                Tensor::from_shape(&[self.tokens, 64], &vec![1f32; self.tokens * 64])?
                    .cast_to_dt(self.dt)?
                    .into_owned()
                    .into_tvalue()
            ])
        });
        if self.invalid_k {
            ensure!(result.is_err(), "top-k larger than expert count was accepted");
        } else {
            let outputs = result?;
            outputs[1].close_enough(
                &Tensor::from_shape(&[self.tokens], &vec![1i64; self.tokens])?,
                Approximation::Exact,
            )?;
            outputs[2].close_enough(
                &Tensor::from_shape(&[self.tokens], &vec![64. * high; self.tokens])?,
                Approximation::Approximate,
            )?;
        }
        Ok(())
    }
}

pub fn suite() -> TractResult<TestSuite> {
    let mut suite = TestSuite::default();
    for tokens in [1, 5] {
        for large_weights in [false, true] {
            for dt in [DatumType::F16, DatumType::F32] {
                suite.add(
                    format!("scores_{dt:?}_tokens{tokens}_large{large_weights}"),
                    RouterCase { tokens, invalid_k: false, large_weights, dt },
                );
            }
        }
    }
    suite.add(
        "invalid_topk",
        RouterCase { tokens: 1, invalid_k: true, large_weights: false, dt: DatumType::F32 },
    );
    Ok(suite)
}
