use infra::Test;
use infra::TestResult;
use infra::TestSuite;
use tract_core::internal::*;
use tract_core::ops::math::add;
use tract_transformers::ops::moe_ffn::RoutedCombine;

#[derive(Clone)]
struct CombineCase {
    dt: DatumType,
    rank3: bool,
    constant_ids: bool,
    invalid_id: Option<i64>,
}

impl Test for CombineCase {
    fn run_with_approx(
        &self,
        _: &'static str,
        runtime: &dyn Runtime,
        _: Approximation,
    ) -> TestResult {
        let shape = if self.rank3 { tvec![1, 2, 2] } else { tvec![2, 2] };
        let mut model = TypedModel::default();
        let x = model.add_source("shape_like", self.dt.fact(&shape))?;
        let values = model.add_const(
            "values",
            Tensor::from_shape(&[4, 2], &[2f32, 4., 4., 8., 6., 8., 8., 12.])?,
        )?;
        let mut inputs = tvec![Tensor::zero_dt(self.dt, &shape)?.into_tvalue()];
        let id_values = tensor1(&[self.invalid_id.unwrap_or(0), 1, 0, 1]);
        let ids = if self.constant_ids {
            model.add_const("ids", id_values)?
        } else {
            inputs.push(id_values.into_tvalue());
            model.add_source("ids", i64::fact([4]))?
        };
        let weights = model.add_const("weights", tensor1(&[0.5f32, 0.25, 0.5, 0.75]))?;
        let combined = model.wire_node("combine", RoutedCombine, &[x, values, ids, weights])?[0];
        let added = model.wire_node("add", add(), &[combined, x])?[0];
        model.select_output_outlets(&[combined, added])?;
        let expected =
            Tensor::from_shape(&shape, &[4f32, 6., 7., 11.])?.cast_to_dt(self.dt)?.into_owned();
        let result = runtime.prepare(model).and_then(|plan| plan.run(inputs));
        if self.invalid_id.is_some() {
            ensure!(result.is_err(), "invalid combine token ID was accepted");
            return Ok(());
        }
        let outputs = result?;
        for output in outputs {
            ensure!(output.datum_type() == self.dt, "RoutedCombine output dtype changed");
            output.close_enough(&expected, Approximation::Exact)?;
        }
        Ok(())
    }
}

pub fn suite() -> TractResult<TestSuite> {
    let mut suite = TestSuite::default();
    for dt in [DatumType::F16, DatumType::F32] {
        for rank3 in [false, true] {
            for constant_ids in [false, true] {
                for invalid_id in [None, Some(-1), Some(2), Some(4_294_967_296)] {
                    suite.add(
                        format!(
                            "{dt:?}_rank{}_constant{constant_ids}_id{}",
                            if rank3 { 3 } else { 2 },
                            invalid_id.map_or("valid".to_string(), |id| id.to_string())
                        )
                        .replace('-', "neg"),
                        CombineCase { dt, rank3, constant_ids, invalid_id },
                    );
                }
            }
        }
    }
    Ok(suite)
}
