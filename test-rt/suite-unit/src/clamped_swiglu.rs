use infra::{Test, TestResult, TestSuite};
use tract_core::internal::*;
use tract_core::ops::nn::ClampedSwiGlu;

#[derive(Clone, Debug)]
struct Case {
    gate: Tensor,
    up: Tensor,
    op: ClampedSwiGlu,
}

impl Test for Case {
    fn run_with_approx(
        &self,
        id: &'static str,
        runtime: &dyn Runtime,
        approx: Approximation,
    ) -> TestResult {
        let mut model = TypedModel::default();
        let gate = model.add_source("gate", TypedFact::shape_and_dt_of(&self.gate))?;
        let up = model.add_source("up", TypedFact::shape_and_dt_of(&self.up))?;
        let output = model.wire_node("activation", self.op.clone(), &[gate, up])?;
        model.select_output_outlets(&output)?;
        model.properties.insert("tract-rt-test.id".into(), rctensor0(id.to_string()));
        let expected = self
            .gate
            .cast_to::<f32>()?
            .to_plain_array_view::<f32>()?
            .iter()
            .zip(self.up.cast_to::<f32>()?.to_plain_array_view::<f32>()?.iter())
            .map(|(&g, &u)| {
                let g = f64::from(g).min(f64::from(self.op.limit));
                let u = f64::from(u).clamp(-f64::from(self.op.limit), f64::from(self.op.limit));
                ((u + 1.0) * g / (1.0 + (-f64::from(self.op.alpha) * g).exp())) as f32
            })
            .collect::<Vec<_>>();
        let expected = Tensor::from_shape(self.gate.shape(), &expected)?;
        let outputs = runtime
            .prepare(model)?
            .run(tvec![self.gate.clone().into_tvalue(), self.up.clone().into_tvalue()])?;
        ensure!(outputs[0].datum_type() == DatumType::F32);
        outputs[0].close_enough(&expected, approx)
    }
}

#[derive(Clone, Debug)]
struct Contract;

impl Test for Contract {
    fn run_with_approx(
        &self,
        _id: &'static str,
        _runtime: &dyn Runtime,
        _approx: Approximation,
    ) -> TestResult {
        let op = ClampedSwiGlu { alpha: 1.7, limit: 2.0 };
        let valid = f32::fact([2]);
        for facts in [
            vec![],
            vec![valid.clone()],
            vec![valid.clone(); 3],
            vec![valid.clone(), f32::fact([1])],
            vec![valid.clone(), i32::fact([2])],
        ] {
            ensure!(op.output_facts(&facts.iter().collect::<Vec<_>>()).is_err());
            let inputs = facts
                .iter()
                .map(|f| {
                    Ok(Tensor::zero_dt(f.datum_type, f.shape.as_concrete().unwrap())?.into_tvalue())
                })
                .collect::<TractResult<TVec<_>>>()?;
            ensure!(op.eval(&EvalContext::out_of_plan(), inputs).is_err());
        }
        for (alpha, limit) in [
            (f32::NAN, 2.0),
            (f32::INFINITY, 2.0),
            (1.7, f32::NAN),
            (1.7, f32::INFINITY),
            (1.7, 0.0),
            (1.7, -2.0),
        ] {
            let invalid = ClampedSwiGlu { alpha, limit };
            ensure!(invalid.output_facts(&[&valid, &valid]).is_err());
            ensure!(
                invalid
                    .eval(
                        &EvalContext::out_of_plan(),
                        tvec![
                            tensor1(&[1f32, 2.]).into_tvalue(),
                            tensor1(&[1f32, 2.]).into_tvalue()
                        ]
                    )
                    .is_err()
            );
        }
        let symbols = SymbolScope::default();
        let fact = f32::fact([symbols.sym("S").to_dim(), 2.to_dim()]);
        ensure!(op.output_facts(&[&fact, &fact])?[0].shape == fact.shape);
        let outputs = op.eval(
            &EvalContext::out_of_plan(),
            tvec![
                Tensor::zero::<f32>(&[0, 2])?.into_tvalue(),
                Tensor::zero::<f32>(&[0, 2])?.into_tvalue()
            ],
        )?;
        ensure!(outputs[0].shape() == [0, 2]);
        Ok(())
    }
}

pub fn suite() -> TractResult<TestSuite> {
    let mut suite = TestSuite::default();
    let gate = Tensor::from_shape(&[2, 4], &[-4f32, -2., -0.5, 0., 0.5, 2., 4., 10.])?;
    let up = Tensor::from_shape(&[2, 4], &[4f32, -4., 0., -1., 0.5, 2., -2., 10.])?;
    for (name, gate_dt, up_dt) in [
        ("f32", DatumType::F32, DatumType::F32),
        ("f16", DatumType::F16, DatumType::F16),
        ("mixed", DatumType::F16, DatumType::F32),
    ] {
        suite.add_test(
            name,
            Case {
                gate: gate.cast_to_dt(gate_dt)?.into_owned(),
                up: up.cast_to_dt(up_dt)?.into_owned(),
                op: ClampedSwiGlu { alpha: 1.7, limit: 2.0 },
            },
        );
    }
    suite.add_test(
        "scalar",
        Case {
            gate: tensor0(3f32),
            up: tensor0(-4f32),
            op: ClampedSwiGlu { alpha: 0.5, limit: 1.5 },
        },
    );
    suite.add_test("contract", Contract);
    Ok(suite)
}
