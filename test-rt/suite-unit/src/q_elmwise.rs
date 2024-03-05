use infra::{Test, TestSuite};
use proptest::prelude::*;
use tract_core::internal::*;

use crate::q_helpers::*;

#[derive(Debug, Clone)]
struct QElmWiseOpProblem {
    operator: tract_core::ops::element_wise::ElementWiseOp,
    tensor_input: Tensor,
    out_dt: DatumType,
}

impl QOpProblem for QElmWiseOpProblem {
    fn reference_float_ops(&self) -> TractResult<Tensor> {
        let inp = self.tensor_input.cast_to::<f32>()?.clone().into_owned();
        Ok(self.operator.eval(tvec![inp.into_tvalue()])?.remove(0).into_tensor())
    }
}

impl Arbitrary for QElmWiseOpProblem {
    type Parameters = ();
    type Strategy = BoxedStrategy<QElmWiseOpProblem>;

    fn arbitrary_with(_: Self::Parameters) -> Self::Strategy {
        let tested_operators = prop_oneof![
            Just(tract_core::ops::math::cos()),
            Just(tract_core::ops::math::tanh()),
            Just(tract_core::ops::nn::sigmoid())
        ];
        ((1..20usize), any::<bool>(), any::<bool>(), (1..10usize), (1..10usize), tested_operators)
            .prop_flat_map(|(len, inp_signed, out_signed, inp_scale, out_scale, op)| {
                let inp_dt = pick_signed_datum(inp_signed);
                let out_dt = pick_signed_datum(out_signed);
                fn just_scale(scale: usize) -> Just<f32> {
                    Just(scale as f32 * 0.5)
                }
                (
                    // tensor a
                    Just(inp_dt),
                    qtensor(vec![1], inp_dt),
                    just_scale(inp_scale),
                    qtensor(vec![len], inp_dt),
                    // dt of c
                    Just(out_dt),
                    qtensor(vec![1], out_dt),
                    just_scale(out_scale),
                    Just(op),
                )
            })
            .prop_map(|(inp_dt, inp_zp, inp_scale, inp_values, out_dt, out_zp, out_scale, op)| {
                let tensor_input = build_qtensor(
                    inp_values.into_tensor(),
                    inp_dt,
                    inp_zp.into_tensor(),
                    inp_scale,
                );
                let out_dt = out_dt.quantize(QParams::ZpScale {
                    zero_point: out_zp.into_tensor().cast_to_scalar::<i32>().unwrap(),
                    scale: out_scale,
                });
                QElmWiseOpProblem { operator: op.to_owned(), tensor_input, out_dt }
            })
            .boxed()
    }
}

impl Test for QElmWiseOpProblem {
    fn run_with_approx(
        &self,
        _suite: &str,
        id: &str,
        runtime: &dyn Runtime,
        approx: Approximation,
    ) -> infra::TestResult {
        let mut model = TypedModel::default();
        model.properties.insert("tract-rt-test.id".to_string(), rctensor0(id.to_string()));

        let inp = model.add_source(
            "inp",
            TypedFact::dt_shape(self.tensor_input.datum_type(), self.tensor_input.shape()),
        )?;
        // we need to wire correctly output to the provided operator {
        let mut op = self.operator.clone();
        op.1 = Some(self.out_dt);
        // }
        let out = model.wire_node("out", op, &[inp])?[0];
        model.set_output_outlets(&[out])?;

        let result = runtime
            .prepare(model)?
            .run(tvec![self.tensor_input.clone().into_tvalue()])?
            .remove(0)
            .into_tensor();
        self.check_ref_with_approx(result, approx)
    }
}

pub fn suite() -> TractResult<TestSuite> {
    let mut suite = TestSuite::default();
    suite.add_arbitrary::<QElmWiseOpProblem>("proptest", ());

    suite.add(
        "tanh_sweep_case",
        QElmWiseOpProblem {
            operator: tract_core::ops::math::tanh(),
            tensor_input: qu8_tensor1(&(0u8..=100).collect::<Box<[u8]>>(), 50, 0.05)?,
            out_dt: qu8_dt(127, 0.001),
        },
    );

    suite.add(
        "cos_switch_qi8_to_qu8_case",
        QElmWiseOpProblem {
            operator: tract_core::ops::math::cos(),
            tensor_input: qi8_tensor1(&[-16], 39, 0.5)?,
            out_dt: qu8_dt(2, 0.5),
        },
    );
    Ok(suite)
}
