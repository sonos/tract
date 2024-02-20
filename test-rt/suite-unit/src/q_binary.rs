use infra::{Test, TestSuite};
use proptest::prelude::*;
use tract_core::internal::*;

use crate::conv_q::qtensor;

#[derive(Debug, Clone)] //, Default
struct QBinaryOpProblem {
    operator: tract_core::ops::binary::TypedBinOp,
    tensor_a: Tensor,
    tensor_b: Tensor,
    c_dt: DatumType,
}

impl QBinaryOpProblem {
    fn pick_signed_datum(signed: bool) -> DatumType {
        if signed {
            DatumType::I8
        } else {
            DatumType::U8
        }
    }

    fn get_qtensor(values: Tensor, dt: DatumType, zp: Tensor, scale: f32) -> Tensor {
        let mut values = values;
        let zp = zp.cast_to_scalar::<i32>().unwrap();
        let dt = dt.quantize(QParams::ZpScale { zero_point: zp, scale });
        unsafe {
            values.set_datum_type(dt);
        }
        values
    }

    fn reference_float_ops(&self) -> TractResult<Tensor> {
        let a = self.tensor_a.cast_to::<f32>()?.clone().into_owned();
        let b = self.tensor_b.cast_to::<f32>()?.clone().into_owned();
        Ok(self.operator.eval(tvec![a.into_tvalue(), b.into_tvalue()])?.remove(0).into_tensor())
    }
}

impl Arbitrary for QBinaryOpProblem {
    type Parameters = ();
    type Strategy = BoxedStrategy<QBinaryOpProblem>;

    fn arbitrary_with(_: Self::Parameters) -> Self::Strategy {
        (
            (1..20usize),
            any::<bool>(),
            any::<bool>(),
            any::<bool>(),
            (1..4usize),
            (1..4usize),
            (1..4usize),
        )
            .prop_flat_map(|(len, a_signed, b_signed, c_signed, a_scale, b_scale, c_scale)| {
                let a_dt = Self::pick_signed_datum(a_signed);
                let b_dt = Self::pick_signed_datum(b_signed);
                let c_dt = Self::pick_signed_datum(c_signed);
                fn just_scale(scale: usize) -> Just<f32> {
                    Just(scale as f32 * 0.5)
                }
                (
                    // tensor a
                    Just(a_dt),
                    qtensor(vec![1], a_dt),
                    just_scale(a_scale),
                    qtensor(vec![len], a_dt),
                    // tensor b
                    Just(b_dt),
                    qtensor(vec![1], b_dt),
                    just_scale(b_scale),
                    qtensor(vec![len], b_dt),
                    // dt of c
                    Just(c_dt),
                    qtensor(vec![1], c_dt),
                    just_scale(c_scale),
                )
            })
            .prop_map(
                |(
                    a_dt,
                    a_zp,
                    a_scale,
                    a_values,
                    b_dt,
                    b_zp,
                    b_scale,
                    b_values,
                    c_dt,
                    c_zp,
                    c_scale,
                )| {
                    let tensor_a = Self::get_qtensor(
                        a_values.into_tensor(),
                        a_dt,
                        a_zp.into_tensor(),
                        a_scale,
                    );
                    let tensor_b = Self::get_qtensor(
                        b_values.into_tensor(),
                        b_dt,
                        b_zp.into_tensor(),
                        b_scale,
                    );
                    let c_dt = c_dt.quantize(QParams::ZpScale {
                        zero_point: c_zp.into_tensor().cast_to_scalar::<i32>().unwrap(),
                        scale: c_scale,
                    });
                    QBinaryOpProblem {
                        operator: tract_core::ops::math::mul(),
                        tensor_a,
                        tensor_b,
                        c_dt,
                    }
                },
            )
            .boxed()
    }
}

impl Test for QBinaryOpProblem {
    fn run_with_approx(
        &self,
        id: &str,
        runtime: &dyn Runtime,
        approx: Approximation,
    ) -> infra::TestResult {
        let mut model = TypedModel::default();
        model.properties.insert("tract-rt-test.id".to_string(), rctensor0(id.to_string()));

        let a = model.add_source(
            "a",
            TypedFact::dt_shape(self.tensor_a.datum_type(), self.tensor_a.shape()),
        )?;
        let b = model.add_const("b", self.tensor_b.clone().into_arc_tensor())?;
        // we need to wire correctly output to the provided operator {
        let mut op = self.operator.clone();
        op.1 = Some(self.c_dt);
        // }
        let c = model.wire_node("c", op, &[a, b])?[0];
        model.set_output_outlets(&[c])?;

        let result = runtime
            .prepare(model)?
            .run(tvec![self.tensor_a.clone().into_tvalue()])?
            .remove(0)
            .into_tensor();
        let mut reference = self.reference_float_ops()?;

        let (zero_point, scale) = self.c_dt.zp_scale();
        let min_repr_val = (*self.c_dt.min_value().to_scalar::<f32>()? - zero_point as f32) * scale;
        let max_repr_val = (*self.c_dt.max_value().to_scalar::<f32>()? - zero_point as f32) * scale;

        reference.to_array_view_mut()?.iter_mut().for_each(|x: &mut f32| {
            *x = round_ties_to_even((*x).clamp(min_repr_val, max_repr_val))
        });

        let mut comparison = result.cast_to::<f32>()?.into_owned();
        comparison.to_array_view_mut()?.iter_mut().for_each(|x: &mut f32| {
            *x = round_ties_to_even((*x).clamp(min_repr_val, max_repr_val))
        });

        dbg!(min_repr_val, max_repr_val);
        comparison.close_enough(&reference, approx)
    }
}

pub fn suite() -> TractResult<TestSuite> {
    let mut suite = TestSuite::default();
    // suite.add_arbitrary::<QBinaryOpProblem>("proptest", ());
    suite.add(
        "trivial_0",
        QBinaryOpProblem {
            operator: tract_core::ops::math::mul(),
            tensor_a: tensor0(0u8)
                .cast_to_dt(
                    u8::datum_type().quantize(QParams::ZpScale { zero_point: 0, scale: 1. }),
                )
                .unwrap()
                .into_owned(),
            tensor_b: tensor0(0u8)
                .cast_to_dt(
                    u8::datum_type().quantize(QParams::ZpScale { zero_point: 0, scale: 1. }),
                )
                .unwrap()
                .into_owned(),
            c_dt: DatumType::QU8(QParams::ZpScale { zero_point: 0, scale: 1. }),
        },
    );
    Ok(suite)
}
