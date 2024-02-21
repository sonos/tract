use infra::{Test, TestSuite};
use proptest::prelude::*;
use tract_core::internal::*;

use crate::conv_q::qtensor;

#[derive(Debug, Clone)]
struct QBinaryOpProblem {
    operator: tract_core::ops::binary::TypedBinOp,
    tensor_a: Tensor,
    tensor_b: Tensor,
    c_dt: DatumType,
}

impl Default for QBinaryOpProblem {
    fn default() -> QBinaryOpProblem {
        QBinaryOpProblem {
            operator: tract_core::ops::math::mul(),
            tensor_a: Tensor::default(),
            tensor_b: Tensor::default(),
            c_dt: DatumType::QU8(QParams::ZpScale { zero_point: 0, scale: 1. }),
        }
    }
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
            (1..6usize),
        )
            .prop_flat_map(
                |(len, a_signed, b_signed, c_signed, a_scale, b_scale, c_scale, op_index)| {
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
                        Just(op_index),
                    )
                },
            )
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
                    op_index,
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
                    let ops = [
                        tract_core::ops::math::mul(),
                        tract_core::ops::math::div(),
                        tract_core::ops::math::add(),
                        tract_core::ops::math::sub(),
                        tract_core::ops::math::min(),
                        tract_core::ops::math::max(),
                    ];
                    QBinaryOpProblem {
                        operator: ops[op_index].to_owned(),
                        tensor_a,
                        tensor_b,
                        c_dt,
                    }
                },
            )
            .prop_filter("div does not allow 0 divisor", |q_prob| {
                !(q_prob.operator.name().to_string().as_str().to_lowercase() == "div"
                    && q_prob
                        .tensor_b
                        .to_owned()
                        .cast_to_dt(DatumType::F32)
                        .unwrap()
                        .to_array_view()
                        .unwrap()
                        .iter()
                        .any(|x: &f32| *x == 0.0))
            })
            .boxed()
    }
}

impl Test for QBinaryOpProblem {
    #[warn(unused_variables)]
    fn run_with_approx(
        &self,
        _suite: &str,
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
        let min_repr_val = (self.c_dt.unquantized().min_value().cast_to_scalar::<f32>()?
            - zero_point as f32)
            * scale;
        let max_repr_val = (self.c_dt.unquantized().max_value().cast_to_scalar::<f32>()?
            - zero_point as f32)
            * scale;

        reference
            .to_array_view_mut()?
            .iter_mut()
            .for_each(|x: &mut f32| *x = (*x).clamp(min_repr_val, max_repr_val));

        let mut diff = result.cast_to::<f32>()?.into_owned();

        let acceptable_scale_error_ratio = match approx {
            Approximation::Exact => 0.,
            Approximation::Approximate => 1.,
            _ => 2.,
        };
        tract_core::ndarray::Zip::from(diff.to_array_view_mut()?)
            .and(reference.to_array_view()?)
            .all(|x: &mut f32, xref: &f32| {
                let closest_x = (*x).clamp(min_repr_val, max_repr_val);
                // core maximal accepted distance by default
                let distance = if &closest_x < xref {
                    (xref - closest_x).abs()
                } else {
                    (closest_x - xref).abs()
                };
                distance <= scale * acceptable_scale_error_ratio
            });
        Ok(())
    }
}

pub fn suite() -> TractResult<TestSuite> {
    let mut suite = TestSuite::default();
    suite.add_arbitrary::<QBinaryOpProblem>("proptest", ());

    // simplification 0 at declutter constant
    suite.add(
        "trivial_mul_0_case",
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

    suite.add(
        "trivial_mul_as_qu8_overflow_clamp",
        QBinaryOpProblem {
            operator: tract_core::ops::math::mul(),
            tensor_a: tensor1(&[1_u8, 2, 3, 128])
                .cast_to_dt(
                    u8::datum_type().quantize(QParams::ZpScale { zero_point: 0, scale: 1. }),
                )
                .unwrap()
                .into_owned(),
            tensor_b: tensor1(&[4u8])
                .cast_to_dt(
                    u8::datum_type().quantize(QParams::ZpScale { zero_point: 0, scale: 1. }),
                )
                .unwrap()
                .into_owned(),
            c_dt: DatumType::QU8(QParams::ZpScale { zero_point: 0, scale: 1. }),
        },
    );

    suite.add(
        "trivial_mul_as_qu8_non_neutral_scale_and_offset",
        QBinaryOpProblem {
            operator: tract_core::ops::math::mul(),
            tensor_a: tensor1(&[1_u8, 2, 3, 128])
                .cast_to_dt(
                    u8::datum_type().quantize(QParams::ZpScale { zero_point: 3, scale: 2. }),
                )
                .unwrap()
                .into_owned(),
            tensor_b: tensor1(&[4u8])
                .cast_to_dt(
                    u8::datum_type().quantize(QParams::ZpScale { zero_point: 3, scale: 2. }),
                )
                .unwrap()
                .into_owned(),
            c_dt: DatumType::QU8(QParams::ZpScale { zero_point: 3, scale: 2. }),
        },
    );

    suite.add(
        "trivial_mul_as_qu8_non_aligned_scale_and_offset",
        QBinaryOpProblem {
            operator: tract_core::ops::math::mul(),
            tensor_a: tensor1(&[3_u8, 4, 10, 25])
                .cast_to_dt(
                    u8::datum_type().quantize(QParams::ZpScale { zero_point: 3, scale: 4.5 }),
                )
                .unwrap()
                .into_owned(),
            tensor_b: tensor1(&[6u8])
                .cast_to_dt(
                    u8::datum_type().quantize(QParams::ZpScale { zero_point: 4, scale: 2.5 }),
                )
                .unwrap()
                .into_owned(),
            c_dt: DatumType::QU8(QParams::ZpScale { zero_point: 0, scale: 1. }),
        },
    );

    suite.add(
        "trivial_max_0_as_qu8_non_aligned_scale_and_offset",
        QBinaryOpProblem {
            operator: tract_core::ops::math::max(),
            tensor_a: tensor1(&[100_u8, 5, 110, 99])
                .cast_to_dt(
                    u8::datum_type().quantize(QParams::ZpScale { zero_point: 100, scale: 4.5 }),
                )
                .unwrap()
                .into_owned(),
            tensor_b: tensor1(&[100u8])
                .cast_to_dt(
                    u8::datum_type().quantize(QParams::ZpScale { zero_point: 100, scale: 4.5 }),
                )
                .unwrap()
                .into_owned(),
            c_dt: DatumType::QU8(QParams::ZpScale { zero_point: 0, scale: 1. }),
        },
    );

    suite.add(
        "trivial_min_15_as_qu8_non_aligned_scale_and_offset",
        QBinaryOpProblem {
            operator: tract_core::ops::math::min(),
            tensor_a: tensor1(&[5_u8, 9, 8, 20])
                .cast_to_dt(
                    u8::datum_type().quantize(QParams::ZpScale { zero_point: 5, scale: 4. }),
                )
                .unwrap()
                .into_owned(),
            tensor_b: tensor1(&[15u8])
                .cast_to_dt(
                    u8::datum_type().quantize(QParams::ZpScale { zero_point: 10, scale: 3. }),
                )
                .unwrap()
                .into_owned(),
            c_dt: DatumType::QU8(QParams::ZpScale { zero_point: 0, scale: 1. }),
        },
    );

    suite.add(
        "trivial_max_15_as_qu8_non_aligned_scale_and_offset",
        QBinaryOpProblem {
            operator: tract_core::ops::math::max(),
            tensor_a: tensor1(&[5_u8, 9, 8, 20])
                .cast_to_dt(
                    u8::datum_type().quantize(QParams::ZpScale { zero_point: 5, scale: 4. }),
                )
                .unwrap()
                .into_owned(),
            tensor_b: tensor1(&[15u8])
                .cast_to_dt(
                    u8::datum_type().quantize(QParams::ZpScale { zero_point: 10, scale: 3. }),
                )
                .unwrap()
                .into_owned(),
            c_dt: DatumType::QU8(QParams::ZpScale { zero_point: 0, scale: 1. }),
        },
    );

    suite.add(
        "trivial_add_as_qu8_non_aligned_scale_and_offset",
        QBinaryOpProblem {
            operator: tract_core::ops::math::add(),
            tensor_a: tensor1(&[3_u8, 4, 10, 25])
                .cast_to_dt(
                    u8::datum_type().quantize(QParams::ZpScale { zero_point: 3, scale: 4.5 }),
                )
                .unwrap()
                .into_owned(),
            tensor_b: tensor1(&[6u8])
                .cast_to_dt(
                    u8::datum_type().quantize(QParams::ZpScale { zero_point: 4, scale: 2.5 }),
                )
                .unwrap()
                .into_owned(),
            c_dt: DatumType::QU8(QParams::ZpScale { zero_point: 0, scale: 1. }),
        },
    );

    suite.add(
        "trivial_div_as_qu8_non_aligned_scale_and_offset",
        QBinaryOpProblem {
            operator: tract_core::ops::math::div(),
            tensor_a: tensor1(&[3_u8, 4, 10, 25])
                .cast_to_dt(
                    u8::datum_type().quantize(QParams::ZpScale { zero_point: 3, scale: 4.5 }),
                )
                .unwrap()
                .into_owned(),
            tensor_b: tensor1(&[6u8])
                .cast_to_dt(
                    u8::datum_type().quantize(QParams::ZpScale { zero_point: 4, scale: 2.5 }),
                )
                .unwrap()
                .into_owned(),
            c_dt: DatumType::QU8(QParams::ZpScale { zero_point: 0, scale: 1. }),
        },
    );

    Ok(suite)
}
