use infra::{Test, TestSuite};
use proptest::prelude::*;
use tract_core::internal::*;

use crate::q_helpers::*;

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

impl QOpProblem for QBinaryOpProblem {
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
        let tested_operators = prop_oneof![
            Just(tract_core::ops::math::mul()),
            Just(tract_core::ops::math::div()),
            Just(tract_core::ops::math::add()),
            Just(tract_core::ops::math::sub()),
            Just(tract_core::ops::math::min()),
            Just(tract_core::ops::math::max()),
        ];
        (
            (1..20usize),
            any::<bool>(),
            any::<bool>(),
            any::<bool>(),
            (1..4usize),
            (1..4usize),
            (1..4usize),
            tested_operators,
        )
            .prop_flat_map(|(len, a_signed, b_signed, c_signed, a_scale, b_scale, c_scale, op)| {
                let a_dt = pick_signed_datum(a_signed);
                let b_dt = pick_signed_datum(b_signed);
                let c_dt = pick_signed_datum(c_signed);
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
                    Just(op),
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
                    op,
                )| {
                    let tensor_a =
                        build_qtensor(a_values.into_tensor(), a_dt, a_zp.into_tensor(), a_scale);
                    let tensor_b =
                        build_qtensor(b_values.into_tensor(), b_dt, b_zp.into_tensor(), b_scale);
                    let c_dt = c_dt.quantize(QParams::ZpScale {
                        zero_point: c_zp.into_tensor().cast_to_scalar::<i32>().unwrap(),
                        scale: c_scale,
                    });
                    QBinaryOpProblem { operator: op.to_owned(), tensor_a, tensor_b, c_dt }
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
            .prepare(model)
            .context("Preparing model for runtime")?
            .run(tvec![self.tensor_a.clone().into_tvalue()])
            .context("Running model with runtime")?
            .remove(0)
            .into_tensor();
        self.check_ref_with_approx(result, approx)
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
            tensor_a: qu8_tensor0(0u8, 0, 1.)?,
            tensor_b: qu8_tensor0(0u8, 0, 1.)?,
            c_dt: qu8_dt(0, 1.),
        },
    );

    suite.add(
        "trivial_mul_as_qu8_overflow_clamp",
        QBinaryOpProblem {
            operator: tract_core::ops::math::mul(),
            tensor_a: qu8_tensor1(&[1_u8, 2, 3, 128], 0, 1.)?,
            tensor_b: qu8_tensor1(&[4u8], 0, 1.)?,
            c_dt: qu8_dt(0, 1.),
        },
    );

    suite.add(
        "trivial_mul_as_qu8_non_neutral_scale_and_offset",
        QBinaryOpProblem {
            operator: tract_core::ops::math::mul(),
            tensor_a: qu8_tensor1(&[1_u8, 2, 3, 128], 3, 2.)?,
            tensor_b: qu8_tensor1(&[4u8], 3, 2.)?,
            c_dt: qu8_dt(3, 2.),
        },
    );

    suite.add(
        "trivial_mul_as_qu8_non_aligned_scale_and_offset",
        QBinaryOpProblem {
            operator: tract_core::ops::math::mul(),
            tensor_a: qu8_tensor1(&[3_u8, 4, 10, 25], 3, 4.5)?,
            tensor_b: qu8_tensor1(&[6u8], 4, 2.5)?,
            c_dt: qu8_dt(0, 1.),
        },
    );

    suite.add(
        "trivial_max_0_as_qu8_non_aligned_scale_and_offset",
        QBinaryOpProblem {
            operator: tract_core::ops::math::max(),
            tensor_a: qu8_tensor1(&[100_u8, 5, 110, 99], 100, 4.5)?,
            tensor_b: qu8_tensor1(&[100u8], 100, 4.5)?,
            c_dt: qu8_dt(0, 1.),
        },
    );

    suite.add(
        "trivial_min_15_as_qu8_non_aligned_scale_and_offset",
        QBinaryOpProblem {
            operator: tract_core::ops::math::min(),
            tensor_a: qu8_tensor1(&[5_u8, 9, 8, 20], 5, 4.)?,
            tensor_b: qu8_tensor1(&[15u8], 10, 3.)?,
            c_dt: qu8_dt(0, 1.),
        },
    );

    suite.add(
        "trivial_max_15_as_qu8_non_aligned_scale_and_offset",
        QBinaryOpProblem {
            operator: tract_core::ops::math::max(),
            tensor_a: qu8_tensor1(&[5_u8, 9, 8, 20], 5, 4.)?,
            tensor_b: qu8_tensor1(&[15u8], 10, 3.)?,
            c_dt: qu8_dt(0, 1.),
        },
    );

    suite.add(
        "trivial_add_as_qu8_non_aligned_scale_and_offset",
        QBinaryOpProblem {
            operator: tract_core::ops::math::add(),
            tensor_a: qu8_tensor1(&[3_u8, 4, 10, 25], 3, 4.5)?,
            tensor_b: qu8_tensor1(&[6u8], 4, 2.5)?,
            c_dt: qu8_dt(0, 1.),
        },
    );

    suite.add(
        "trivial_div_as_qu8_non_aligned_scale_and_offset",
        QBinaryOpProblem {
            operator: tract_core::ops::math::div(),
            tensor_a: qu8_tensor1(&[3_u8, 4, 10, 25], 3, 4.5)?,
            tensor_b: qu8_tensor1(&[6u8], 4, 2.5)?,
            c_dt: qu8_dt(0, 1.),
        },
    );

    suite.add(
        "bug_invalid_to_scalar_0",
        QBinaryOpProblem {
            operator: tract_core::ops::math::max(),
            tensor_a: qu8_tensor1(&[0u8, 0u8], 0, 0.5)?,
            tensor_b: qu8_tensor1(&[0u8, 0u8], 0, 0.5)?,
            c_dt: qu8_dt(0, 0.5),
        },
    );

    suite.add(
        "bug_invalid_to_scalar_1",
        QBinaryOpProblem {
            operator: tract_core::ops::math::max(),
            tensor_a: qu8_tensor1(&[0u8, 0u8], 0, 0.5)?,
            tensor_b: qu8_tensor1(&[0u8, 0u8], 0, 0.5)?,
            c_dt: qu8_dt(1, 0.5),
        },
    );

    suite.add(
        "bug_aligned_dt_0",
        QBinaryOpProblem {
            operator: tract_core::ops::math::mul(),
            tensor_a: qu8_tensor1(&[0u8, 0, 0, 0, 0], 95, 1.5)?,
            tensor_b: qu8_tensor1(&[0u8, 0, 0, 0, 0], 95, 1.5)?,
            c_dt: qu8_dt(95, 1.5),
        },
    );

    Ok(suite)
}
