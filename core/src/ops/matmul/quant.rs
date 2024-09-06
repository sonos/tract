use anyhow::ensure;
use ops::change_axes::wire_with_rank_broadcast;

use crate::internal::*;
use crate::ops;
use crate::ops::cast::cast;

/// Wires the offsetting of a matrix and zero point node.
///
/// Only wires nodes of u8 type and leaves nodes of different type untouched.
pub fn wire_ensure_q8_flavour(
    model: &mut TypedModel,
    prefix: &str,
    input: &mut OutletId,
    input_name: &str,
    zero_point: &mut OutletId,
    wanted_raw_dt: DatumType,
) -> TractResult<()> {
    ensure!(wanted_raw_dt.qparams().is_none());
    ensure!(wanted_raw_dt.size_of() == 1);
    let current = model.outlet_fact(*input)?.datum_type.unquantized();
    ensure!(current.size_of() == 1);
    ensure!(wanted_raw_dt.size_of() == 1);
    if model.outlet_fact(*zero_point)?.datum_type != i32::datum_type() {
        *zero_point = model.wire_node(
            format!("{prefix}.{input_name}_zp.cast"),
            cast(i32::datum_type()),
            &[*zero_point],
        )?[0];
    }
    if current == wanted_raw_dt {
        return Ok(());
    }
    let zp_rank = model.outlet_fact(*zero_point)?.rank();
    let i32_128 = model.add_const(
        format!("{prefix}.{input_name}.128"),
        tensor0(128i32).broadcast_into_rank(zp_rank)?,
    )?;
    if current.unquantized().is_signed() {
        *zero_point = model.wire_node(
            format!("{prefix}.offset_{input_name}_zp_as_u8"),
            ops::math::add(),
            &[*zero_point, i32_128],
        )?[0];
        *input = model.wire_node(
            format!("{prefix}.offset_{input_name}_as_u8"),
            ops::quant::offset_i8_as_u8(),
            &[*input],
        )?[0];
    } else {
        *zero_point = model.wire_node(
            format!("{prefix}.offset_{input_name}_zp_as_i8"),
            ops::math::sub(),
            &[*zero_point, i32_128],
        )?[0];
        *input = model.wire_node(
            format!("{prefix}.offset_{input_name}_as_i8"),
            ops::quant::offset_u8_as_i8(),
            &[*input],
        )?[0];
    }
    Ok(())
}

pub(crate) fn combine_scales(
    model: &mut TypedModel,
    name: &str,
    a_scale: OutletId,
    b_scale: OutletId,
    c_scale: OutletId,
) -> TractResult<OutletId> {
    let ab_scale = wire_with_rank_broadcast(
        format!("{name}.ab_scale"),
        model,
        ops::math::mul(),
        &[a_scale, b_scale],
    )?[0];
    let abc_scale = wire_with_rank_broadcast(
        format!("{name}.abc_scales"),
        model,
        ops::math::div(),
        &[ab_scale, c_scale],
    )?[0];
    Ok(abc_scale)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn compensate_zero_points(
    model: &mut TypedModel,
    name: &str,
    result: OutletId,
    k: TDim,
    a0: OutletId,
    b0: OutletId,
    sum_a: OutletId,
    sum_b: OutletId,
) -> TractResult<OutletId> {
    let output_rank = model.outlet_fact(result)?.rank();
    ensure!(model.outlet_fact(sum_a)?.rank() == output_rank);
    ensure!(model.outlet_fact(sum_b)?.rank() == output_rank);

    let a0 =
        model.wire_node(format!("{name}.cast_a0"), ops::cast::cast(i32::datum_type()), &[a0])?[0];

    let b0 =
        model.wire_node(format!("{name}.cast_b0"), ops::cast::cast(i32::datum_type()), &[b0])?[0];

    let k = model.add_const(format!("{name}.k"), rctensor0(k))?;
    let k = model.wire_node(format!("{name}.cast_k"), ops::cast::cast(i32::datum_type()), &[k])?[0];

    let a0_sum_b = wire_with_rank_broadcast(
        format!("{name}.a0_sum_b"),
        model,
        ops::math::mul(),
        &[a0, sum_b],
    )?[0];

    let b0_sum_a = wire_with_rank_broadcast(
        format!("{name}.b0_sum_a"),
        model,
        ops::math::mul(),
        &[b0, sum_a],
    )?[0];

    let a0_k =
        wire_with_rank_broadcast(format!("{name}.a0_k"), model, ops::math::mul(), &[a0, k])?[0];

    let a0_k_b0 =
        wire_with_rank_broadcast(format!("{name}.a0_k_b0"), model, ops::math::mul(), &[a0_k, b0])?
            [0];

    let result = wire_with_rank_broadcast(
        format!("{}.minus_a0_B", &name),
        model,
        ops::math::sub(),
        &[result, a0_sum_b],
    )?[0];
    let result = wire_with_rank_broadcast(
        format!("{}.minus_b0_A", &name),
        model,
        ops::math::sub(),
        &[result, b0_sum_a],
    )?[0];

    let result = wire_with_rank_broadcast(
        format!("{}.plus_a0_k_b0", &name),
        model,
        ops::math::add(),
        &[result, a0_k_b0],
    )?[0];

    Ok(result)
}

pub(crate) fn requant(
    model: &mut TypedModel,
    name: &str,
    wire: OutletId,
    dt: DatumType,
    scale: OutletId,
    zero_point: OutletId,
) -> TractResult<OutletId> {
    let wire = wire_with_rank_broadcast(
        format!("{name}.scale"),
        model,
        ops::quant::scale(),
        &[scale, wire],
    )?[0];

    let zero_point = model.wire_node(
        format!("{name}.cast_c0"),
        ops::cast::cast(i32::datum_type()),
        &[zero_point],
    )?[0];

    let wire = wire_with_rank_broadcast(
        format!("{name}.zeropoint"),
        model,
        ops::math::add(),
        &[wire, zero_point],
    )?[0];

    clamp_and_cast_to(model, name, dt, wire)
}

pub(crate) fn clamp_and_cast_to(
    model: &mut TypedModel,
    name: &str,
    dt: DatumType,
    wire: OutletId,
) -> TractResult<OutletId> {
    if dt == i32::datum_type() {
        return Ok(wire);
    }
    let rank = model.outlet_fact(wire)?.rank();
    let inf = dt
        .unquantized()
        .min_value()
        .cast_to_dt(DatumType::I32)?
        .into_owned()
        .broadcast_into_rank(rank)?
        .into_arc_tensor();
    let inf = model.add_const(format!("{name}.min.const"), inf)?;
    let sup = dt
        .unquantized()
        .max_value()
        .cast_to_dt(DatumType::I32)?
        .into_owned()
        .broadcast_into_rank(rank)?
        .into_arc_tensor();
    let sup = model.add_const(format!("{name}.max.const"), sup)?;
    let wire = model.wire_node(format!("{name}.min"), ops::math::min(), &[wire, sup])?;
    let wire = model.wire_node(format!("{name}.max"), ops::math::max(), &[wire[0], inf])?;
    let wire = model.wire_node(format!("{name}.cast"), ops::cast::cast(dt), &wire)?;
    Ok(wire[0])
}

#[cfg(test)]
mod test {
    use super::*;
    use proptest::collection::vec;
    use proptest::prelude::*;
    use tract_ndarray::prelude::*;

    proptest! {
        #[test]
        fn prop_i8_i8_i8(pb in any::<QMatMulProblemI8I8I8>()) {
            pb.check();
        }

        #[test]
        fn prop_i8_i8_u8(pb in any::<QMatMulProblemI8I8U8>()) {
            pb.check();
        }

        #[test]
        fn prop_i8_u8_i8(pb in any::<QMatMulProblemI8U8I8>()) {
            pb.check();
        }

        #[test]
        fn prop_u8_i8_i8(pb in any::<QMatMulProblemU8I8I8>()) {
            pb.check();
        }

        #[test]
        fn prop_i8_u8_u8(pb in any::<QMatMulProblemI8U8U8>()) {
            pb.check();
        }

        #[test]
        fn prop_u8_i8_u8(pb in any::<QMatMulProblemU8I8U8>()) {
            pb.check();
        }

        #[test]
        fn prop_u8_u8_i8(pb in any::<QMatMulProblemU8U8I8>()) {
            pb.check();
        }

        #[test]
        fn prop_u8_u8_u8(pb in any::<QMatMulProblemU8U8U8>()) {
            pb.check();
        }
    }

    #[test]
    fn c0() {
        QMatMulProblemI8I8I8 {
            a: arr2(&[[0]]),
            b: arr2(&[[0]]),
            bias: tensor0(0i32),
            a0: 0,
            b0: 0,
            c0: 1,
            a_scale: 1.0,
            b_scale: 1.0,
            c_scale: 1.0,
        }
        .check();
    }

    #[test]
    fn b_scale() {
        QMatMulProblemI8I8I8 {
            a: arr2(&[[0]]),
            b: arr2(&[[0]]),
            bias: tensor0(0i32),
            a0: 0,
            b0: 0,
            c0: 1,
            a_scale: 1.0,
            b_scale: 2.0,
            c_scale: 1.0,
        }
        .check();
    }

    #[test]
    fn sat() {
        QMatMulProblemI8I8I8 {
            a: arr2(&[[0]]),
            b: arr2(&[[34]]),
            bias: tensor0(0i32),
            a0: -17,
            b0: 1,
            c0: 0,
            a_scale: 1.0,
            b_scale: 0.05,
            c_scale: 0.25,
        }
        .check();
    }

    #[test]
    fn rounding() {
        QMatMulProblemI8I8I8 {
            a: arr2(&[[26]]),
            b: arr2(&[[0]]),
            bias: tensor0(0i32),
            a0: 27,
            b0: -1,
            c0: 1,
            a_scale: 1.0,
            b_scale: 0.05,
            c_scale: 1.0,
        }
        .check();
    }

    #[test]
    fn neg_rounding() {
        QMatMulProblemI8I8I8 {
            a: arr2(&[[-23]]),
            b: arr2(&[[-2]]),
            bias: tensor0(0i32),
            a0: -11,
            b0: -45,
            c0: 0,
            a_scale: 0.1,
            b_scale: 1.0,
            c_scale: 1.0,
        }
        .check();
    }

    #[test]
    fn rounding_ties_2() {
        QMatMulProblemI8I8I8 {
            a: arr2(&[[47], [0]]),
            b: arr2(&[[1, 0, 30]]),
            bias: tensor0(0i32),
            a0: 86,
            b0: 19,
            c0: 0,
            a_scale: 0.1,
            b_scale: 1.0,
            c_scale: 0.6,
        }
        .check();
    }

    #[test]
    fn rounding_ties_3() {
        QMatMulProblemI8I8I8 {
            a: arr2(&[[-30]]),
            b: arr2(&[[0, 107, 0]]),
            bias: tensor0(0i32),
            a0: -59,
            b0: 117,
            c0: 0,
            a_scale: 1.0,
            b_scale: 0.15,
            c_scale: 0.6,
        }
        .check();
    }

    #[test]
    fn onnx_test_matmulinteger() {
        QMatMulProblemI8I8I8 {
            a: arr2(&[[11, 7, 3], [10, 6, 2], [9, 5, 1], [8, 4, 0]]),
            b: arr2(&[[1, 4], [2, 5], [3, 6]]),
            bias: tensor0(0i32),
            a0: 12,
            b0: 0,
            c0: 0,
            a_scale: 1.0,
            b_scale: 1.0,
            c_scale: 1.0,
        }
        .check();
    }

    fn round_ties_to_right(x: f32) -> i32 {
        (x + 0.5).floor() as i32
    }

    fn scale() -> BoxedStrategy<f32> {
        prop_oneof![Just(1.0), (1i32..=20).prop_map(|x| x as f32 / 20.0)].boxed()
    }

    macro_rules! impl_qmmp {
        ($name:ident, $a:ty, $b:ty, $c:ty $(,)?) => {
            #[derive(Debug)]
            struct $name {
                a: Array2<$a>,
                b: Array2<$b>,
                bias: Tensor,
                a0: $a,
                b0: $b,
                c0: $c,
                a_scale: f32,
                b_scale: f32,
                c_scale: f32,
            }

            impl $name {
                fn check(&self) {
                    let check_with = |r: &Array2<$c>, opt: bool, qp: bool| {
                        let t = self.tract(opt, qp);
                        assert!(
                            r.iter().zip(t.iter()).all(|(r, t)| r.max(t) - r.min(t) <= 1),
                            "mismatch! optimized plan: {}, dynamic qparams: {}, reference: {:?}, tract: {:?}",
                            opt,
                            qp,
                            r,
                            t,
                            );
                    };

                    let r = self.reference();
                    check_with(&r, false, false);
                    check_with(&r, false, true);
                    check_with(&r, true, false);
                    check_with(&r, true, true);
                }

                fn reference(&self) -> Array2<$c> {
                    let a = self.a.map(|&x| (x as f32 - self.a0 as f32) * self.a_scale);
                    let b = self.b.map(|&x| (x as f32 - self.b0 as f32) * self.b_scale);
                    let c = a.dot(&b);
                    let c = c.map(|&x| round_ties_to_right(x / self.c_scale) + self.c0 as i32);
                    c.map(|&x| x.max(<$c>::MIN as i32).min(<$c>::MAX as i32) as $c)
                }

                fn tract(&self, opt: bool, qp: bool) -> Array2<$c> {
                    let mut model = TypedModel::default();
                    let mut inputs = tvec![];
                    inputs.push(
                        model
                        .add_source("a", <$a>::fact( &[self.a.nrows(), self.a.ncols()]))
                        .unwrap(),
                        );
                    inputs.push(
                        model
                        .add_source("b", <$b>::fact(&[self.b.nrows(), self.b.ncols()]))
                        .unwrap(),
                        );
                    inputs.push(
                        model
                        .add_source("bias", i32::fact(self.bias.shape()))
                        .unwrap(),
                        );
                    if qp {
                        inputs.push(model.add_source("a0", TypedFact::scalar::<$a>()).unwrap());
                        inputs
                            .push(model.add_source("a_scale", TypedFact::scalar::<f32>()).unwrap());
                        inputs.push(model.add_source("b0", TypedFact::scalar::<$b>()).unwrap());
                        inputs
                            .push(model.add_source("b_scale", TypedFact::scalar::<f32>()).unwrap());
                        inputs.push(model.add_source("c0", TypedFact::scalar::<$c>()).unwrap());
                        inputs
                            .push(model.add_source("c_scale", TypedFact::scalar::<f32>()).unwrap());
                    } else {
                        inputs.push(model.add_const("a0", rctensor0(self.a0)).unwrap());
                        inputs
                            .push(model.add_const("a_scale", rctensor0(self.a_scale)).unwrap());
                        inputs.push(model.add_const("b0", rctensor0(self.b0)).unwrap());
                        inputs
                            .push(model.add_const("b_scale", rctensor0(self.b_scale)).unwrap());
                        inputs.push(model.add_const("c0", rctensor0(self.c0)).unwrap());
                        inputs
                            .push(model.add_const("c_scale", rctensor0(self.c_scale)).unwrap());
                    };
                    let result = model
                        .wire_node(
                            "einsum",
                            crate::ops::einsum::EinSum {
                                axes: "mk,kn,,,,,,,->mn".parse().unwrap(),
                                operating_dt: i32::datum_type(),
                                q_params: Some(<$c>::datum_type())
                            },
                            &inputs,
                        ).unwrap();
                    model.set_output_outlets(&result).unwrap();

                    let inputs = if qp {
                        tvec![
                            self.a.clone().into_tensor(),
                            self.b.clone().into_tensor(),
                            self.bias.clone(),
                            self.a0.into(),
                            self.a_scale.into(),
                            self.b0.into(),
                            self.b_scale.into(),
                            self.c0.into(),
                            self.c_scale.into(),
                        ]
                    } else {
                        tvec![
                            self.a.clone().into_tensor(),
                            self.b.clone().into_tensor(),
                            self.bias.clone(),
                        ]
                    };
                    let inputs = inputs.into_iter().map(|t| t.into_tvalue()).collect();
                    let optimized = if opt { model.into_optimized().unwrap() } else { model };
                    let mut outputs = optimized.into_runnable()
                        .unwrap()
                        .run(inputs)
                        .unwrap();
                    outputs
                        .remove(0)
                        .into_tensor()
                        .into_array::<$c>()
                        .unwrap()
                        .into_dimensionality()
                        .unwrap()
                }
            }

            impl Arbitrary for $name {
                type Parameters = ();
                type Strategy = BoxedStrategy<$name>;
                fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
                    (1usize..=4, 1usize..=4, 1usize..=4)
                        .prop_flat_map(|(m, k, n)| {
                            (
                                Just((m, k, n)),
                                vec(any::<$a>(), m * k..=m * k),
                                vec(any::<$b>(), k * n..=k * n),
                                any::<$a>(),
                                any::<$b>(),
                                any::<$c>(),
                                scale(),
                                scale(),
                                scale(),
                                )
                        })
                    .prop_map(|((m, k, n), a, b, a0, b0, c0, a_scale, b_scale, c_scale)| {
                        $name {
                            a: Array2::from_shape_vec((m, k), a).unwrap(),
                            b: Array2::from_shape_vec((k, n), b).unwrap(),
                            bias: tensor0(0i32),
                            a0,
                            b0,
                            c0,
                            a_scale,
                            b_scale,
                            c_scale,
                        }
                    })
                    .boxed()
                }
            }
        };
    }

    impl_qmmp! { QMatMulProblemI8I8I8, i8, i8, i8 }
    impl_qmmp! { QMatMulProblemI8I8U8, i8, i8, u8 }
    impl_qmmp! { QMatMulProblemI8U8I8, i8, u8, i8 }
    impl_qmmp! { QMatMulProblemU8I8I8, u8, i8, i8 }
    impl_qmmp! { QMatMulProblemI8U8U8, i8, u8, u8 }
    impl_qmmp! { QMatMulProblemU8I8U8, u8, i8, u8 }
    impl_qmmp! { QMatMulProblemU8U8I8, u8, u8, i8 }
    impl_qmmp! { QMatMulProblemU8U8U8, u8, u8, u8 }

    #[test]
    fn test_qmmp_i8_i8_i8_0() {
        QMatMulProblemI8I8I8 {
            a: arr2(&[[76, 76, 76], [127, -127, 102]]),
            b: arr2(&[[25, 51, 76, 102, 127], [-51, -25, 0, 25, 51], [-25, -51, -76, -102, -127]]),
            bias: tensor0(0i32),
            a0: 51,
            b0: 0,
            c0: -31,
            a_scale: 0.039215688,
            b_scale: 0.039215688,
            c_scale: 0.09411765,
        }
        .check(); // c: [[-52, -41, -31, -21, -10], [127, 64, 0, -62, -126]]
    }

    #[test]
    fn test_qmmp_i8_i8_i8_1() {
        QMatMulProblemI8I8I8 {
            a: arr2(&[[-34, -2]]),
            b: arr2(&[[-79], [21]]),
            bias: tensor0(0i32),
            a0: -87,
            b0: -17,
            c0: 0,
            a_scale: 1.0,
            b_scale: 1.0,
            c_scale: 1.0,
        }
        .check(); // c: [[-52, -41, -31, -21, -10], [127, 64, 0, -62, -126]]
    }

    #[test]
    fn test_qmmp_i8_i8_u8() {
        QMatMulProblemI8I8U8 {
            a: arr2(&[[76, 76, 76], [127, -127, 102]]),
            b: arr2(&[[25, 51, 76, 102, 127], [-51, -25, 0, 25, 51], [-25, -51, -76, -102, -127]]),
            bias: tensor0(0i32),
            a0: 51,
            b0: 0,
            c0: 96,
            a_scale: 0.039215688,
            b_scale: 0.039215688,
            c_scale: 0.09411765,
        }
        .check(); // c: [[75, 86, 96, 106, 117], [255, 191, 127, 65, 1]]
    }

    #[test]
    fn test_qmmp_i8_u8_i8() {
        QMatMulProblemI8U8I8 {
            a: arr2(&[[76, 76, 76], [127, -127, 102]]),
            b: arr2(&[[152, 178, 203, 229, 254], [76, 102, 127, 152, 178], [102, 76, 51, 25, 0]]),
            bias: tensor0(0i32),
            a0: 51,
            b0: 127,
            c0: -31,
            a_scale: 0.039215688,
            b_scale: 0.039215688,
            c_scale: 0.09411765,
        }
        .check(); // c: [[-52, -41, -31, -21, -10], [127, 64, 0, -62, -126]]
    }

    #[test]
    fn test_qmmp_u8_i8_i8() {
        QMatMulProblemU8I8I8 {
            a: arr2(&[[204, 204, 204], [255, 1, 230]]),
            b: arr2(&[[25, 51, 76, 102, 127], [-51, -25, 0, 25, 51], [-25, -51, -76, -102, -127]]),
            bias: tensor0(0i32),
            a0: 179,
            b0: 0,
            c0: -31,
            a_scale: 0.039215688,
            b_scale: 0.039215688,
            c_scale: 0.09411765,
        }
        .check(); // c: [[-52, -41, -31, -21, -10], [127, 64, 0, -62, -126]]
    }

    #[test]
    fn test_qmmp_i8_u8_u8() {
        QMatMulProblemI8U8U8 {
            a: arr2(&[[76, 76, 76], [127, -127, 102]]),
            b: arr2(&[[152, 178, 203, 229, 254], [76, 102, 127, 152, 178], [102, 76, 51, 25, 0]]),
            bias: tensor0(0i32),
            a0: 51,
            b0: 127,
            c0: 96,
            a_scale: 0.039215688,
            b_scale: 0.039215688,
            c_scale: 0.09411765,
        }
        .check(); // c: [[75, 86, 96, 106, 117], [255, 191, 127, 65, 1]]
    }

    #[test]
    fn test_qmmp_u8_i8_u8() {
        QMatMulProblemU8I8U8 {
            a: arr2(&[[204, 204, 204], [255, 1, 230]]),
            b: arr2(&[[25, 51, 76, 102, 127], [-51, -25, 0, 25, 51], [-25, -51, -76, -102, -127]]),
            bias: tensor0(0i32),
            a0: 179,
            b0: 0,
            c0: 96,
            a_scale: 0.039215688,
            b_scale: 0.039215688,
            c_scale: 0.09411765,
        }
        .check(); // c: [[75, 86, 96, 106, 117], [255, 191, 127, 65, 1]]
    }

    #[test]
    fn test_qmmp_u8_u8_i8() {
        QMatMulProblemU8U8I8 {
            a: arr2(&[[204, 204, 204], [255, 1, 230]]),
            b: arr2(&[[152, 178, 203, 229, 254], [76, 102, 127, 152, 178], [102, 76, 51, 25, 0]]),
            bias: tensor0(0i32),
            a0: 179,
            b0: 127,
            c0: -31,
            a_scale: 0.039215688,
            b_scale: 0.039215688,
            c_scale: 0.09411765,
        }
        .check(); // c: [[-52, -41, -31, -21, -10], [127, 64, 0, -62, -126]]
    }

    #[test]
    fn test_qmmp_u8_u8_u8() {
        QMatMulProblemU8U8U8 {
            a: arr2(&[[204, 204, 204], [255, 1, 230]]),
            b: arr2(&[[152, 178, 203, 229, 254], [76, 102, 127, 152, 178], [102, 76, 51, 25, 0]]),
            bias: tensor0(0i32),
            a0: 179,
            b0: 127,
            c0: 96,
            a_scale: 0.039215688,
            b_scale: 0.039215688,
            c_scale: 0.09411765,
        }
        .check(); // c: [[75, 86, 96, 106, 117], [255, 191, 127, 65, 1]]
    }
}
