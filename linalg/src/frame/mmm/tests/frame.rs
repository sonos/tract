use crate::frame::mmm::pack::PackedFormat;
use crate::frame::mmm::*;
use crate::LADatum;
use num_traits::AsPrimitive;
use proptest::prelude::*;
use std::ops::Neg;
use tests::display_error;
use tract_data::internal::*;

#[macro_export]
macro_rules! mmm_frame_tests {
    ($cond:expr, $ker:ident, $ta:ty, $tb:ty, $tc:ty, $ti:ty) => {
        mod frame {
            use super::*;
            use tract_data::internal::*;
            #[allow(unused_imports)]
            use $crate::frame::mmm::tests::frame::*;

            proptest::proptest! {
                #[test]
                fn mat_mul_prepacked_prop((m, k, n, ref a, ref b) in strat_mat_mat_mul(& $ker, 0)) {
                    if $cond {
                        test_mat_mat_mul_prep::<_, $ta, $tb, $tc, $ti>($ker, m, k, n, &a, &b).unwrap();
                    }
                }
            }

            #[test]
            fn mat_mul_1() -> TractResult<()> {
                if $cond {
                    let a = tensor2(&[[-3i32, 3, 5, -5], [6, 0, -6, -5], [0, 0, 9, 7]])
                        .cast_to::<$ta>()?
                        .into_owned();
                    let b = tensor2(&[[-8i32, 5], [5, -3], [5, 7], [-8, -1]])
                        .cast_to::<$tb>()?
                        .into_owned();
                    test_mat_mat_mul_prep::<_, $ta, $tb, $tc, $ti>($ker, 3, 4, 2, &a, &b)?;
                }
                Ok(())
            }

            #[test]
            fn mat_mul_2() -> TractResult<()> {
                if $cond {
                    let a = tensor2(&[[1i32]]).cast_to::<$ta>()?.into_owned();
                    let b = tensor2(&[[0i32, 0, 1]]).cast_to::<$tb>()?.into_owned();
                    test_mat_mat_mul_prep::<_, $ta, $tb, $tc, $ti>($ker, 1, 1, 3, &a, &b)?
                }
                Ok(())
            }

            #[test]
            fn mat_mul_3() -> TractResult<()> {
                if $cond {
                    let a = tensor2(&[[-3i32, 3, 5, -5], [6, 0, -6, -5], [0, 0, 9, 7]])
                        .cast_to::<$ta>()
                        ?
                        .into_owned();
                    let b = tensor2(&[[-8i32, 5], [5, -3], [5, 7], [-8, -1]])
                        .cast_to::<$tb>()
                        ?
                        .into_owned();
                    test_mat_mat_mul_prep::<_, $ta, $tb, $tc, $ti>($ker, 3, 4, 2, &a, &b)?
                }
                Ok(())
            }

            #[test]
            fn mat_mul_4() -> TractResult<()> {
                if $cond {
                    let a = tensor2(&[[122, 82]]).cast_to::<$ta>()?.into_owned();
                    let b =
                        tensor2(&[[0, 0, 37], [0, 0, 57]]).cast_to::<$tb>()?.into_owned();
                    test_mat_mat_mul_prep::<_, $ta, $tb, $tc, $ti>($ker, 1, 2, 3, &a, &b)?
                }
                Ok(())
            }

            #[test]
            fn mat_mul_1_1_1() -> TractResult<()> {
                if $cond {
                    test_mat_mat_mul_prep::<_, $ta, $tb, $tc, $ti>(
                        $ker,
                        1,
                        1,
                        1,
                        &*tensor2(&[[26]]).cast_to::<$ta>()?,
                        &*tensor2(&[[48]]).cast_to::<$tb>()?,
                    )
                    ?
                }
                Ok(())
            }

            #[test]
            fn mat_mul_1_2_1() -> TractResult<()> {
                if $cond {
                    test_mat_mat_mul_prep::<_, $ta, $tb, $tc, $ti>(
                        $ker,
                        1,
                        2,
                        1,
                        &*tensor2(&[[0, 1]]).cast_to::<$ta>()?,
                        &*tensor2(&[[0], [1]]).cast_to::<$tb>()?,
                    )
                    ?
                }
                Ok(())
            }

            #[test]
            fn mat_vec_1() -> TractResult<()> {
                if $cond {
                    let a = tensor2(&[[0], [1]]).cast_to::<$ta>()?.into_owned();
                    let b = tensor2(&[[1]]).cast_to::<$tb>()?.into_owned();
                    test_mat_mat_mul_prep::<_, $ta, $tb, $tc, $ti>($ker, 2, 1, 1, &a, &b)?
                }
                Ok(())
            }

            #[test]
            fn mat_vec_2() -> TractResult<()> {
                if $cond {
                    let a = tensor1(&[0, 0, 0, 0, 0, 0, -4, 1]).into_shape(&[8, 1])?;
                    let a = a.cast_to::<$ta>()?;
                    let b = tensor2(&[[-64]]).cast_to::<$tb>()?.into_owned();
                    test_mat_mat_mul_prep::<_, $ta, $tb, $tc, $ti>($ker, 8, 1, 1, &a, &b)?
                }
                Ok(())
            }

            #[test]
            fn mat_vec_3() -> TractResult<()> {
                if $cond {
                    let a = tensor2(&[[0, 0]]);
                    let a = a.cast_to::<$ta>()?;
                    let b = tensor2(&[[0, 0]]).cast_to::<$tb>()?.into_owned();
                    test_mat_mat_mul_prep::<_, $ta, $tb, $tc, $ti>($ker, 1, 2, 1, &a, &b)?
                }
                Ok(())
            }

            #[test]
            fn row_mul_2_1_3() -> TractResult<()> {
                if $cond {
                    unsafe { row_mul::<_, $ta, $tb, $tc, $ti>($ker, 2, 3)? }
                }
                Ok(())
            }

            #[test]
            fn row_add_2_1_3() -> TractResult<()> {
                if $cond {
                    unsafe { row_add::<_, $ta, $tb, $tc, $ti>($ker, 2, 3)? }
                }
                Ok(())
            }

            #[test]
            fn col_mul_2_1_3() -> TractResult<()> {
                if $cond {
                    unsafe { col_mul::<_, $ta, $tb, $tc, $ti>($ker, 2, 3)? }
                }
                Ok(())
            }

            #[test]
            fn col_add_2_1_3() -> TractResult<()> {
                if $cond {
                    unsafe { col_add::<_, $ta, $tb, $tc, $ti>($ker, 2, 3)? }
                }
                Ok(())
            }

            #[test]
            fn max_2_1_3() -> TractResult<()> {
                if $cond {
                    unsafe { max::<_, $ta, $tb, $tc, $ti>($ker, 2, 3)? }
                }
                Ok(())
            }

            #[test]
            fn min_2_1_3() -> TractResult<()> {
                if $cond {
                    unsafe { min::<_, $ta, $tb, $tc, $ti>($ker, 2, 3)? }
                }
                Ok(())
            }

            #[test]
            fn add_d_2_1_3() -> TractResult<()> {
                if $cond {
                    unsafe { add_d::<_, $ta, $tb, $tc, $ti>($ker, 2, 3)? }
                }
                Ok(())
            }

            #[test]
            fn add_d_big() -> TractResult<()> {
                if $cond {
                    unsafe { add_d::<_, $ta, $tb, $tc, $ti>($ker, 197, 1)? }
                }
                Ok(())
            }
        }
    };
}

fn tensor(dt: DatumType, shape: Vec<usize>) -> BoxedStrategy<Tensor> {
    let len = shape.iter().product::<usize>();
    // for f16, positive numbers only to avoid worst rounding side effects
    // and not too big either to avoid overflow :)
    let number = if dt == f16::datum_type() {
        (0i16..100).boxed()
    } else {
        any::<i8>().prop_map(|i| i as i16).boxed()
    };
    proptest::collection::vec(number, len..=len)
        .prop_map(move |vec| {
            tract_ndarray::ArrayD::from_shape_vec(shape.clone(), vec)
                .unwrap()
                .into_tensor()
                .cast_to_dt(dt)
                .unwrap()
                .into_owned()
        })
        .boxed()
}

pub fn strat_mat_mat_mul(
    ker: &dyn MatMatMul,
    packing: usize,
) -> BoxedStrategy<(usize, usize, usize, Tensor, Tensor)> {
    let dta = ker.packings()[packing].0.downcast_ref::<PackedFormat>().unwrap().dt;
    let dtb = ker.packings()[packing].1.downcast_ref::<PackedFormat>().unwrap().dt;
    (1usize..5, 1usize..5, 1usize..5)
        .prop_flat_map(move |(m, k, n)| {
            (Just(m), Just(k), Just(n), tensor(dta, vec![m, k]), tensor(dtb, vec![k, n]))
        })
        .boxed()
}

pub fn test_mat_mat_mul_prep<K: MatMatMulKer<Acc = TI> + 'static, TA, TB, TC, TI>(
    ker: K,
    m: usize,
    k: usize,
    n: usize,
    a: &Tensor,
    b: &Tensor,
) -> TractResult<()>
where
    TA: LADatum + AsPrimitive<TI> + 'static,
    TB: LADatum + AsPrimitive<TI> + 'static,
    TC: LADatum + AsPrimitive<TI> + 'static,
    TI: LADatum + AsPrimitive<TC>,
    i32: AsPrimitive<TI>,
    usize: AsPrimitive<TI>,
{
    crate::setup_test_logger();
    assert_eq!(a.datum_type(), TA::datum_type());
    unsafe {
        let packing = 0;
        let pack = &ker.packings()[0];
        let packed_a = pack.0.prepare_tensor(a, 1, 0)?;
        let packed_b = pack.1.prepare_tensor(b, 0, 1)?;
        let a = a.as_slice::<TA>()?;
        let b = b.as_slice::<TB>()?;

        fused_ops::<K, TA, TB, TC, TI, _>(
            ker,
            m,
            n,
            &[FusedSpec::AddMatMul { a: &*packed_a, b: &*packed_b, packing }],
            |r, c| {
                let mut v: TI = TI::zero();
                for i in 0..k {
                    let a: TI = a[i + k * r].as_();
                    let b: TI = b[c + i * n].as_();
                    v += a * b;
                }
                v.as_()
            },
        )
    }
}

pub unsafe fn fused_ops<
    K: MatMatMulKer<Acc = TI> + 'static,
    TA,
    TB,
    TC,
    TI,
    F: Fn(usize, usize) -> TC,
>(
    ker: K,
    m: usize,
    n: usize,
    spec: &[FusedSpec],
    expect: F,
) -> TractResult<()>
where
    TA: LADatum + AsPrimitive<TI> + 'static,
    TB: LADatum + AsPrimitive<TI> + 'static,
    TC: LADatum + AsPrimitive<TI> + 'static,
    TI: LADatum + AsPrimitive<TC> + 'static,
    i32: AsPrimitive<TI>,
    usize: AsPrimitive<TI>,
{
    crate::setup_test_logger();

    let mut found = Tensor::zero::<TC>(&[m, n])?;
    let c_store = ker
        .c_from_data_and_strides(TC::datum_type().size_of(), n as isize, 1)
        .wrap(&found.view_mut());
    let mut spec: TVec<FusedSpec> = spec.into();
    spec.push(FusedSpec::Store(c_store));

    ker.run(m, n, &spec)?;
    let expected =
        tract_ndarray::prelude::Array2::from_shape_fn((m, n), |(r, c)| expect(r, c)).into_tensor();
    let err = found.close_enough(&expected, true);
    if err.is_err() {
        display_error(found.as_slice::<TC>()?, expected.as_slice::<TC>()?, m, n);
    }
    err
}

pub unsafe fn row_add<K: MatMatMulKer<Acc = TI> + 'static, TA, TB, TC, TI>(
    ker: K,
    m: usize,
    n: usize,
) -> TractResult<()>
where
    TA: LADatum + AsPrimitive<TI> + 'static,
    TB: LADatum + AsPrimitive<TI> + 'static,
    TC: LADatum + AsPrimitive<TI> + 'static,
    TI: LADatum + AsPrimitive<TC> + 'static + Neg<Output = TI>,
    i32: AsPrimitive<TI>,
    usize: AsPrimitive<TI>,
{
    let bias = (0..m).map(|i| i.as_()).collect::<Vec<TI>>();
    fused_ops::<K, TA, TB, TC, TI, _>(
        ker,
        m,
        n,
        &[FusedSpec::BinPerRow(tensor1(&bias).view(), BinOp::Add)],
        |r, _| bias[r].as_(),
    )
}

pub unsafe fn row_mul<K: MatMatMulKer<Acc = TI> + 'static, TA, TB, TC, TI>(
    ker: K,
    m: usize,
    n: usize,
) -> TractResult<()>
where
    TA: LADatum + AsPrimitive<TI> + 'static,
    TB: LADatum + AsPrimitive<TI> + 'static,
    TC: LADatum + AsPrimitive<TI> + 'static,
    TI: LADatum + AsPrimitive<TC> + 'static + Neg<Output = TI>,
    i32: AsPrimitive<TI>,
    usize: AsPrimitive<TI>,
{
    let bias = (0..m).map(|i| i.as_()).collect::<Vec<TI>>();
    fused_ops::<K, TA, TB, TC, TI, _>(
        ker,
        m,
        n,
        &[
            FusedSpec::BinScalar(&tensor0(1i32.as_()), BinOp::Add),
            FusedSpec::BinPerRow(tensor1(&bias).view(), BinOp::Mul),
        ],
        |r, _| bias[r].as_(),
    )
}

pub unsafe fn col_add<K: MatMatMulKer<Acc = TI> + 'static, TA, TB, TC, TI>(
    ker: K,
    m: usize,
    n: usize,
) -> TractResult<()>
where
    TA: LADatum + AsPrimitive<TI> + 'static,
    TB: LADatum + AsPrimitive<TI> + 'static,
    TC: LADatum + AsPrimitive<TI> + 'static,
    TI: LADatum + AsPrimitive<TC> + 'static + Neg<Output = TI>,
    i32: AsPrimitive<TI>,
    usize: AsPrimitive<TI>,
{
    let bias = (0..n).map(|i| i.as_()).collect::<Vec<TI>>();
    fused_ops::<K, TA, TB, TC, TI, _>(
        ker,
        m,
        n,
        &[FusedSpec::BinPerCol(tensor1(&bias).view(), BinOp::Add)],
        |_, c| bias[c].as_(),
    )
}

pub unsafe fn col_mul<K: MatMatMulKer<Acc = TI> + 'static, TA, TB, TC, TI>(
    ker: K,
    m: usize,
    n: usize,
) -> TractResult<()>
where
    TA: LADatum + AsPrimitive<TI> + 'static,
    TB: LADatum + AsPrimitive<TI> + 'static,
    TC: LADatum + AsPrimitive<TI> + 'static,
    TI: LADatum + AsPrimitive<TC> + 'static + Neg<Output = TI>,
    i32: AsPrimitive<TI>,
    usize: AsPrimitive<TI>,
{
    let bias = (0..n).map(|i| i.as_()).collect::<Vec<TI>>();
    fused_ops::<K, TA, TB, TC, TI, _>(
        ker,
        m,
        n,
        &[
            FusedSpec::BinScalar(&tensor0(1i32.as_()), BinOp::Add),
            FusedSpec::BinPerCol(tensor1(&bias).view(), BinOp::Mul),
        ],
        |_, c| bias[c].as_(),
    )
}

pub unsafe fn add_d<K: MatMatMulKer<Acc = TI> + 'static, TA, TB, TC, TI>(
    ker: K,
    m: usize,
    n: usize,
) -> TractResult<()>
where
    TA: LADatum + AsPrimitive<TI> + 'static,
    TB: LADatum + AsPrimitive<TI> + 'static,
    TC: LADatum + AsPrimitive<TI> + 'static,
    TI: LADatum + AsPrimitive<TC> + 'static + Neg<Output = TI>,
    i32: AsPrimitive<TI>,
    usize: AsPrimitive<TI>,
{
    let d = (0..m * n).map(|i| i.as_()).collect::<Vec<TI>>();
    let d = tensor1(&d).into_shape(&[m, n])?;
    let store_spec = OutputStoreSpec::View { m_axis: 0, n_axis: 1, mr: ker.mr(), nr: ker.nr() };
    let view_d = d.to_array_view::<TI>()?.into_dimensionality()?;
    fused_ops::<K, TA, TB, TC, TI, _>(
        ker,
        m,
        n,
        &[FusedSpec::AddUnicast(store_spec.wrap(&d.view()))],
        |r, c| view_d[(r, c)].as_(),
    )
}

pub unsafe fn max<K: MatMatMulKer<Acc = TI>, TA, TB, TC, TI>(
    ker: K,
    m: usize,
    n: usize,
) -> TractResult<()>
where
    TA: LADatum + AsPrimitive<TI> + 'static,
    TB: LADatum + AsPrimitive<TI> + 'static,
    TC: LADatum + AsPrimitive<TI> + 'static,
    TI: LADatum + AsPrimitive<TC> + 'static + Neg<Output = TI>,
    i32: AsPrimitive<TI>,
    usize: AsPrimitive<TI>,
{
    let five: TI = 5.as_();
    fused_ops::<K, TA, TB, TC, TI, _>(
        ker,
        m,
        n,
        &[FusedSpec::BinScalar(&tensor0(five), BinOp::Max)],
        |_, _| five.as_(),
    )
}

pub unsafe fn min<K: MatMatMulKer<Acc = TI>, TA, TB, TC, TI>(
    ker: K,
    m: usize,
    n: usize,
) -> TractResult<()>
where
    TA: LADatum + AsPrimitive<TI> + 'static,
    TB: LADatum + AsPrimitive<TI> + 'static,
    TC: LADatum + AsPrimitive<TI> + 'static,
    TI: LADatum + AsPrimitive<TC> + 'static + Neg<Output = TI>,
    i32: AsPrimitive<TI>,
    usize: AsPrimitive<TI>,
{
    let five: TI = 5.as_();
    fused_ops::<K, TA, TB, TC, TI, _>(
        ker,
        m,
        n,
        &[FusedSpec::BinScalar(&tensor0(five), BinOp::Min)],
        |_, _| TC::zero(),
    )
}
