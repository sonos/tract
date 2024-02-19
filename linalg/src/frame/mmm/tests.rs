use super::*;
use crate::LADatum;
use num_traits::AsPrimitive;
use proptest::prelude::*;
use std::ops::Neg;
use tract_data::internal::*;

#[macro_export]
macro_rules! mmm_frame_tests {
    ($cond:expr, $ker:ident, $ta:ty, $tb:ty, $tc:ty, $ti:ty) => {
        mod frame {
            #[allow(unused_imports)]
            use $crate::frame::mmm::tests::*;
            use tract_data::internal::*;
            use super::super::$ker;

            proptest::proptest! {
                #[test]
                fn mat_mul_prepacked_prop((m, k, n, ref a, ref b) in strat_mat_mat_mul::<$ta, $tb>()) {
                    if $cond {
                        test_mat_mat_mul_prep::<$ker, $ta, $tb, $tc, $ti>(m, k, n, &a, &b)?
                    }
                }

                #[test]
                fn mat_vec_prepacked_prop((m, k, ref a, ref b) in strat_mat_vec_mul::<$ta, $tb>()) {
                    if $cond {
                        test_mat_vec_mul_prep::<$ker, $ta, $tb, $tc, $ti>(m, k, &*a, b)?
                    }
                }
            }

            #[test]
            fn mat_mul_1() {
                if $cond {
                    let a = tensor2(&[[-3i32, 3, 5, -5], [6, 0, -6, -5], [0, 0, 9, 7]]).cast_to::<$ta>().unwrap().into_owned();
                    let b = tensor2(&[[-8i32, 5],[ 5, -3], [5, 7],[ -8, -1]]).cast_to::<$tb>().unwrap().into_owned();
                    test_mat_mat_mul_prep::<$ker, $ta, $tb, $tc, $ti>(3, 4, 2, &a, &b).unwrap()
                }
            }

            #[test]
            fn mat_mul_2() {
                if $cond {
                    let a = tensor2(&[[1i32]]).cast_to::<$ta>().unwrap().into_owned();
                    let b = tensor2(&[[0i32, 0, 1]]).cast_to::<$tb>().unwrap().into_owned();
                    test_mat_mat_mul_prep::<$ker, $ta, $tb, $tc, $ti>(1, 1, 3, &a, &b).unwrap()
                }
            }

            #[test]
            fn mat_mul_3() {
                if $cond {
                    let a = tensor2(&[[-3i32, 3, 5, -5], [6, 0, -6, -5], [0, 0, 9, 7]]).cast_to::<$ta>().unwrap().into_owned();
                    let b = tensor2(&[[-8i32, 5],[ 5, -3], [5, 7],[ -8, -1]]).cast_to::<$tb>().unwrap().into_owned();
                    test_mat_mat_mul_prep::<$ker, $ta, $tb, $tc, $ti>(3, 4, 2, &a, &b).unwrap()
                }
            }

            #[test]
            fn mat_mul_4() {
                if $cond {
                    let a = tensor2(&[[122, 82]]).cast_to::<$ta>().unwrap().into_owned();
                    let b = tensor2(&[[0, 0, 37],[ 0, 0, 57]]).cast_to::<$tb>().unwrap().into_owned();
                    test_mat_mat_mul_prep::<$ker, $ta, $tb, $tc, $ti>(1, 2, 3, &a, &b).unwrap()
                }
            }

            #[test]
            fn mat_mul_1_2_1() {
                if $cond {
                    test_mat_mat_mul_prep::<$ker, $ta, $tb, $tc, $ti>(
                        1,
                        2,
                        1,
                        &tensor2(&[[0, 1]]).cast_to::<$ta>().unwrap(),
                        &tensor2(&[[0], [1]]).cast_to::<$tb>().unwrap(),
                        )
                        .unwrap()
                }
            }

            #[test]
            fn mat_vec_1() {
                if $cond {
                    let a = tensor2(&[[0], [1]]).cast_to::<$ta>().unwrap().into_owned();
                    let b = tensor1(&[1]).cast_to::<$tb>().unwrap().into_owned();
                    test_mat_vec_mul_prep::<$ker, $ta, $tb, $tc, $ti>(2, 1, &a, &b).unwrap()
                }
            }

            #[test]
            fn mat_vec_2() {
                if $cond {
                    let a = tensor1(&[0, 0, 0, 0, 0, 0, -4, 1]).into_shape(&[8,1]).unwrap();
                    let a = a.cast_to::<$ta>().unwrap();
                    let b = tensor1(&[-64]).cast_to::<$tb>().unwrap().into_owned();
                    test_mat_vec_mul_prep::<$ker, $ta, $tb, $tc, $ti>(8, 1, &a, &b).unwrap()
                }
            }

            #[test]
            fn mat_vec_3() {
                if $cond {
                    let a = tensor1(&[0, 0]).into_shape(&[1, 2]).unwrap();
                    let a = a.cast_to::<$ta>().unwrap();
                    let b = tensor1(&[0, 0]).cast_to::<$tb>().unwrap().into_owned();
                    test_mat_vec_mul_prep::<$ker, $ta, $tb, $tc, $ti>(1, 2, &a, &b).unwrap()
                }
            }

            #[test]
            fn row_mul_2_1_3() {
                if $cond {
                    unsafe { row_mul::<$ker, $ta, $tb, $tc, $ti>(2, 3).unwrap() }
                }
            }

            #[test]
            fn row_add_2_1_3() {
                if $cond {
                    unsafe { row_add::<$ker, $ta, $tb, $tc, $ti>(2, 3).unwrap() }
                }
            }

            #[test]
            fn col_mul_2_1_3() {
                if $cond {
                    unsafe { col_mul::<$ker, $ta, $tb, $tc, $ti>(2, 3).unwrap() }
                }
            }

            #[test]
            fn col_add_2_1_3() {
                if $cond {
                    unsafe { col_add::<$ker, $ta, $tb, $tc, $ti>(2, 3).unwrap() }
                }
            }

            #[test]
            fn max_2_1_3() {
                if $cond {
                    unsafe { max::<$ker, $ta, $tb, $tc, $ti>(2, 3).unwrap() }
                }
            }

            #[test]
            fn min_2_1_3() {
                if $cond {
                    unsafe { min::<$ker, $ta, $tb, $tc, $ti>(2, 3).unwrap() }
                }
            }

            #[test]
            fn add_d_2_1_3() {
                if $cond {
                    unsafe { add_d::<$ker, $ta, $tb, $tc, $ti>(2, 3).unwrap() }
                }
            }

            #[test]
            fn add_d_big() {
                if $cond {
                    unsafe { add_d::<$ker, $ta, $tb, $tc, $ti>(197, 1).unwrap() }
                }
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

pub fn strat_mat_mat_mul<TA: LADatum, TB: LADatum>(
) -> BoxedStrategy<(usize, usize, usize, Tensor, Tensor)> {
    (1usize..5, 1usize..5, 1usize..5)
        .prop_flat_map(move |(m, k, n)| {
            (
                Just(m),
                Just(k),
                Just(n),
                tensor(TA::datum_type(), vec![m, k]),
                tensor(TB::datum_type(), vec![k, n]),
            )
        })
        .boxed()
}

pub fn strat_mat_vec_mul<TA: LADatum, TB: LADatum>() -> BoxedStrategy<(usize, usize, Tensor, Tensor)>
{
    (1usize..15, 1usize..15)
        .prop_flat_map(move |(m, k)| {
            (
                Just(m),
                Just(k),
                tensor(TA::datum_type(), vec![m, k]),
                tensor(TB::datum_type(), vec![k, 1]),
            )
        })
        .boxed()
}

pub fn test_mat_mat_mul_prep<K: MatMatMulKer<TI> + 'static, TA, TB, TC, TI>(
    m: usize,
    k: usize,
    n: usize,
    a: &Tensor,
    b: &Tensor,
) -> Result<(), proptest::test_runner::TestCaseError>
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
    let op = MatMatMulImpl::<K, TI>::default();
    unsafe {
        let mut packed_a =
            Tensor::uninitialized_aligned::<TA>(&[op.a_pack().len(k, m)], op.a_pack().alignment())
                .unwrap();
        op.a_pack().pack(packed_a.view_mut(), a.view(), 1, 0);

        let mut packed_b =
            Tensor::uninitialized_aligned::<TB>(&[op.b_pack().len(k, n)], op.b_pack().alignment())
                .unwrap();
        op.b_pack().pack(packed_b.view_mut(), b.view(), 0, 1);

        fused_ops::<K, TA, TB, TC, TI, _>(
            m,
            n,
            &[FusedSpec::AddMatMul {
                a: op.a_packed(TA::datum_type().size_of(), k).wrap(&packed_a.view()),
                b: op.b_packed(TB::datum_type().size_of(), k).wrap(&packed_b.view()),
                k,
            }],
            |r, c| {
                let mut v: TI = TI::zero();
                for i in 0..k {
                    let a: TI = a.as_slice::<TA>().unwrap()[i + k * r].as_();
                    let b: TI = b.as_slice::<TB>().unwrap()[c + i * n].as_();
                    v += a * b;
                }
                v.as_()
            },
        )
    }
}

pub fn test_mat_vec_mul_prep<K: MatMatMulKer<TI> + 'static, TA, TB, TC, TI>(
    m: usize,
    k: usize,
    a: &Tensor,
    b: &Tensor,
) -> Result<(), proptest::test_runner::TestCaseError>
where
    TA: LADatum + AsPrimitive<TI> + 'static,
    TB: LADatum + AsPrimitive<TI> + 'static,
    TC: LADatum + AsPrimitive<TI> + 'static,
    TI: LADatum + AsPrimitive<TC> + 'static + Neg<Output = TI>,
    i32: AsPrimitive<TI>,
    usize: AsPrimitive<TI>,
{
    crate::setup_test_logger();
    unsafe {
        let op = MatMatMulImpl::<K, TI>::default();
        let mut packed_a =
            Tensor::uninitialized_aligned::<TA>(&[op.a_pack().len(k, m)], op.a_pack().alignment())
                .unwrap();
        let mut packed_b =
            Tensor::uninitialized_aligned::<TB>(&[op.b_pack().len(k, 1)], op.b_pack().alignment())
                .unwrap();
        op.a_pack().pack(&mut packed_a.view_mut(), &a.view(), 1, 0);
        let b = b.clone().into_shape(&[k, 1]).unwrap();
        op.b_pack().pack(&mut packed_b.view_mut(), &b.view(), 0, 1);

        let pa = op.a_packed(TA::datum_type().size_of(), k).wrap(&packed_a.view());
        let pb = op.b_packed(b.datum_type().size_of(), k).wrap(&packed_b.view());

        fused_ops::<K, TA, TB, TC, TI, _>(
            m,
            1,
            &[FusedSpec::AddMatMul { k, a: pa, b: pb }],
            |r, _| {
                let mut inter = TI::zero();
                for i in 0..k {
                    let a: TI = a.as_slice::<TA>().unwrap()[i + k * r].as_();
                    let b: TI = b.as_slice::<TB>().unwrap()[i].as_();
                    inter += a * b;
                }
                inter.as_()
            },
        )
    }
}

pub unsafe fn fused_ops<K: MatMatMulKer<TI> + 'static, TA, TB, TC, TI, F: Fn(usize, usize) -> TC>(
    m: usize,
    n: usize,
    spec: &[FusedSpec],
    expect: F,
) -> proptest::test_runner::TestCaseResult
where
    TA: LADatum + AsPrimitive<TI> + 'static,
    TB: LADatum + AsPrimitive<TI> + 'static,
    TC: LADatum + AsPrimitive<TI> + 'static,
    TI: LADatum + AsPrimitive<TC> + 'static,
    i32: AsPrimitive<TI>,
    usize: AsPrimitive<TI>,
{
    crate::setup_test_logger();
    let op = MatMatMulImpl::<K, TI>::default();

    let mut found = Tensor::zero::<TC>(&[m, n]).unwrap();
    let c_store = op
        .c_from_data_and_strides(TC::datum_type().size_of(), n as isize, 1)
        .wrap(&found.view_mut());
    let mut spec: TVec<FusedSpec> = spec.into();
    spec.push(FusedSpec::Store(c_store));

    op.run(m, n, &spec).unwrap();
    let expected =
        tract_ndarray::prelude::Array2::from_shape_fn((m, n), |(r, c)| expect(r, c)).into_tensor();
    if found.close_enough(&expected, true).is_err() {
        println!("found, expected:");
        for r in 0..m {
            for c in 0..n {
                let f = found.as_slice_unchecked::<TC>()[r * n + c];
                let e = expected.as_slice_unchecked::<TC>()[r * n + c];
                let mut s = format!("{:4} ", f);
                if f != e {
                    s = nu_ansi_term::Color::Red.paint(s).to_string();
                }
                print!("{:4} ", s);
            }
            print!("      ");
            for c in 0..n {
                print!("{:4} ", expected.as_slice_unchecked::<TC>()[r * n + c]);
            }
            println!();
        }
    }
    found.close_enough(&expected, true).map_err(|e| TestCaseError::Fail(e.to_string().into()))
}

pub unsafe fn row_add<K: MatMatMulKer<TI> + 'static, TA, TB, TC, TI>(
    m: usize,
    n: usize,
) -> proptest::test_runner::TestCaseResult
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
        m,
        n,
        &[FusedSpec::BinPerRow(tensor1(&bias).view(), BinOp::Add)],
        |r, _| bias[r].as_(),
    )
}

pub unsafe fn row_mul<K: MatMatMulKer<TI> + 'static, TA, TB, TC, TI>(
    m: usize,
    n: usize,
) -> proptest::test_runner::TestCaseResult
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
        m,
        n,
        &[
            FusedSpec::BinScalar(&tensor0(1i32.as_()), BinOp::Add),
            FusedSpec::BinPerRow(tensor1(&bias).view(), BinOp::Mul),
        ],
        |r, _| bias[r].as_(),
    )
}

pub unsafe fn col_add<K: MatMatMulKer<TI> + 'static, TA, TB, TC, TI>(
    m: usize,
    n: usize,
) -> proptest::test_runner::TestCaseResult
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
        m,
        n,
        &[FusedSpec::BinPerCol(tensor1(&bias).view(), BinOp::Add)],
        |_, c| bias[c].as_(),
    )
}

pub unsafe fn col_mul<K: MatMatMulKer<TI> + 'static, TA, TB, TC, TI>(
    m: usize,
    n: usize,
) -> proptest::test_runner::TestCaseResult
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
        m,
        n,
        &[
            FusedSpec::BinScalar(&tensor0(1i32.as_()), BinOp::Add),
            FusedSpec::BinPerCol(tensor1(&bias).view(), BinOp::Mul),
        ],
        |_, c| bias[c].as_(),
    )
}

pub unsafe fn add_d<K: MatMatMulKer<TI> + 'static, TA, TB, TC, TI>(
    m: usize,
    n: usize,
) -> proptest::test_runner::TestCaseResult
where
    TA: LADatum + AsPrimitive<TI> + 'static,
    TB: LADatum + AsPrimitive<TI> + 'static,
    TC: LADatum + AsPrimitive<TI> + 'static,
    TI: LADatum + AsPrimitive<TC> + 'static + Neg<Output = TI>,
    i32: AsPrimitive<TI>,
    usize: AsPrimitive<TI>,
{
    let d = (0..m * n).map(|i| i.as_()).collect::<Vec<TI>>();
    let d = tensor1(&d).into_shape(&[m, n]).unwrap();
    let store_spec = OutputStoreSpec::View { m_axis: 0, n_axis: 1, mr: K::mr(), nr: K::nr() };
    fused_ops::<K, TA, TB, TC, TI, _>(
        m,
        n,
        &[FusedSpec::AddUnicast(store_spec.wrap(&d.view()))],
        |r, c| d.to_array_view_unchecked::<TI>().into_dimensionality().unwrap()[(r, c)].as_(),
    )
}

pub unsafe fn max<K: MatMatMulKer<TI>, TA, TB, TC, TI>(
    m: usize,
    n: usize,
) -> proptest::test_runner::TestCaseResult
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
        m,
        n,
        &[FusedSpec::BinScalar(&tensor0(five), BinOp::Max)],
        |_, _| five.as_(),
    )
}

pub unsafe fn min<K: MatMatMulKer<TI>, TA, TB, TC, TI>(
    m: usize,
    n: usize,
) -> proptest::test_runner::TestCaseResult
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
        m,
        n,
        &[FusedSpec::BinScalar(&tensor0(five), BinOp::Min)],
        |_, _| TC::zero(),
    )
}
