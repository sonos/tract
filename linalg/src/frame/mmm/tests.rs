use super::*;
use crate::LADatum;
use num_traits::AsPrimitive;
use proptest::prelude::*;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::Neg;
use tract_data::internal::*;

#[macro_export]
macro_rules! mmm_frame_tests {
    ($cond:expr, $ker:ty, $ta:ty, $tb:ty, $tc:ty, $ti:ty) => {
        mod frame {
            #[allow(unused_imports)]
            use $crate::frame::mmm::tests::*;
            use $crate::num_traits::{One, Zero};
            use tract_data::internal::*;

            proptest::proptest! {
                #[test]
                fn mat_mul_prepacked_prop((m, k, n, ref a, ref b) in strat_mat_mat_mul::<$ta, $tb>()) {
                    if $cond {
                        test_mat_mat_mul_prep::<$ker, $ta, $tb, $tc, $ti>(m, k, n, &a, &b)?
                    }
                }

                #[test]
                fn mat_mul_prepacked_late((m, k, n, ref a, ref b) in strat_mat_mat_mul::<$ta, $tb>()) {
                    if $cond {
                        test_mat_mat_mul_late::<$ker, $ta, $tb, $tc, $ti>(m, k, n, &a, &b)?
                    }
                }

                #[test]
                fn mat_vec_prepacked_prop((m, k, ref a, ref b) in strat_mat_vec_mul::<$ta, $tb>()) {
                    if $cond {
                        test_mat_vec_mul_prep::<$ker, $ta, $tb, $tc, $ti>(m, k, &*a, b)?
                    }
                }

                #[test]
                fn conv_prepacked_prop(pb in strat_conv_1d::<$ta, $tb>()) {
                    if $cond {
                        let found = pb.run::<$ker, $tc, $ti>();
                        let expected = pb.expected::<$tc, $ti>();
                        found.close_enough(&expected, true).unwrap()
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
            fn conv_prepacked_1() {
                if $cond {
                    let pb = ConvProblem::<$ta, $tb> {
                        ci: 1,
                        co: 1,
                        kt: 1,
                        stride: 1,
                        dilation: 1,
                        filters: tensor2(&[[1i32]]).cast_to::<$ta>().unwrap().into_owned(),
                        data: tensor2(&[[<$tb>::zero(), <$tb>::one()]]),
                        phantom: std::marker::PhantomData,
                    };
                    let expected = pb.expected::<$tc, $ti>();
                    pb.run::<$ker, $tc, $ti>().close_enough(&expected, true).unwrap()
                }
            }

            #[test]
            fn conv_prepacked_2() {
                if $cond {
                    let mut filters = Tensor::zero::<$ta>(&[14, 3*2]).unwrap();
                    filters.as_slice_mut().unwrap()[13 * 6 + 5] = <$ta>::one();
                    let mut data = vec![<$tb>::zero(); 3 * 10];
                    data[8 + 2 * 10] = <$tb>::one(); // last used input
                    let pb = ConvProblem::<$ta, $tb> {
                        ci: 3,
                        co: 14,
                        kt: 2,
                        stride: 3,
                        dilation: 2,
                        filters,
                        data: tensor1(&data).into_shape(&[6, 5]).unwrap(),
                        phantom: std::marker::PhantomData,
                    };
                    let expected = pb.expected::<$tc, $ti>();
                    pb.run::<$ker, $tc, $ti>().close_enough(&expected, true).unwrap();
                }
            }

            #[test]
            fn conv_3() {
                if $cond {
                    let pb = ConvProblem::<$ta, $tb> {
                        ci: 1,
                        co: 1,
                        kt: 1,
                        stride: 1,
                        dilation: 1,
                        filters: tensor2(&[[2 as $ta]]),
                        data: tensor2(&[[-65i32 as $tb]]),
                        phantom: std::marker::PhantomData,
                    };
                    let expected = pb.expected::<$tc, $ti>();
                    pb.run::<$ker, $tc, $ti>().close_enough(&expected, true).unwrap();
                }
            }

            #[test]
            fn conv_4() {
                if $cond {
                    let pb = ConvProblem::<$ta, $tb> {
                        ci: 1,
                        co: 5,
                        kt: 1,
                        stride: 1,
                        dilation: 1,
                        filters: tensor2(&[[0 as $ta], [0 as $ta], [0 as $ta], [0 as $ta], [5 as $ta]]),
                        data: tensor2(&[[1 as $tb]]),
                        phantom: std::marker::PhantomData,
                    };
                    let expected = pb.expected::<$tc, $ti>();
                    pb.run::<$ker, $tc, $ti>().close_enough(&expected, true).unwrap();
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

pub fn strat_mat_mat_mul<TA: LADatum, TB: LADatum>(
) -> BoxedStrategy<(usize, usize, usize, Tensor, Tensor)> {
    (1usize..5, 1usize..5, 1usize..5)
        .prop_flat_map(move |(m, k, n)| {
            (
                Just(m),
                Just(k),
                Just(n),
                proptest::collection::vec(TA::strat(), m * k),
                proptest::collection::vec(TB::strat(), n * k),
            )
        })
        .prop_map(move |(m, k, n, a, b)| {
            (
                m,
                k,
                n,
                tensor1(&a).into_shape(&[m, k]).unwrap(),
                tensor1(&b).into_shape(&[k, n]).unwrap(),
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
                proptest::collection::vec(TA::strat(), m * k),
                proptest::collection::vec(TB::strat(), k),
            )
        })
        .prop_map(move |(m, k, a, b)| (m, k, tensor1(&a).into_shape(&[m, k]).unwrap(), tensor1(&b)))
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
    assert_eq!(a.datum_type(), TA::datum_type());
    let op = MatMatMulImpl::<K, TI>::new();
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
                b: op.b_packed(TB::datum_type().size_of(), k).wrap(&packed_b.view()).unwrap(),
                k,
            }],
            |r, c| {
                let mut v: TI = TI::zero();
                for i in 0..k {
                    let a: TI = a.as_slice::<TA>().unwrap()[i + k * r].as_();
                    let b: TI = b.as_slice::<TB>().unwrap()[c + i * n].as_();
                    v = v + a * b;
                }
                v.as_()
            },
        )
    }
}

pub fn test_mat_mat_mul_late<K: MatMatMulKer<TI> + 'static, TA, TB, TC, TI>(
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
    TI: LADatum + AsPrimitive<TC> + 'static + Neg<Output = TI>,
    i32: AsPrimitive<TI>,
    usize: AsPrimitive<TI>,
{
    assert_eq!(a.datum_type(), TA::datum_type());
    let op = MatMatMulImpl::<K, TI>::new();
    unsafe {
        let mut packed_a =
            Tensor::uninitialized_aligned::<TA>(&[op.a_pack().len(k, m)], op.a_pack().alignment())
                .unwrap();
        op.a_pack().pack(packed_a.view_mut(), a.view(), 1, 0);

        fused_ops::<K, TA, TB, TC, TI, _>(
            m,
            n,
            &[FusedSpec::AddMatMul {
                a: op.a_packed(TA::datum_type().size_of(), k).wrap(&packed_a.view()),
                b: InputStore::LatePacking {
                    packer: op.b_pack(),
                    ptr: b.as_bytes().as_ptr(),
                    dt: TB::datum_type(),
                    k,
                    mn: n,
                    k_stride: n as isize,
                    mn_stride: 1,
                },
                k,
            }],
            |r, c| {
                let mut v: TI = TI::zero();
                for i in 0..k {
                    let a: TI = a.as_slice::<TA>().unwrap()[i + k * r].as_();
                    let b: TI = b.as_slice::<TB>().unwrap()[c + i * n].as_();
                    v = v + a * b;
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
    unsafe {
        let op = MatMatMulImpl::<K, TI>::new();
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
        let pb = op.b_packed(b.datum_type().size_of(), k).wrap(&packed_b.view()).unwrap();

        fused_ops::<K, TA, TB, TC, TI, _>(
            m,
            1,
            &[FusedSpec::AddMatMul { k, a: pa, b: pb }],
            |r, _| {
                let mut inter = TI::zero();
                for i in 0..k {
                    let a: TI = a.as_slice::<TA>().unwrap()[i + k * r].as_();
                    let b: TI = b.as_slice::<TB>().unwrap()[i].as_();
                    inter = inter + a * b;
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
    TI: LADatum + AsPrimitive<TC> + 'static + Neg<Output = TI>,
    i32: AsPrimitive<TI>,
    usize: AsPrimitive<TI>,
{
    let op = MatMatMulImpl::<K, TI>::new();

    let mut found = Tensor::zero::<TC>(&[m, n]).unwrap();
    let c_store = op
        .c_from_data_and_strides(TC::datum_type().size_of(), n as isize, 1)
        .wrap(&mut found.view_mut());
    let mut spec: TVec<FusedSpec> = spec.into();
    spec.push(FusedSpec::Store(c_store));

    op.run(m, n, &spec).unwrap();
    let expected =
        tract_ndarray::prelude::Array2::from_shape_fn((m, n), |(r, c)| expect(r, c)).into_tensor();
    if found.close_enough(&expected, true).is_err() {
        println!("found, expected:");
        for r in 0..m {
            for c in 0..n {
                print!("{:4} ", found.as_slice_unchecked::<TC>()[r * n + c]);
            }
            print!("      ");
            for c in 0..n {
                print!("{:4} ", expected.as_slice_unchecked::<TC>()[r * n + c]);
            }
            println!();
        }
    }
    assert!(found == expected);
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
        &[FusedSpec::BinPerRow(&tensor1(&*bias), BinOp::Add)],
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
            FusedSpec::BinPerRow(&tensor1(&*bias), BinOp::Mul),
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
        &[FusedSpec::BinPerCol(&tensor1(&*bias), BinOp::Add)],
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
            FusedSpec::BinPerCol(&tensor1(&*bias), BinOp::Mul),
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
    let d = tensor1(&*d).into_shape(&[m, n]).unwrap();
    let store_spec = OutputStoreSpec::View { axes: None, mr: K::mr(), nr: K::nr() };
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

#[derive(Clone, Debug)]
pub struct ConvProblem<TA: LADatum, TB: LADatum> {
    pub ci: usize,
    pub co: usize,
    pub kt: usize,
    pub stride: usize,
    pub dilation: usize,
    pub filters: Tensor,
    pub data: Tensor,
    pub phantom: std::marker::PhantomData<(TA, TB)>,
}

impl<TA: LADatum, TB: LADatum> ConvProblem<TA, TB> {
    pub fn kernel_field(&self) -> usize {
        self.dilation * (self.kt - 1) + 1
    }
    // this is not n, but the T in NTC of input to direct convolution
    pub fn input_width(&self) -> usize {
        assert!(self.data.len() % self.ci == 0);
        self.data.len() / self.ci
    }
    pub fn output_width(&self) -> usize {
        (self.input_width() - self.kernel_field()) / self.stride + 1
    }
    pub fn m(&self) -> usize {
        self.co
    }
    pub fn k(&self) -> usize {
        self.ci * self.kt
    }
    pub fn n(&self) -> usize {
        self.output_width()
    }
    pub fn data_cols_offsets(&self) -> Vec<isize> {
        (0..self.output_width()).map(|i| (i * self.stride) as isize).collect()
    }
    pub fn data_rows_offsets(&self) -> Vec<isize> {
        (0..self.ci)
            .flat_map(move |ici| {
                (0..self.kt)
                    .map(move |ikt| (ikt * self.dilation + ici * self.input_width()) as isize)
            })
            .collect()
    }

    pub fn expected<TC, TI>(&self) -> Tensor
    where
        TA: LADatum + AsPrimitive<TI>,
        TB: LADatum + AsPrimitive<TI>,
        TC: LADatum,
        TI: LADatum + AsPrimitive<TC>,
    {
        let mut expect = Tensor::zero::<TI>(&[self.co, self.output_width()]).unwrap();
        for x in 0..self.output_width() {
            for ico in 0..self.co {
                for ikt in 0..self.kt {
                    for ici in 0..self.ci {
                        let f = self.filters.as_slice::<TA>().unwrap()
                            [ici * self.kt + ikt + self.ci * self.kt * ico];
                        let d = self.data.as_slice::<TB>().unwrap()
                            [x * self.stride + ikt * self.dilation + ici * self.input_width()];
                        let ref mut pv =
                            expect.as_slice_mut::<TI>().unwrap()[x + ico * self.output_width()];
                        *pv += f.as_() * d.as_();
                    }
                }
            }
        }
        expect.cast_to::<TC>().unwrap().into_owned()
    }

    pub fn run<K: MatMatMulKer<TI>, TC, TI>(&self) -> Tensor
    where
        TA: LADatum + AsPrimitive<TI> + 'static,
        TB: LADatum + AsPrimitive<TI> + 'static,
        TC: LADatum + AsPrimitive<TI> + 'static,
        TI: LADatum + AsPrimitive<TC> + 'static + Neg<Output = TI>,
        i32: AsPrimitive<TI>,
        usize: AsPrimitive<TI>,
    {
        unsafe {
            let op = MatMatMulImpl::<K, TI>::new();
            let mut packed_a = Tensor::uninitialized_aligned::<TA>(
                &[op.a_pack().len(self.k(), self.m())],
                op.a_pack().alignment(),
            )
            .unwrap();
            op.a_pack().pack(packed_a.view_mut(), self.filters.view(), 1, 0);

            let mut found = tensor0(TC::max_value())
                .broadcast_scalar_to_shape(&[self.co, self.output_width()])
                .unwrap();
            op.run(
                self.m(),
                self.n(),
                &[
                    FusedSpec::AddMatMul {
                        a: op.a_packed(TA::datum_type().size_of(), self.k()).wrap(&packed_a.view()),
                        b: op
                            .b_from_data_and_offsets(
                                TB::datum_type().size_of(),
                                &self.data_rows_offsets(),
                                &self.data_cols_offsets(),
                            )
                            .wrap(&self.data.view())
                            .unwrap(),
                        k: self.k(),
                    },
                    FusedSpec::Store(
                        op.c_from_data_and_strides(
                            TC::datum_type().size_of(),
                            self.n() as isize,
                            1,
                        )
                        .wrap(&found.view_mut()),
                    ),
                ],
            )
            .unwrap();
            found
        }
    }
}

pub fn strat_conv_1d<TA: LADatum, TB: LADatum>() -> BoxedStrategy<ConvProblem<TA, TB>>
where
    isize: AsPrimitive<TA> + AsPrimitive<TB>,
{
    (1usize..40, 1usize..40, 1usize..10, 1usize..5, 1usize..5)
        .prop_flat_map(|(ci, co, kt, stride, dilation)| {
            let min = ((kt - 1) * dilation + 1) * stride;
            (Just(ci), Just(co), Just(kt), Just(stride), Just(dilation), min..min + 10)
        })
        .prop_flat_map(move |(ci, co, kt, stride, dilation, t)| {
            (
                Just(ci),
                Just(co),
                Just(kt),
                Just(t),
                Just(stride),
                Just(dilation),
                proptest::collection::vec(TA::strat(), ci * co * kt),
                proptest::collection::vec(TB::strat(), t * ci),
            )
        })
        .prop_map(move |(ci, co, kt, t, stride, dilation, filters, data)| ConvProblem {
            ci,
            co,
            kt,
            stride,
            dilation,
            filters: tensor1(&filters).into_shape(&[co, ci * kt]).unwrap(),
            data: tensor1(&data).into_shape(&[ci, t]).unwrap(),
            phantom: PhantomData,
        })
        .boxed()
}
