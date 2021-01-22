use super::*;
use crate::test::*;
use num_traits::AsPrimitive;
use proptest::prelude::*;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::Neg;
use tract_data::prelude::*;

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
            fn row_mul_2_1_3() {
                if $cond {
                    unsafe { row_mul::<$ker, $ta, $tb, $tc, $ti>(2, 1, 3).unwrap() }
                }
            }

            #[test]
            fn row_add_2_1_3() {
                if $cond {
                    unsafe { row_add::<$ker, $ta, $tb, $tc, $ti>(2, 1, 3).unwrap() }
                }
            }

            #[test]
            fn col_mul_2_1_3() {
                if $cond {
                    unsafe { col_mul::<$ker, $ta, $tb, $tc, $ti>(2, 1, 3).unwrap() }
                }
            }

            #[test]
            fn col_add_2_1_3() {
                if $cond {
                    unsafe { col_add::<$ker, $ta, $tb, $tc, $ti>(2, 1, 3).unwrap() }
                }
            }

            #[test]
            fn max_2_1_3() {
                if $cond {
                    unsafe { max::<$ker, $ta, $tb, $tc, $ti>(2, 1, 3).unwrap() }
                }
            }

            #[test]
            fn min_2_1_3() {
                if $cond {
                    unsafe { min::<$ker, $ta, $tb, $tc, $ti>(2, 3, 3).unwrap() }
                }
            }
        }
    };
}

#[macro_export]
macro_rules! mmm_s_frame_tests {
    ($cond:expr, $ker:ty, $ta: ty, $tb: ty, $tc: ty, $ti: ty) => {
        mod frame_s {
            /*
            #[allow(unused_imports)]
            use num_traits::*;
            use std::ops::Neg;
            use $crate::frame::mmm::tests::ConvProblem;
            use $crate::*;
            */
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
    TI: LADatum + AsPrimitive<TC> + 'static + Neg<Output = TI>,
    i32: AsPrimitive<TI>,
    usize: AsPrimitive<TI>,
{
    assert_eq!(a.datum_type(), TA::datum_type());
    let op = MatMatMulImpl::<K, TA, TB, TC, TI>::new(m, k, n);
    unsafe {
        let mut packed_a =
            Tensor::uninitialized_aligned::<TA>(&[op.a_pack().len(m)], op.a_pack().alignment())
                .unwrap();
        op.a_pack().pack(packed_a.view_mut(), a.view(), 1, 0);

        let mut packed_b =
            Tensor::uninitialized_aligned::<TB>(&[op.b_pack().len(n)], op.b_pack().alignment())
                .unwrap();
        op.b_pack().pack(packed_b.view_mut(), b.view(), 0, 1);

        let mut found = tensor0(TC::max_value()).broadcast_scalar_to_shape(&[m, n]).unwrap();

        op.run(
            &op.a_packed().wrap(&packed_a.view()),
            &op.b_packed().wrap(&packed_b.view()),
            &mut op.c_from_data_and_strides(n as isize, 1).wrap(&found.view_mut()),
            &[],
        )
        .unwrap();

        let mut expected = Tensor::zero::<TC>(&[m, n]).unwrap();
        for x in 0..n {
            for y in 0..m {
                let mut v: TI = TI::zero();
                for i in 0..k {
                    let a: TI = a.as_slice::<TA>().unwrap()[i + k * y].as_();
                    let b: TI = b.as_slice::<TB>().unwrap()[x + i * n].as_();
                    v = v + a * b;
                }
                expected.as_slice_mut::<TC>().unwrap()[x + y * n] = v.as_();
            }
        }
        found.close_enough(&expected, true).unwrap();
        Ok(())
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
        let op = MatMatMulImpl::<K, TA, TB, TC, TI>::new(m, k, 1);
        op.b_vec_from_data_and_stride(1);
        op.c_vec_from_data_and_stride(1);
        let mut packed_a =
            Tensor::uninitialized_aligned::<TA>(&[op.a_pack().len(m)], op.a_pack().alignment())
                .unwrap();
        op.a_pack().pack(&mut packed_a.view_mut(), &a.view(), 1, 0);

        let mut found = Tensor::uninitialized::<TC>(&[m]).unwrap();

        op.run(
            &op.a_packed().wrap(&packed_a.view()),
            &op.b_vec_from_data().wrap(&b.view()),
            &mut op.c_vec_from_data().wrap(&found.view_mut()),
            &[],
        )
        .unwrap();

        let mut expected = Tensor::zero::<TC>(&[m]).unwrap();
        for y in 0..m {
            let mut inter = TI::zero();
            for i in 0..k {
                let a: TI = a.as_slice::<TA>().unwrap()[i + k * y].as_();
                let b: TI = b.as_slice::<TB>().unwrap()[i].as_();
                inter = inter + a * b;
            }
            expected.as_slice_mut().unwrap()[y] = inter.as_();
        }

        found.close_enough(&expected, true).unwrap();
        Ok(())
    }
}

pub unsafe fn fused_op<K: MatMatMulKer<TI> + 'static, TA, TB, TC, TI, F: Fn(&mut [TI])>(
    m: usize,
    k: usize,
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
    let a = tensor1(&*vec![TA::one(); m * k]).into_shape(&[m, k]).unwrap();
    let b = tensor1(&*vec![TB::one(); k * n]).into_shape(&[k, n]).unwrap();
    let op = MatMatMulImpl::<K, TA, TB, TC, TI>::new(m, k, n);

    let mut packed_a =
        Tensor::uninitialized_aligned::<TA>(&[op.a_pack().len(m)], op.a_pack().alignment())
            .unwrap();
    op.a_pack().pack(packed_a.view_mut(), a.view(), 1, 0);

    let mut packed_b =
        Tensor::uninitialized_aligned::<TB>(&[op.b_pack().len(n)], op.b_pack().alignment())
            .unwrap();
    op.b_pack().pack(packed_b.view_mut(), b.view(), 0, 1);

    let mut found = Tensor::zero::<TC>(&[m, n]).unwrap();

    op.run(
        &op.a_packed().wrap(&packed_a.view()),
        &op.b_packed().wrap(&packed_b.view()),
        &mut op.c_from_data_and_strides(n as isize, 1).wrap(&found.view_mut()),
        spec,
    )
    .unwrap();

    let mut inter = Tensor::zero::<TI>(&[m, n]).unwrap();
    for x in 0..n {
        for y in 0..m {
            let mut s = TI::zero();
            for i in 0..k {
                s += a.as_slice::<TA>().unwrap()[i + k * y].as_()
                    * b.as_slice::<TB>().unwrap()[x + i * n].as_()
            }
            inter.as_slice_mut::<TI>().unwrap()[x + y * n] = s;
        }
    }
    expect(inter.as_slice_mut::<TI>().unwrap());
    let expected = inter.cast_to::<TC>().unwrap().into_owned();

    found.close_enough(&expected, true).unwrap();
    Ok(())
}

pub unsafe fn row_add<K: MatMatMulKer<TI> + 'static, TA, TB, TC, TI>(
    m: usize,
    k: usize,
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
    fused_op::<K, TA, TB, TC, TI, _>(m, k, n, &[FusedSpec::PerRowAdd(tensor1(&*bias))], |exp| {
        for x in 0..n {
            for y in 0..m {
                exp[x + y * n] += bias[y]
            }
        }
    })
}

pub unsafe fn row_mul<K: MatMatMulKer<TI> + 'static, TA, TB, TC, TI>(
    m: usize,
    k: usize,
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
    fused_op::<K, TA, TB, TC, TI, _>(m, k, n, &[FusedSpec::PerRowMul(tensor1(&*bias))], |exp| {
        for x in 0..n {
            for y in 0..m {
                exp[x + y * n] *= bias[y]
            }
        }
    })
}

pub unsafe fn col_add<K: MatMatMulKer<TI> + 'static, TA, TB, TC, TI>(
    m: usize,
    k: usize,
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
    fused_op::<K, TA, TB, TC, TI, _>(m, k, n, &[FusedSpec::PerColAdd(tensor1(&*bias))], |exp| {
        for x in 0..n {
            for y in 0..m {
                exp[x + y * n] += bias[x]
            }
        }
    })
}

pub unsafe fn col_mul<K: MatMatMulKer<TI> + 'static, TA, TB, TC, TI>(
    m: usize,
    k: usize,
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
    fused_op::<K, TA, TB, TC, TI, _>(m, k, n, &[FusedSpec::PerColMul(tensor1(&*bias))], |exp| {
        for x in 0..n {
            for y in 0..m {
                exp[x + y * n] *= bias[x]
            }
        }
    })
}

pub unsafe fn max<K: MatMatMulKer<TI>, TA, TB, TC, TI>(
    m: usize,
    k: usize,
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
    fused_op::<K, TA, TB, TC, TI, _>(m, k, n, &[FusedSpec::Max(tensor0(five))], |exp| {
        exp.iter_mut().for_each(|x| *x = if *x < five { five } else { *x })
    })
}

pub unsafe fn min<K: MatMatMulKer<TI>, TA, TB, TC, TI>(
    m: usize,
    k: usize,
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
    fused_op::<K, TA, TB, TC, TI, _>(m, k, n, &[FusedSpec::Min(tensor0(five))], |exp| {
        exp.iter_mut().for_each(|x| *x = if *x > five { five } else { *x })
    })
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
            let op = MatMatMulImpl::<K, TA, TB, TC, TI>::new(self.m(), self.k(), self.n());
            let mut packed_a = Tensor::uninitialized_aligned::<TA>(
                &[op.a_pack().len(self.m())],
                op.a_pack().alignment(),
            )
            .unwrap();
            op.a_pack().pack(packed_a.view_mut(), self.filters.view(), 1, 0);

            let mut found = tensor0(TC::max_value())
                .broadcast_scalar_to_shape(&[self.co, self.output_width()])
                .unwrap();
            op.run(
                &op.a_packed().wrap(&packed_a.view()),
                &op.b_from_data_and_offsets(&self.data_rows_offsets(), &self.data_cols_offsets())
                    .wrap(&self.data.view()),
                &mut op.c_from_data_and_strides(self.n() as isize, 1).wrap(&found.view_mut()),
                &[],
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

/*
#[derive(Debug)]
pub struct QMatMulProblem<TA, TB, TC, TI> {
pub m: usize,
pub k: usize,
pub n: usize,
pub a: Tensor,
pub a0: Tensor,
pub b: Tensor,
pub b0: Tensor,
pub boo: PhantomData<(TA, TB, TC, TI)>,
}

fn arbitrary_zero_point_with<TI: Arbitrary + Datum>(n: usize) -> BoxedStrategy<Tensor> {
prop_oneof![any::<TI>().prop_map(tensor0), vec(any::<TI>(), n..=n).prop_map(|v| tensor1(&*v)),]
.boxed()
}

impl<TA, TB, TC, TI> Arbitrary for QMatMulProblem<TA, TB, TC, TI>
where
TA: Datum + Arbitrary + 'static,
TB: Datum + Arbitrary + 'static,
TC: Arbitrary + 'static + Debug + 'static,
TI: Arbitrary + 'static + Debug + 'static,
{
type Parameters = ();
type Strategy = BoxedStrategy<Self>;

fn arbitrary_with(_: Self::Parameters) -> Self::Strategy {
(1usize..10, 1usize..10, 1usize..10)
.prop_flat_map(|(m, k, n)| {
(
Just(m),
Just(k),
Just(n),
vec(any::<TA>(), m * k..=m * k),
arbitrary_zero_point_with::<TA>(m),
vec(any::<TB>(), k * n..=k * n),
arbitrary_zero_point_with::<TB>(n),
)
})
.prop_map(|(m, k, n, a, a0, b, b0)| QMatMulProblem {
m,
k,
n,
a: tensor1(&a).into_shape(&[m, k]).unwrap(),
a0,
b: tensor1(&b).into_shape(&[k, n]).unwrap(),
b0,
boo: PhantomData,
})
.boxed()
}
}

impl<TA, TB, TC, TI> QMatMulProblem<TA, TB, TC, TI>
where
TA: Arbitrary + 'static + Datum + AsPrimitive<TI> + Zero + Copy + crate::test::LADatum,
TB: Arbitrary + 'static + Datum + AsPrimitive<TI> + Zero + Copy + crate::test::LADatum,
TC: Arbitrary + 'static + Datum + Copy + Bounded + AsPrimitive<TI> + Zero + 'static,
TI: Arbitrary
+ 'static
+ Datum
+ Copy
+ AsPrimitive<TC>
+ Add<Output = TI>
+ Mul<Output = TI>
+ Sub<Output = TI>
+ AddAssign
+ Neg<Output = TI>
+ Zero
+ Ord,
    usize: AsPrimitive<TI>,
    i32: AsPrimitive<TI>,
{
    pub fn reference(&self) -> Tensor {
        let mut i = Tensor::zero::<TI>(&[self.m, self.n]).unwrap();
        for m in 0..self.m {
            for n in 0..self.n {
                for k in 0..self.k {
                    let a: TI = self.a.as_slice::<TA>().unwrap()[k + self.k * m].as_();
                    let b: TI = self.b.as_slice::<TB>().unwrap()[n + self.n * k].as_();
                    let a0: TI = if self.a0.rank() == 0 {
                        self.a0.to_scalar::<TA>().unwrap().as_()
                    } else {
                        self.a0.as_slice::<TA>().unwrap()[m].as_()
                    };
                    let b0: TI = if self.b0.rank() == 0 {
                        self.b0.to_scalar::<TB>().unwrap().as_()
                    } else {
                        self.b0.as_slice::<TB>().unwrap()[n].as_()
                    };
                    i.as_slice_mut::<TI>().unwrap()[n + self.n * m] += (a - a0) * (b - b0);
                }
            }
        }
        i.as_slice_mut::<TI>()
            .unwrap()
            .iter_mut()
            .for_each(|i| *i = (*i).max(TC::min_value().as_()).min(TC::max_value().as_()));
        i.cast_to::<TC>().unwrap().into_owned()
    }

    pub fn run<K: MatMatMulKer<TI>>(&self) -> Tensor {
        unsafe {
            let mut c = Tensor::zero::<TC>(&[self.m, self.n]).unwrap();
            let mut mmm = MatMatMulImpl::from(MatMatMulImpl::<K, TA, TB, TC, TI>::new(
                    self.m, self.k, self.n,
                    ));
            let mut packed_a = Tensor::uninitialized_aligned::<TA>(
                &[mmm.a_pack().len(self.m)],
                mmm.a_pack().alignment(),
                )
                .unwrap();
            mmm.a_pack().pack(packed_a.view_mut(), self.a.view(), 1, 0);

            let mut packed_b = Tensor::uninitialized_aligned::<TB>(
                &[mmm.b_pack().len(self.n)],
                mmm.b_pack().alignment(),
                )
                .unwrap();
            mmm.b_pack().pack(packed_b.view_mut(), self.b.view(), 0, 1);

            mmm.set_zero_point_a(self.a0.clone());
            mmm.set_zero_point_b(self.b0.clone());

            mmm.run(&packed_a.view(), &packed_b.view(), &mut c.view_mut(), &[]).unwrap();
            c
        }
    }
}
*/

#[macro_export]
macro_rules! qmmm_frame_tests {
    ($cond:expr, $ker:ty, $ta: ty, $tb: ty, $tc: ty, $ti: ty) => {
        /*
               mod qframe {
               use num_traits::{One, Zero};
               use proptest::prelude::*;
               use std::marker::PhantomData;
               use tract_data::prelude::*;
               #[allow(unused_imports)]
               use $crate::frame::mmm::tests::*;

               type QProblem = QMatMulProblem<$ta, $tb, $tc, $ti>;

               proptest::proptest! {
               #[test]
               fn q_mat_mul_prop(pb in any::<QMatMulProblem<$ta, $tb, $tc, $ti>>()) {
               if $cond {
               prop_assert_eq!(pb.run::<$ker>(), pb.reference())
               }
               }
               }

               #[test]
               fn q_mat_mul_1() {
               if $cond {
               let pb = QProblem {
               m: 1,
               k: 1,
               n: 1,
               a0: tensor1(&[<$ta>::one()]),
               a: tensor2(&[[<$ta>::zero()]]),
               b0: tensor1(&[<$tb>::one()]),
               b: tensor2(&[[<$tb>::one()]]),
               boo: PhantomData,
               };
               assert_eq!(pb.run::<$ker>(), pb.reference());
               }
               }

               #[test]
               fn q_mat_mul_sat_1() {
               if $cond {
               let pb = QProblem {
               m: 1,
               k: 1,
               n: 1,
               a0: tensor1(&[0]).cast_to::<$ta>().unwrap().into_owned(),
               a: tensor2(&[[3]]).cast_to::<$ta>().unwrap().into_owned(),
               b0: tensor1(&[43]).cast_to::<$tb>().unwrap().into_owned(),
               b: tensor2(&[[0]]).cast_to::<$tb>().unwrap().into_owned(),
               boo: PhantomData,
               };
               assert_eq!(pb.run::<$ker>(), pb.reference());
               }
               }
               #[test]

               fn q_mat_mul_sat_2() {
               if $cond {
               let pb = QProblem {
               m: 1,
               k: 1,
               n: 1,
               a0: tensor1(&[0]).cast_to::<$ta>().unwrap().into_owned(),
               a: tensor2(&[[<$ta>::min_value()]]),
               b0: tensor1(&[0]).cast_to::<$tb>().unwrap().into_owned(),
               b: tensor2(&[[<$tb>::one()]]),
               boo: PhantomData,
               };
               assert_eq!(pb.run::<$ker>(), pb.reference());
               }
               }

            #[test]
            fn q_mat_mul_n2() {
                if $cond {
                    let pb = QProblem {
                        m: 1,
                        k: 1,
                        n: 2,
                        a0: tensor1(&[1]).cast_to::<$ta>().unwrap().into_owned(),
                        a: tensor2(&[[<$ta>::zero()]]),
                        b0: tensor1(&[0, 1]).cast_to::<$tb>().unwrap().into_owned(),
                        b: tensor2(&[[<$tb>::zero(), <$tb>::zero()]]),
                        boo: PhantomData,
                    };
                    assert_eq!(pb.run::<$ker>(), pb.reference());
                }
            }

            #[test]
            fn q_mat_mul_k2() {
                if $cond {
                    let pb = QProblem {
                        m: 1,
                        k: 2,
                        n: 1,
                        a0: tensor1(&[0]).cast_to::<$ta>().unwrap().into_owned(),
                        a: tensor2(&[[<$ta>::zero(), <$ta>::one()]]),
                        b0: tensor1(&[0]).cast_to::<$tb>().unwrap().into_owned(),
                        b: tensor2(&[[<$tb>::zero()], [<$tb>::zero()]]),
                        boo: PhantomData,
                    };
                    assert_eq!(pb.run::<$ker>(), pb.reference());
                }
            }
        }
        */
    };
}

#[macro_export]
macro_rules! qmmm_s_frame_tests {
    ($cond:expr, $ker:ty, $ta: ty, $tb: ty, $tc: ty, $ti: ty) => {
        /*
        mod qframe_s {
        use std::marker::PhantomData;
        use tract_data::prelude::*;
        #[allow(unused_imports)]
        use $crate::frame::mmm::tests::*;

        type QProblem = QMatMulProblem<$ta, $tb, $tc, $ti>;

        #[test]
        fn q_mat_mul_1_1_5() {
        if $cond {
        let pb = QProblem {
        m: 1,
        k: 1,
        n: 5,
        a: tensor2(&[[-1]]).cast_to::<$ta>().unwrap().into_owned(),
        a0: tensor0(0i32).cast_to::<$ta>().unwrap().into_owned(),
        b: tensor2(&[[0, 0, 0, 0, -2]]).cast_to::<$tb>().unwrap().into_owned(),
        b0: tensor0(0i32).cast_to::<$tb>().unwrap().into_owned(),
        boo: PhantomData,
        };
        assert_eq!(pb.run::<$ker>(), pb.reference());
        }
        }

        #[test]
        fn q_mat_mul_1_1_1() {
        if $cond {
        let pb = QProblem {
        m: 1,
        k: 1,
        n: 1,
        a: tensor2(&[[11]]).cast_to::<$ta>().unwrap().into_owned(),
        a0: tensor0(10i32).cast_to::<$ta>().unwrap().into_owned(),
        b: tensor2(&[[-1]]).cast_to::<$tb>().unwrap().into_owned(),
        b0: tensor0(0i32).cast_to::<$tb>().unwrap().into_owned(),
        boo: PhantomData,
        };
        assert_eq!(pb.run::<$ker>(), pb.reference());
        }
        }
        }
        */
    };
}
