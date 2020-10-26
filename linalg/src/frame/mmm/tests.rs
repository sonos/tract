use super::*;
use crate::test::*;
use num_traits::{AsPrimitive, Bounded, Zero};
use proptest::collection::vec;
use proptest::prelude::*;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::{Add, AddAssign, Mul, Neg, Sub};
use tract_data::prelude::*;

#[macro_export]
macro_rules! mmm_frame_tests {
    ($cond:expr, $ker:ty, $ta:ty, $tb:ty, $tc:ty, $ti:ty) => {
        mod frame {
            #[allow(unused_imports)]
            use $crate::frame::mmm::tests::*;
            use $crate::num_traits::{AsPrimitive, One, Zero};

            proptest::proptest! {
                #[test]
                fn mat_mul_prepacked_prop((m, k, n, ref a, ref b) in strat_mat_mat_mul()) {
                    if $cond {
                        test_mat_mat_mul_prep::<$ker, $ta, $tb, $tc, $ti>(m, k, n, &**a, &*b)?
                    }
                }

                #[test]
                fn mat_vec_prepacked_prop((m, k, ref a, ref b) in strat_mat_vec_mul()) {
                    if $cond {
                        test_mat_vec_mul_prep::<$ker, $ta, $tb, $tc, $ti>(m, k, a, b)?
                    }
                }

                #[test]
                fn conv_prepacked_prop(pb in strat_conv_1d()) {
                    if $cond {
                        let found = pb.run::<$ker, $tc, $ti>();
                        let expected = pb.expected::<$tc, $ti>();
                        crate::test::check_close(&found, &expected)?;
                    }
                }
            }

            #[test]
            fn mat_mul_1() {
                if $cond {
                    let a: Vec<$ta> = [-3isize, 3, 5, -5, 6, 0, -6, -5, 0, 0, 9, 7]
                        .iter()
                        .map(|x| x.as_())
                        .collect();
                    let b: Vec<$tb> =
                        [-8isize, 5, 5, -3, 5, 7, -8, -1].iter().map(|x| x.as_()).collect();
                    test_mat_mat_mul_prep::<$ker, $ta, $tb, $tc, $ti>(3, 4, 2, &*a, &*b).unwrap()
                }
            }

            #[test]
            fn mat_mul_2() {
                if $cond {
                    let a: Vec<$ta> = [-1isize, -1, 0, 0].iter().map(|x| x.as_()).collect();
                    let b: Vec<$tb> = [0isize, 1, 0, 1].iter().map(|x| x.as_()).collect();
                    test_mat_mat_mul_prep::<$ker, $ta, $tb, $tc, $ti>(2, 2, 2, &*a, &*b).unwrap()
                }
            }

            #[test]
            fn mat_mul_1_2_1() {
                if $cond {
                    test_mat_mat_mul_prep::<$ker, $ta, $tb, $tc, $ti>(
                        1,
                        2,
                        1,
                        &[<$ta>::zero(), <$ta>::one()],
                        &[<$tb>::zero(), <$tb>::one()],
                    )
                    .unwrap()
                }
            }

            #[test]
            fn conv_prepacked_1() {
                if $cond {
                    let filters: Vec<$ta> = vec![1isize.as_()];
                    let data: Vec<$tb> = vec![0.as_(), 1.as_()];
                    let pb = ConvProblem::<$ta, $tb> {
                        ci: 1,
                        co: 1,
                        kt: 1,
                        stride: 1,
                        dilation: 1,
                        filters,
                        data,
                    };
                    let expected: Vec<$tc> = pb.expected::<$tc, $ti>();
                    crate::test::check_close(&*pb.run::<$ker, $tc, $ti>(), &*expected).unwrap();
                }
            }

            #[test]
            fn conv_prepacked_2() {
                if $cond {
                    let mut filters = vec![<$ta>::zero(); 3 * 14 * 2];
                    filters[13 * 6 + 5] = <$ta>::one();
                    let mut data = vec![<$tb>::zero(); 3 * 10];
                    data[8 + 2 * 10] = <$tb>::one(); // last used input
                    let pb = ConvProblem::<$ta, $tb> {
                        ci: 3,
                        co: 14,
                        kt: 2,
                        stride: 3,
                        dilation: 2,
                        filters,
                        data,
                    };
                    let expected: Vec<$tc> = pb.expected::<$tc, $ti>();
                    crate::test::check_close(&*pb.run::<$ker, $tc, $ti>(), &*expected).unwrap();
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
            #[allow(unused_imports)]
            use num_traits::*;
            use std::ops::Neg;
            use $crate::frame::mmm::tests::ConvProblem;

            #[test]
            fn conv_prepacked_3() {
                if $cond {
                    let mut filters = vec![<$ta>::zero(); 4];
                    filters[3] = <$ta>::one().neg();
                    let data = vec![<$tb>::zero(); 4];
                    let pb = ConvProblem::<$ta, $tb> {
                        ci: 1,
                        co: 1,
                        kt: 1,
                        stride: 1,
                        dilation: 1,
                        filters,
                        data,
                    };
                    let expected: Vec<$tc> = pb.expected::<$tc, $ti>();
                    crate::test::check_close(&*pb.run::<$ker, $tc, $ti>(), &*expected).unwrap();
                }
            }
        }
    };
}

pub fn strat_mat_mat_mul<TA: LADatum, TB: LADatum>(
) -> BoxedStrategy<(usize, usize, usize, Vec<TA>, Vec<TB>)> {
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
        .boxed()
}

pub fn strat_mat_vec_mul<TA: LADatum, TB: LADatum>(
) -> BoxedStrategy<(usize, usize, Vec<TA>, Vec<TB>)> {
    (1usize..15, 1usize..15)
        .prop_flat_map(move |(m, k)| {
            (
                Just(m),
                Just(k),
                proptest::collection::vec(TA::strat(), m * k),
                proptest::collection::vec(TB::strat(), k),
            )
        })
        .boxed()
}

pub fn test_mat_mat_mul_prep<K: MatMatMulKer<TA, TB, TC, TI> + 'static, TA, TB, TC, TI>(
    m: usize,
    k: usize,
    n: usize,
    a: &[TA],
    b: &[TB],
) -> Result<(), proptest::test_runner::TestCaseError>
where
    TA: LADatum + AsPrimitive<TI> + 'static,
    TB: LADatum + AsPrimitive<TI> + 'static,
    TC: LADatum + AsPrimitive<TI> + 'static,
    TI: LADatum + AsPrimitive<TC> + 'static + Neg<Output = TI>,
    i32: AsPrimitive<TI>,
    usize: AsPrimitive<TI>,
{
    let op = MatMatMulImpl::<K, TA, TB, TC, TI>::new(m, k, n);
    unsafe {
        let mut packed_a =
            Tensor::uninitialized_aligned::<TA>(&[op.a_pack().len()], op.a_pack().alignment())
                .unwrap();
        op.a_pack().pack(packed_a.as_ptr_mut_unchecked(), a.as_ptr(), k as isize, 1);

        let mut packed_b =
            Tensor::uninitialized_aligned::<TB>(&[op.b_pack().len()], op.b_pack().alignment())
                .unwrap();
        op.b_pack().pack(packed_b.as_ptr_mut_unchecked(), b.as_ptr(), n as isize, 1);

        let mut found = vec![TC::max_value(); m * n];

        op.run(packed_a.as_ptr_unchecked(), packed_b.as_ptr_unchecked(), found.as_mut_ptr(), &[]);

        let mut expected = vec![TC::zero(); m * n];
        for x in 0..n {
            for y in 0..m {
                let mut v: TI = TI::zero();
                for i in 0..k {
                    let a: TI = a[i + k * y].as_();
                    let b: TI = b[x + i * n].as_();
                    v = v + a * b;
                }
                expected[x + y * n] = v.as_();
            }
        }
        crate::test::check_close(&*found, &*expected)
    }
}

pub fn test_mat_vec_mul_prep<K: MatMatMulKer<TA, TB, TC, TI> + 'static, TA, TB, TC, TI>(
    m: usize,
    k: usize,
    a: &[TA],
    b: &[TB],
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
        let mut op = MatMatMulImpl::<K, TA, TB, TC, TI>::new(m, k, 1);
        op.b_vec_from_data_and_stride(1);
        op.c_vec_from_data_and_stride(1);
        let mut packed_a =
            Tensor::uninitialized_aligned::<TA>(&[op.a_pack().len()], op.a_pack().alignment())
                .unwrap();
        op.a_pack().pack(packed_a.as_ptr_mut_unchecked(), a.as_ptr(), k as isize, 1);

        let mut found = vec![TC::zero(); m];

        op.run(packed_a.as_ptr_unchecked(), b.as_ptr(), found.as_mut_ptr(), &[]);

        let mut expected = vec![TC::zero(); m];
        for y in 0..m {
            let mut inter = TI::zero();
            for i in 0..k {
                let a: TI = a[i + k * y].as_();
                let b: TI = b[i].as_();
                inter = inter + a * b;
            }
            expected[y] = inter.as_();
        }

        crate::test::check_close(&*found, &*expected)
    }
}

pub unsafe fn fused_op<
    K: MatMatMulKer<TA, TB, TC, TI> + 'static,
    TA,
    TB,
    TC,
    TI,
    F: Fn(&mut [TI]),
>(
    m: usize,
    k: usize,
    n: usize,
    spec: &[FusedSpec<TI>],
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
    let a = vec![TA::one(); m * k];
    let b = vec![TB::one(); n * k];
    let op = MatMatMulImpl::<K, TA, TB, TC, TI>::new(m, k, n);

    let mut packed_a =
        Tensor::uninitialized_aligned::<TA>(&[op.a_pack().len()], op.a_pack().alignment()).unwrap();
    op.a_pack().pack(packed_a.as_ptr_mut_unchecked(), a.as_ptr(), k as isize, 1);

    let mut packed_b =
        Tensor::uninitialized_aligned::<TB>(&[op.b_pack().len()], op.b_pack().alignment()).unwrap();
    op.b_pack().pack(packed_b.as_ptr_mut_unchecked(), b.as_ptr(), n as isize, 1);

    let mut found = vec![TC::zero(); m * n];

    op.run(packed_a.as_ptr_unchecked(), packed_b.as_ptr_unchecked(), found.as_mut_ptr(), spec);

    let mut inter = vec![TI::zero(); m * n];
    for x in 0..n {
        for y in 0..m {
            let mut s = TI::zero();
            for i in 0..k {
                s += a[i + k * y].as_() * b[x + i * n].as_()
            }
            inter[x + y * n] = s;
        }
    }
    expect(&mut inter);
    let expected: Vec<TC> = inter.into_iter().map(|i| i.as_()).collect();

    crate::test::check_close(&*found, &*expected)
}

pub unsafe fn row_add<K: MatMatMulKer<TA, TB, TC, TI> + 'static, TA, TB, TC, TI>(
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
    fused_op::<K, TA, TB, TC, TI, _>(m, k, n, &[FusedSpec::PerRowAdd(bias.clone())], |exp| {
        for x in 0..n {
            for y in 0..m {
                exp[x + y * n] += bias[y]
            }
        }
    })
}

pub unsafe fn row_mul<K: MatMatMulKer<TA, TB, TC, TI> + 'static, TA, TB, TC, TI>(
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
    fused_op::<K, TA, TB, TC, TI, _>(m, k, n, &[FusedSpec::PerRowMul(bias.clone())], |exp| {
        for x in 0..n {
            for y in 0..m {
                exp[x + y * n] *= bias[y]
            }
        }
    })
}

pub unsafe fn col_add<K: MatMatMulKer<TA, TB, TC, TI> + 'static, TA, TB, TC, TI>(
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
    fused_op::<K, TA, TB, TC, TI, _>(m, k, n, &[FusedSpec::PerColAdd(bias.clone())], |exp| {
        for x in 0..n {
            for y in 0..m {
                exp[x + y * n] += bias[x]
            }
        }
    })
}

pub unsafe fn col_mul<K: MatMatMulKer<TA, TB, TC, TI> + 'static, TA, TB, TC, TI>(
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
    fused_op::<K, TA, TB, TC, TI, _>(m, k, n, &[FusedSpec::PerColMul(bias.clone())], |exp| {
        for x in 0..n {
            for y in 0..m {
                exp[x + y * n] *= bias[x]
            }
        }
    })
}

pub unsafe fn max<K: MatMatMulKer<TA, TB, TC, TI>, TA, TB, TC, TI>(
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
    fused_op::<K, TA, TB, TC, TI, _>(m, k, n, &[FusedSpec::Max(five)], |exp| {
        exp.iter_mut().for_each(|x| *x = if *x < five { five } else { *x })
    })
}

pub unsafe fn min<K: MatMatMulKer<TA, TB, TC, TI>, TA, TB, TC, TI>(
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
    fused_op::<K, TA, TB, TC, TI, _>(m, k, n, &[FusedSpec::Min(five)], |exp| {
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
    pub filters: Vec<TA>,
    pub data: Vec<TB>,
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

    pub fn expected<TC, TI>(&self) -> Vec<TC>
    where
        TA: LADatum + AsPrimitive<TI>,
        TB: LADatum + AsPrimitive<TI>,
        TC: LADatum,
        TI: LADatum + AsPrimitive<TC>,
    {
        let mut expect = vec![TI::zero(); self.co * self.output_width()];
        for x in 0..self.output_width() {
            for ico in 0..self.co {
                for ikt in 0..self.kt {
                    for ici in 0..self.ci {
                        let f = self.filters[ici * self.kt + ikt + self.ci * self.kt * ico];
                        let d = self.data
                            [x * self.stride + ikt * self.dilation + ici * self.input_width()];
                        let ref mut pv = expect[x + ico * self.output_width()];
                        *pv += f.as_() * d.as_();
                    }
                }
            }
        }
        expect.into_iter().map(|ti| ti.as_()).collect()
    }

    pub fn run<K: MatMatMulKer<TA, TB, TC, TI>, TC, TI>(&self) -> Vec<TC>
    where
        TA: LADatum + AsPrimitive<TI> + 'static,
        TB: LADatum + AsPrimitive<TI> + 'static,
        TC: LADatum + AsPrimitive<TI> + 'static,
        TI: LADatum + AsPrimitive<TC> + 'static + Neg<Output = TI>,
        i32: AsPrimitive<TI>,
        usize: AsPrimitive<TI>,
    {
        unsafe {
            let mut op = MatMatMulImpl::<K, TA, TB, TC, TI>::new(self.m(), self.k(), self.n());
            op.b_from_data_and_offsets(&self.data_rows_offsets(), &self.data_cols_offsets());
            let mut packed_a =
                Tensor::uninitialized_aligned::<TA>(&[op.a_pack().len()], op.a_pack().alignment())
                    .unwrap();
            op.a_pack().pack(
                packed_a.as_ptr_mut_unchecked(),
                self.filters.as_ptr(),
                self.k() as isize,
                1,
            );

            let mut found: Vec<TC> = vec![TC::max_value(); self.co * self.output_width()];
            op.run(packed_a.as_ptr_unchecked(), self.data.as_ptr(), found.as_mut_ptr(), &[]);
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
                Just(stride),
                Just(dilation),
                proptest::collection::vec(TA::strat(), ci * co * kt),
                proptest::collection::vec(TB::strat(), t * ci),
            )
        })
        .prop_map(move |(ci, co, kt, stride, dilation, filters, data)| ConvProblem {
            ci,
            co,
            kt,
            stride,
            dilation,
            filters,
            data,
        })
        .boxed()
}

#[derive(Debug)]
pub struct QMatMulProblem<TA, TB, TC, TI> {
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub a: Vec<TA>,
    pub a0: QuantizedParam<TA>,
    pub b: Vec<TB>,
    pub b0: QuantizedParam<TB>,
    pub boo: PhantomData<(TC, TI)>,
}

impl<TI: Arbitrary + 'static> Arbitrary for QuantizedParam<TI> {
    type Parameters = usize;
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with(n: usize) -> Self::Strategy {
        prop_oneof![
            any::<TI>().prop_map(QuantizedParam::Scalar),
            vec(any::<TI>(), n..=n).prop_map(QuantizedParam::Vector),
        ]
        .boxed()
    }
}

impl<TA, TB, TC, TI> Arbitrary for QMatMulProblem<TA, TB, TC, TI>
where
    TA: Arbitrary + 'static + Debug + 'static,
    TB: Arbitrary + 'static + Debug + 'static,
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
                    any_with::<QuantizedParam<TA>>(m),
                    vec(any::<TB>(), k * n..=k * n),
                    any_with::<QuantizedParam<TB>>(n),
                )
            })
            .prop_map(|(m, k, n, a, a0, b, b0)| QMatMulProblem {
                m,
                k,
                n,
                a,
                a0,
                b,
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
    pub fn reference(&self) -> Vec<TC> {
        let mut i = vec![TI::zero(); self.m * self.n];
        for m in 0..self.m {
            for n in 0..self.n {
                for k in 0..self.k {
                    let a: TI = self.a[k + self.k * m].as_();
                    let b: TI = self.b[n + self.n * k].as_();
                    let a0 = match &self.a0 {
                        QuantizedParam::Scalar(a0) => a0.as_(),
                        QuantizedParam::Vector(a0) => a0[m].as_(),
                    };
                    let b0 = match &self.b0 {
                        QuantizedParam::Scalar(b0) => b0.as_(),
                        QuantizedParam::Vector(b0) => b0[n].as_(),
                    };
                    i[n + self.n * m] += (a - a0) * (b - b0);
                }
            }
        }
        i.iter().map(|i| i.max(&TC::min_value().as_()).min(&TC::max_value().as_()).as_()).collect()
    }

    pub fn run<K: MatMatMulKer<TA, TB, TC, TI>>(&self) -> Vec<TC> {
        unsafe {
            let mut c = vec![TC::zero(); self.m * self.n];
            let mut mmm = MatMatMulImpl::from(MatMatMulImpl::<K, TA, TB, TC, TI>::new(
                self.m, self.k, self.n,
            ));
            let mut packed_a = Tensor::uninitialized_aligned::<TA>(
                &[mmm.a_pack().len()],
                mmm.a_pack().alignment(),
            )
            .unwrap();
            mmm.a_pack().pack(packed_a.as_ptr_mut_unchecked(), self.a.as_ptr(), self.k as isize, 1);

            let mut packed_b = Tensor::uninitialized_aligned::<TB>(
                &[mmm.b_pack().len()],
                mmm.b_pack().alignment(),
            )
            .unwrap();
            mmm.b_pack().pack(packed_b.as_ptr_mut_unchecked(), self.b.as_ptr(), self.n as isize, 1);

            match &self.a0 {
                QuantizedParam::Scalar(a0) => mmm.set_zero_point_a_scalar(*a0),
                QuantizedParam::Vector(a0) => mmm.set_zero_point_a_vector(a0.clone()),
            }
            match &self.b0 {
                QuantizedParam::Scalar(b0) => mmm.set_zero_point_b_scalar(*b0),
                QuantizedParam::Vector(b0) => mmm.set_zero_point_b_vector(b0.clone()),
            }
            mmm.run(packed_a.as_ptr_unchecked(), packed_b.as_ptr_unchecked(), c.as_mut_ptr(), &[]);
            c
        }
    }
}

#[macro_export]
macro_rules! qmmm_frame_tests {
    ($cond:expr, $ker:ty, $ta: ty, $tb: ty, $tc: ty, $ti: ty) => {
        mod qframe {
            use proptest::prelude::*;
            use std::marker::PhantomData;
            #[allow(unused_imports)]
            use $crate::frame::mmm::tests::*;
            use $crate::frame::mmm::QuantizedParam;

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
                    let pb = QMatMulProblem {
                        m: 1,
                        k: 1,
                        n: 1,
                        a0: QuantizedParam::Vector(vec![1]),
                        a: vec![0],
                        b0: QuantizedParam::Vector(vec![1]),
                        b: vec![0],
                        boo: PhantomData,
                    };
                    assert_eq!(pb.run::<$ker>(), pb.reference());
                }
            }

            #[test]
            fn q_mat_mul_sat_1() {
                if $cond {
                    let pb = QMatMulProblem {
                        m: 1,
                        k: 1,
                        n: 1,
                        a0: QuantizedParam::Vector(vec![0]),
                        a: vec![3],
                        b0: QuantizedParam::Vector(vec![43]),
                        b: vec![0],
                        boo: PhantomData,
                    };
                    assert_eq!(pb.run::<$ker>(), pb.reference());
                }
            }
            #[test]

            fn q_mat_mul_sat_2() {
                if $cond {
                    let pb = QMatMulProblem {
                        m: 1,
                        k: 1,
                        n: 1,
                        a0: QuantizedParam::Vector(vec![0]),
                        a: vec![<$ta>::min_value()],
                        b0: QuantizedParam::Vector(vec![0]),
                        b: vec![1],
                        boo: PhantomData,
                    };
                    assert_eq!(pb.run::<$ker>(), pb.reference());
                }
            }

            #[test]
            fn q_mat_mul_n2() {
                if $cond {
                    let pb = QMatMulProblem {
                        m: 1,
                        k: 1,
                        n: 2,
                        a: vec![0],
                        a0: QuantizedParam::Vector(vec![1]),
                        b: vec![0, 0],
                        b0: QuantizedParam::Vector(vec![0, 1]),
                        boo: PhantomData,
                    };
                    assert_eq!(pb.run::<$ker>(), pb.reference());
                }
            }

            #[test]
            fn q_mat_mul_k2() {
                if $cond {
                    let pb = QMatMulProblem {
                        m: 1,
                        k: 2,
                        n: 1,
                        a: vec![0, 1],
                        a0: QuantizedParam::Vector(vec![0]),
                        b: vec![0, 1],
                        b0: QuantizedParam::Vector(vec![0]),
                        boo: PhantomData,
                    };
                    assert_eq!(pb.run::<$ker>(), pb.reference());
                }
            }
        }
    };
}

#[macro_export]
macro_rules! qmmm_s_frame_tests {
    ($cond:expr, $ker:ty, $ta: ty, $tb: ty, $tc: ty, $ti: ty) => {
        mod qframe_s {
            use std::marker::PhantomData;
            #[allow(unused_imports)]
            use $crate::frame::mmm::tests::*;
            use $crate::frame::mmm::QuantizedParam;

            #[test]
            fn q_mat_mul_1_1_5() {
                if $cond {
                    let pb = QMatMulProblem {
                        m: 1,
                        k: 1,
                        n: 5,
                        a: vec![-1],
                        a0: QuantizedParam::Scalar(0),
                        b: vec![0, 0, 0, 0, -2],
                        b0: QuantizedParam::Scalar(0),
                        boo: PhantomData,
                    };
                    assert_eq!(pb.run::<$ker>(), pb.reference());
                }
            }

            #[test]
            fn q_mat_mul_1_1_1() {
                if $cond {
                    let pb = QMatMulProblem {
                        m: 1,
                        k: 1,
                        n: 1,
                        a: vec![11],
                        a0: QuantizedParam::Scalar(10),
                        b: vec![-1],
                        b0: QuantizedParam::Scalar(0),
                        boo: PhantomData,
                    };
                    assert_eq!(pb.run::<$ker>(), pb.reference());
                }
            }
        }
    };
}
