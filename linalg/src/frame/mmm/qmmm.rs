use std::fmt;
use std::ops::{Add, Deref, Mul};

use num_traits::Zero;

use super::*;

pub trait QMatMatMul<TA, TB, TC, TI>:
    fmt::Debug + fmt::Display + objekt::Clone + Send + Sync
where
    TA: Copy + Zero,
    TB: Copy + Zero,
    TC: Copy,
    TI: Copy + Add + Mul + Zero + fmt::Debug,
{
    fn as_mmm(&self) -> &dyn MatMatMul<TA, TB, TC, TI>;
    fn as_mmm_mut(&mut self) -> &mut dyn MatMatMul<TA, TB, TC, TI>;

    unsafe fn set_zero_point_a_scalar(&mut self, value: TI);
    unsafe fn set_zero_point_a_vector(&mut self, values: Vec<TI>);
    unsafe fn set_zero_point_b_scalar(&mut self, value: TI);
    unsafe fn set_zero_point_b_vector(&mut self, values: Vec<TI>);

    unsafe fn run(&self, a: *const TA, b: *const TB, c: *mut TC);
}

clone_trait_object!(<TA, TB, TC, TI> QMatMatMul<TA, TB, TC, TI> where
    TA: Copy + Zero,
    TB: Copy + Zero,
    TC: Copy,
    TI: Copy + Add + Mul + Zero + fmt::Debug,
);

#[derive(Debug, Clone)]
pub struct QMatMatMulImpl<K, TA, TB, TC, TI>
where
    TA: Copy + Zero,
    TB: Copy + Zero,
    TC: Copy,
    TI: Copy + Add + Mul + Zero + fmt::Debug,
    K: MatMatMulKer<TA, TB, TC, TI>,
{
    pub mmm: MatMatMulImpl<K, TA, TB, TC, TI>,
    pub zero_point_a: Option<Vec<TI>>,
    pub zero_point_b: Option<Vec<TI>>,
}

impl<K, TA, TB, TC, TI> From<MatMatMulImpl<K, TA, TB, TC, TI>> for QMatMatMulImpl<K, TA, TB, TC, TI>
where
    TA: Copy + Zero,
    TB: Copy + Zero,
    TC: Copy,
    TI: Copy + Add + Mul + Zero + fmt::Debug,
    K: MatMatMulKer<TA, TB, TC, TI>,
{
    fn from(mmm: MatMatMulImpl<K, TA, TB, TC, TI>) -> QMatMatMulImpl<K, TA, TB, TC, TI> {
        QMatMatMulImpl { mmm, zero_point_a: None, zero_point_b: None }
    }
}

impl<K, TA, TB, TC, TI> Deref for QMatMatMulImpl<K, TA, TB, TC, TI>
where
    TA: Copy + Zero,
    TB: Copy + Zero,
    TC: Copy,
    TI: Copy + Add + Mul + Zero + fmt::Debug,
    K: MatMatMulKer<TA, TB, TC, TI>,
{
    type Target = MatMatMulImpl<K, TA, TB, TC, TI>;
    fn deref(&self) -> &Self::Target {
        &self.mmm
    }
}

unsafe impl<K, TA, TB, TC, TI> Send for QMatMatMulImpl<K, TA, TB, TC, TI>
where
    TA: Copy + Zero,
    TB: Copy + Zero,
    TC: Copy,
    TI: Copy + Add + Mul + Zero + fmt::Debug,
    K: MatMatMulKer<TA, TB, TC, TI>,
{
}

unsafe impl<K, TA, TB, TC, TI> Sync for QMatMatMulImpl<K, TA, TB, TC, TI>
where
    TA: Copy + Zero,
    TB: Copy + Zero,
    TC: Copy,
    TI: Copy + Add + Mul + Zero + fmt::Debug,
    K: MatMatMulKer<TA, TB, TC, TI>,
{
}

impl<K, TA, TB, TC, TI> QMatMatMul<TA, TB, TC, TI> for QMatMatMulImpl<K, TA, TB, TC, TI>
where
    TA: Copy + Zero + fmt::Debug,
    TB: Copy + Zero + fmt::Debug,
    TC: Copy + fmt::Debug,
    TI: Copy + Add + Mul + Zero + fmt::Debug,
    K: MatMatMulKer<TA, TB, TC, TI>,
{
    fn as_mmm(&self) -> &dyn MatMatMul<TA, TB, TC, TI> {
        &self.mmm
    }

    fn as_mmm_mut(&mut self) -> &mut dyn MatMatMul<TA, TB, TC, TI> {
        &mut self.mmm
    }

    unsafe fn set_zero_point_a_scalar(&mut self, value: TI) {
        self.zero_point_a = Some(vec![value; self.m() + K::mr() - 1 / K::mr() * K::mr()])
    }

    unsafe fn set_zero_point_b_scalar(&mut self, value: TI) {
        self.zero_point_b = Some(vec![value; self.n() + K::nr() - 1 / K::nr() * K::nr()])
    }

    unsafe fn set_zero_point_a_vector(&mut self, mut values: Vec<TI>) {
        let wanted = self.m() + K::mr() - 1 / K::mr() * K::mr();
        while values.len() < wanted {
            values.push(values[values.len() - 1])
        }
        self.zero_point_a = Some(values)
    }

    unsafe fn set_zero_point_b_vector(&mut self, mut values: Vec<TI>) {
        let wanted = self.n() + K::nr() - 1 / K::nr() * K::nr();
        while values.len() < wanted {
            values.push(values[values.len() - 1])
        }
        self.zero_point_b = Some(values)
    }

    unsafe fn run(&self, a: *const TA, b: *const TB, c: *mut TC) {
        self.mmm.run(a, b ,c)
    }
}

impl<K, TA, TB, TC, TI> fmt::Display for QMatMatMulImpl<K, TA, TB, TC, TI>
where
    TA: Copy + Zero,
    TB: Copy + Zero,
    TC: Copy,
    TI: Copy + Add + Mul + Zero + fmt::Debug,
    K: MatMatMulKer<TA, TB, TC, TI>,
{
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(
            fmt,
            "A:{}, B:{} C:{} (m:{}, k:{}, n:{})",
            self.a_storage, self.b_storage, self.c_storage, self.m, self.k, self.n
        )
    }
}

#[cfg(test)]
#[allow(dead_code)]
#[macro_use]
pub mod test {
    use super::*;
    use crate::align::Buffer;
    use proptest::collection::vec;
    use proptest::prelude::*;
    use proptest::test_runner::TestCaseResult;

    #[derive(Debug)]
    pub struct QMatMulProblem {
        pub m: usize,
        pub k: usize,
        pub n: usize,
        pub a: Vec<i8>,
        pub a0: Vec<i8>,
        pub b: Vec<i8>,
        pub b0: Vec<i8>,
    }

    impl Arbitrary for QMatMulProblem {
        type Parameters = ();
        type Strategy = BoxedStrategy<Self>;

        fn arbitrary_with(_: Self::Parameters) -> Self::Strategy {
            (1usize..10, 1usize..10, 1usize..10)
                .prop_flat_map(|(m, k, n)| {
                    (
                        Just(m),
                        Just(k),
                        Just(n),
                        vec(any::<i8>(), m * k..=m * k),
                        vec(any::<i8>(), m..=m),
                        vec(any::<i8>(), k * n..=k * n),
                        vec(any::<i8>(), n..=n),
                    )
                })
                .prop_map(|(m, k, n, a, a0, b, b0)| QMatMulProblem { m, k, n, a, a0, b, b0 })
                .boxed()
        }
    }

    impl QMatMulProblem {
        pub fn ref_i32(&self) -> Vec<i32> {
            let mut c = vec![0; self.m * self.n];
            for m in 0..self.m {
                for n in 0..self.n {
                    for k in 0..self.k {
                        let a = self.a[k + self.k * m] as i32;
                        let b = self.b[n + self.n * k] as i32;
                        let a0 = self.a0[m] as i32;
                        let b0 = self.b0[n] as i32;
                        c[n + self.n * m] += (a - a0) * (b - b0);
                    }
                }
            }
            c
        }

        pub fn run_i32<K: MatMatMulKer<i8, i8, i32, i32>>(&self) -> Vec<i32> {
            unsafe {
                let mut c = vec![0i32; self.m * self.n];
                let mmm = QMatMatMulImpl::from(MatMatMulImpl::<K, i8, i8, i32, i32>::new(
                    self.m, self.k, self.n,
                ));
                let mut packed_a =
                    Buffer::uninitialized(mmm.a_pack().len(), mmm.a_pack().alignment());
                mmm.a_pack().pack(packed_a.as_mut_ptr(), self.a.as_ptr(), self.k as isize, 1);
                let mut packed_b =
                    Buffer::uninitialized(mmm.b_pack().len(), mmm.b_pack().alignment());
                mmm.b_pack().pack(packed_b.as_mut_ptr(), self.b.as_ptr(), self.n as isize, 1);
                mmm.run(packed_a.as_ptr(), packed_b.as_ptr(), c.as_mut_ptr());
                c
            }
        }
    }

    #[macro_export]
    macro_rules! qmmm_frame_tests {
        ($cond:expr, $ker:ty) => {
            mod qframe {
                use proptest::prelude::*;
                #[allow(unused_imports)]
                use $crate::frame::mmm::qmmm::test::*;

                proptest::proptest! {
                    #[test]
                    fn q_mat_mul_i8_i32_prop(pb in any::<QMatMulProblem>()) {
                        if $cond {
                            prop_assert_eq!(pb.run_i32::<$ker>(), pb.ref_i32())
                        }
                    }
                }

                #[test]
                fn q_mat_mul_i8_i32_1() {
                    if $cond {
                        let pb = QMatMulProblem {
                            m: 1,
                            k: 1,
                            n: 1,
                            a0: vec![1],
                            a: vec![0],
                            b0: vec![1],
                            b: vec![0],
                        };
                        assert_eq!(pb.run_i32::<$ker>(), pb.ref_i32());
                    }
                }
            }
        };
    }
}
