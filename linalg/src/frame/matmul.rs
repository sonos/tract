use num::Zero;
use std::fmt::Debug;
use std::ops::{Add, Mul};

use std::marker::PhantomData;

pub trait MatMul<T: Copy + Add + Mul + Zero>: Send + Sync + Debug {
    fn packed_a_len(&self) -> usize;
    fn pack_a(&self, pa: *mut T, a: *const T, rsa: isize, csa: isize);
    fn packed_b_len(&self) -> usize;
    fn pack_b(&self, pb: *mut T, b: *const T, rsb: isize, csb: isize);

    fn mat_mul_prepacked(&self, pa: *const T, pb: *const T, c: *mut T, rsc: isize, csc: isize);
}

pub trait PackedMatMulKer<T: Copy + Add + Mul + Zero>: Copy + Clone + Debug + Send + Sync {
    #[inline(always)]
    fn kernel(k: usize, a: *const T, b: *const T, c: *mut T, rsc: usize, csc: usize);
    #[inline(always)]
    fn mr() -> usize;
    #[inline(always)]
    fn nr() -> usize;
}

#[derive(Copy, Clone, Debug)]
pub struct PackedMatMul<K, T>
where
    K: PackedMatMulKer<T> + Debug,
    T: Copy + Add + Mul + Zero + Debug + Send + Sync,
{
    m: usize,
    k: usize,
    n: usize,
    _kernel: PhantomData<(K, T)>,
}

impl<K, T> PackedMatMul<K, T>
where
    K: PackedMatMulKer<T>,
    T: Copy + Add + Mul + Zero + Debug + Send + Sync,
{
    pub fn new(m: usize, k: usize, n: usize) -> PackedMatMul<K, T> {
        PackedMatMul {
            m,
            k,
            n,
            _kernel: PhantomData,
        }
    }

    fn pack_panel_a(&self, pa: *mut T, a: *const T, rsa: isize, csa: isize, rows: usize) {
        let mr = K::mr();
        for i in 0..self.k {
            for j in 0..rows {
                unsafe {
                    *pa.offset((i * mr + j) as isize) =
                        *a.offset(i as isize * csa + j as isize * rsa)
                }
            }
        }
    }

    fn pack_panel_b(&self, pb: *mut T, b: *const T, rsb: isize, csb: isize, cols: usize) {
        let nr = K::nr();
        for i in 0..self.k {
            for j in 0..cols {
                unsafe {
                    *pb.offset((i * nr + j) as isize) =
                        *b.offset(j as isize * csb + i as isize * rsb)
                }
            }
        }
    }
}

impl<K, T> MatMul<T> for PackedMatMul<K, T>
where
    K: PackedMatMulKer<T>,
    T: Copy + Add + Mul + Zero + Debug + Send + Sync,
{
    fn packed_a_len(&self) -> usize {
        let mr = K::mr();
        (self.m + mr - 1) / mr * mr * self.k
    }

    fn pack_a(&self, pa: *mut T, a: *const T, rsa: isize, csa: isize) {
        let mr = K::mr();
        unsafe {
            for p in 0..(self.m / mr) {
                self.pack_panel_a(
                    pa.offset((p * mr * self.k) as isize),
                    a.offset((p * mr) as isize * rsa),
                    rsa,
                    csa,
                    mr,
                )
            }
            if self.m % mr != 0 {
                self.pack_panel_a(
                    pa.offset((self.m / mr * mr * self.k) as isize),
                    a.offset((self.m / mr * mr) as isize * rsa),
                    rsa,
                    csa,
                    self.m % mr,
                )
            }
        }
    }

    fn packed_b_len(&self) -> usize {
        (self.n + K::nr() - 1) / K::nr() * K::nr() * self.k
    }

    fn pack_b(&self, pb: *mut T, b: *const T, rsb: isize, csb: isize) {
        let nr = K::nr();
        unsafe {
            for p in 0..(self.n / nr) {
                self.pack_panel_b(
                    pb.offset((p * nr * self.k) as isize),
                    b.offset((p * nr) as isize * csb),
                    rsb,
                    csb,
                    K::nr(),
                )
            }
            if self.n % nr != 0 {
                self.pack_panel_b(
                    pb.offset((self.n / nr * nr * self.k) as isize),
                    b.offset((self.n / nr * nr) as isize * csb),
                    rsb,
                    csb,
                    self.n % nr,
                )
            }
        }
    }

    fn mat_mul_prepacked(&self, pa: *const T, pb: *const T, c: *mut T, rsc: isize, csc: isize) {
        let mr = K::mr();
        let nr = K::nr();
        let m = self.m;
        let k = self.k;
        let n = self.n;
        let mut tmpc = vec![T::zero(); mr * nr];
        unsafe {
            for ia in 0..m / mr {
                for ib in 0..n / nr {
                    K::kernel(
                        k,
                        pa.offset((ia * k * mr) as isize),
                        pb.offset((ib * k * nr) as isize),
                        c.offset((mr * ia) as isize * rsc + (nr * ib) as isize * csc),
                        rsc as usize,
                        csc as usize,
                    );
                }
                if n % nr != 0 {
                    K::kernel(
                        k,
                        pa.offset((ia * k * mr) as isize),
                        pb.offset((n / nr * k * nr) as isize),
                        tmpc.as_mut_ptr(),
                        nr,
                        1,
                    );
                    for y in 0..mr {
                        for x in 0..(n % nr) {
                            *c.offset(
                                (mr * ia + y) as isize * rsc + (x + n / nr * nr) as isize * csc,
                            ) = tmpc[y * nr + x];
                        }
                    }
                }
            }
            if m % mr != 0 {
                for ib in 0..n / nr {
                    K::kernel(
                        k,
                        pa.offset((m / mr * mr * k) as isize),
                        pb.offset((ib * nr * k) as isize),
                        tmpc.as_mut_ptr(),
                        nr,
                        1,
                    );
                    for y in 0..(m % mr) {
                        for x in 0..nr {
                            *c.offset(
                                (y + m / mr * mr) as isize * rsc + (x + ib * nr) as isize * csc,
                            ) = tmpc[y * nr + x];
                        }
                    }
                }
                if n % nr != 0 {
                    K::kernel(
                        k,
                        pa.offset((m / mr * mr * k) as isize),
                        pb.offset((n / nr * nr * k) as isize),
                        tmpc.as_mut_ptr(),
                        nr,
                        1,
                    );
                    for y in 0..(m % mr) {
                        for x in 0..(n % nr) {
                            *c.offset(
                                (y + m / mr * mr) as isize * rsc + (x + n / nr * nr) as isize * csc,
                            ) = tmpc[y * nr + x];
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
pub mod test {
    use super::*;
    use proptest::prelude::*;
    use proptest::*;

    pub fn strat(k_fact: usize) -> BoxedStrategy<(usize, usize, usize, Vec<f32>, Vec<f32>)> {
        (1usize..35, 1usize..35, 1usize..35)
            .prop_flat_map(move |(m, k, n)| {
                (
                    Just(m),
                    Just(k * k_fact),
                    Just(n),
                    proptest::collection::vec((-10..10).prop_map(|a| a as f32), m * k * k_fact),
                    proptest::collection::vec((-10..10).prop_map(|a| a as f32), n * k * k_fact),
                )
            })
            .boxed()
    }

    pub fn test_mat_mul_prep_f32<MM: MatMul<f32>>(
        mm: MM,
        m: usize,
        k: usize,
        n: usize,
        a: &[f32],
        b: &[f32],
    ) -> Result<(), proptest::test_runner::TestCaseError> {
        let mut packed_a = vec![0.0f32; mm.packed_a_len()];
        mm.pack_a(packed_a.as_mut_ptr(), a.as_ptr(), k as isize, 1);

        let mut packed_b = vec![0.0f32; mm.packed_b_len()];
        mm.pack_b(packed_b.as_mut_ptr(), b.as_ptr(), n as isize, 1);

        let mut expect = vec![9999.0f32; m * n];
        let mut found = vec![9999.0f32; m * n];
        unsafe {
            mm.mat_mul_prepacked(
                packed_a.as_ptr(),
                packed_b.as_ptr(),
                found.as_mut_ptr(),
                n as isize,
                1,
            );
            ::matrixmultiply::sgemm(
                m,
                k,
                n,
                1.0,
                a.as_ptr(),
                k as _,
                1,
                b.as_ptr(),
                n as _,
                1,
                0.0,
                expect.as_mut_ptr(),
                n as _,
                1,
            );
        }
        prop_assert_eq!(expect, found);
        Ok(())
    }

}
