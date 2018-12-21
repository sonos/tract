use num::Zero;
use std::ops::{Add, Mul};
use std::fmt::Debug;

pub trait MatMul<T: Copy + Add + Mul + Zero> : Send + Sync + Debug {
    #[inline(always)]
    fn kernel(&self, k: usize, a: *const T, b: *const T, c: *mut T, rsc: usize);
    #[inline(always)]
    fn mr(&self) -> usize;
    #[inline(always)]
    fn nr(&self) -> usize;

    fn packed_a_len(&self, m: usize, k: usize) -> usize {
        (m + self.mr() - 1) / self.mr() * self.mr() * k
    }

    fn pack_a(&self, m: usize, k: usize, pa: *mut T, a: *const T, rsa: isize, csa: isize) {
        let mr = self.mr();
        unsafe {
            for p in 0..(m / mr) {
                self.pack_panel_a(
                    pa.offset((p * mr * k) as isize),
                    a.offset((p * mr) as isize * rsa),
                    rsa,
                    csa,
                    mr,
                    k,
                )
            }
            if m % mr != 0 {
                self.pack_panel_a(
                    pa.offset((m / mr * mr * k) as isize),
                    a.offset((m / mr * mr) as isize * rsa),
                    rsa,
                    csa,
                    m % mr,
                    k,
                )
            }
        }
    }

    fn pack_panel_a(
        &self,
        pa: *mut T,
        a: *const T,
        rsa: isize,
        csa: isize,
        rows: usize,
        k: usize,
    ) {
        let mr = self.mr();
        for i in 0..k {
            for j in 0..rows {
                unsafe {
                    *pa.offset((i * mr + j) as isize) =
                        *a.offset(i as isize * csa + j as isize * rsa)
                }
            }
        }
    }

    fn packed_b_len(&self, k: usize, n: usize) -> usize {
        (n + self.nr() - 1) / self.nr() * self.nr() * k
    }

    fn pack_b(&self, k: usize, n: usize, pb: *mut T, b: *const T, rsb: isize, csb: isize) {
        let nr = self.nr();
        unsafe {
            for p in 0..(n / nr) {
                self.pack_panel_b(
                    pb.offset((p * nr * k) as isize),
                    b.offset((p * nr) as isize * csb),
                    rsb,
                    csb,
                    nr,
                    k,
                )
            }
            if n % nr != 0 {
                self.pack_panel_b(
                    pb.offset((n / nr * nr * k) as isize),
                    b.offset((n / nr * nr) as isize * csb),
                    rsb,
                    csb,
                    n % nr,
                    k,
                )
            }
        }
    }

    fn pack_panel_b(&self, pb: *mut T, b: *const T, rsb: isize, csb: isize, cols: usize, k: usize) {
        let nr = self.nr();
        for i in 0..k {
            for j in 0..cols {
                unsafe {
                    *pb.offset((i * nr + j) as isize) =
                        *b.offset(j as isize * csb + i as isize * rsb)
                }
            }
        }
    }

    fn mat_mul(
        &self,
        m: usize,
        k: usize,
        n: usize,
        a: *const T,
        rsa: isize,
        csa: isize,
        b: *const T,
        rsb: isize,
        csb: isize,
        c: *mut T,
        rsc: isize,
        csc: isize,
    ) {
        let mr = self.mr();
        let nr = self.nr();
        let mut pa = vec![T::zero(); mr * k];
        let mut pb = vec![T::zero(); self.packed_b_len(k, n)];
        self.pack_b(k, n, pb.as_mut_ptr(), b, rsb, csb);
        let mut tmpc = vec![T::zero(); mr * nr];
        unsafe {
            for ia in 0..m / mr {
                self.pack_panel_a(
                    pa.as_mut_ptr(),
                    a.offset((mr * ia) as isize * rsa),
                    rsa,
                    csa,
                    mr,
                    k,
                );
                for ib in 0..n / nr {
                    self.kernel(
                        k,
                        pa.as_ptr(),
                        pb.as_ptr().offset((ib * k * nr) as isize),
                        c.offset((mr * ia) as isize * rsc + (nr * ib) as isize * csc),
                        rsc as usize, // FIXME
                    );
                }
                if n % nr != 0 {
                    self.kernel(
                        k,
                        pa.as_ptr(),
                        pb.as_ptr().offset((n / nr * k * nr) as isize),
                        tmpc.as_mut_ptr(),
                        nr,
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
                let row = m - m % mr;
                self.pack_panel_a(
                    pa.as_mut_ptr(),
                    a.offset(row as isize * rsa),
                    rsa,
                    csa,
                    m % mr,
                    k,
                );
                for ib in 0..n / nr {
                    self.kernel(
                        k,
                        pa.as_ptr(),
                        pb.as_ptr().offset((ib * nr * k) as isize),
                        tmpc.as_mut_ptr(),
                        nr,
                    );
                    for y in 0..(m % mr) {
                        for x in 0..nr {
                            *c.offset((y + row) as isize * rsc + (x + ib * nr) as isize * csc) =
                                tmpc[y * nr + x];
                        }
                    }
                }
                if n % nr != 0 {
                    self.kernel(
                        k,
                        pa.as_ptr(),
                        pb.as_ptr().offset((n / nr * nr * k) as isize),
                        tmpc.as_mut_ptr(),
                        nr,
                    );
                    for y in 0..(m % mr) {
                        for x in 0..(n % nr) {
                            *c.offset(
                                (y + row) as isize * rsc + (x + n / nr * nr) as isize * csc,
                            ) = tmpc[y * nr + x];
                        }
                    }
                }
            }
        }
    }

    fn mat_mul_prepacked(
        &self,
        m: usize,
        k: usize,
        n: usize,
        pa: *const T,
        pb: *const T,
        c: *mut T,
        rsc: isize,
        csc: isize,
    ) {
        let mr = self.mr();
        let nr = self.nr();
        let mut tmpc = vec![T::zero(); mr * nr];
        unsafe {
            for ia in 0..m / mr {
                for ib in 0..n / nr {
                    self.kernel(
                        k,
                        pa.offset((ia * k * mr) as isize),
                        pb.offset((ib * k * nr) as isize),
                        c.offset((mr * ia) as isize * rsc + (nr * ib) as isize * csc),
                        rsc as usize, // FIXME
                    );
                }
                if n % nr != 0 {
                    self.kernel(
                        k,
                        pa.offset((ia * k * mr) as isize),
                        pb.offset((n / nr * k * nr) as isize),
                        tmpc.as_mut_ptr(),
                        nr,
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
                    self.kernel(
                        k,
                        pa.offset((m / mr * mr * k) as isize),
                        pb.offset((ib * nr * k) as isize),
                        tmpc.as_mut_ptr(),
                        nr,
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
                    self.kernel(
                        k,
                        pa.offset((m / mr * mr * k) as isize),
                        pb.offset((n / nr * nr * k) as isize),
                        tmpc.as_mut_ptr(),
                        nr,
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

/*
pub fn mat_mul_f32(
    m: usize,
    k: usize,
    n: usize,
    a: *const f32,
    rsa: isize,
    csa: isize,
    b: *const f32,
    rsb: isize,
    csb: isize,
    c: *mut f32,
    rsc: isize,
    csc: isize,
) {
    /*
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("fma") {
            if a as usize % 32 != 0 {
                warn!("FMA kernel not use because a is not 32-byte aligned");
                return two_loops::two_loops::<crate::generic::MatMul>(
                    m, k, n, a, rsa, csa, b, rsb, csb, c, rsc, csc,
                );
            }
            return two_loops::two_loops::<x86_64_fma::KerFma16x6>(
                m, k, n, a, rsa, csa, b, rsb, csb, c, rsc, csc,
            );
        }
    }
    */
    two_loops::two_loops::<crate::generic::SMatMul, f32>(
        m, k, n, a, rsa, csa, b, rsb, csb, c, rsc, csc,
    )
}
*/

#[cfg(test)]
mod test {
    use super::*;
    use proptest::prelude::*;
    use proptest::*;

    fn strat(k_fact: usize) -> BoxedStrategy<(usize, usize, usize, Vec<f32>, Vec<f32>)> {
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

    proptest! {
        #[test]
        fn mat_mul_e2e((m, k, n, ref a, ref b) in strat(1)) {
            use crate::generic::SMatMul as K;

            let mut expect = vec!(0.0; m*n);
            let mut found = vec!(0.0; m*n);
            unsafe {
                K.mat_mul(m, k, n,
                            a.as_ptr(), k as isize, 1,
                            b.as_ptr(), n as isize, 1,
                            found.as_mut_ptr(), n as isize, 1);
                ::matrixmultiply::sgemm(  m, k, n,
                                        1.0, a.as_ptr(), k as _, 1,
                                        b.as_ptr(), n as _, 1,
                                        0.0, expect.as_mut_ptr(), n as _, 1);
            }
            prop_assert_eq!(expect, found);
        }
    }

    proptest! {
        #[test]
        fn mat_mul_prepacked((m, k, n, ref a, ref b) in strat(1)) {
            use crate::generic::SMatMul as K;

            let mut packed_a = vec!(0.0f32; K.packed_a_len(m,k));
            K.pack_a(m, k, packed_a.as_mut_ptr(), a.as_ptr(), k as isize, 1);

            let mut packed_b = vec!(0.0f32; K.packed_b_len(k,n));
            K.pack_b(k, n, packed_b.as_mut_ptr(), b.as_ptr(), n as isize, 1);

            let mut expect = vec!(0.0f32; m*n);
            let mut found = vec!(0.0f32; m*n);
            unsafe {
                K.mat_mul_prepacked(m, k, n, packed_a.as_ptr(), packed_b.as_ptr(), found.as_mut_ptr(), n as isize, 1);
                ::matrixmultiply::sgemm(  m, k, n,
                                        1.0, a.as_ptr(), k as _, 1,
                                        b.as_ptr(), n as _, 1,
                                        0.0, expect.as_mut_ptr(), n as _, 1);
            }
            prop_assert_eq!(expect, found);
        }
    }

    #[test]
    fn t_1x1x1() {
        use crate::generic::SMatMul as K;
        let a = vec![2.0];
        let b = vec![-1.0];
        let mut c = vec![0.0];
        K.mat_mul(
            1,
            1,
            1,
            a.as_ptr(),
            1,
            1,
            b.as_ptr(),
            1,
            1,
            c.as_mut_ptr(),
            1,
            1,
        );
        assert_eq!(c, &[-2.0]);
    }
}
