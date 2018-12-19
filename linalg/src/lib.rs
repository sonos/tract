use log::warn;

pub mod fallback;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod haswell;
mod two_loops;

pub use self::two_loops::two_loops;

pub trait Kernel {
    #[inline(always)]
    fn kernel(k: usize, a: *const f32, b: *const f32, c: *mut f32, rsc: usize);
    #[inline(always)]
    fn mr() -> usize;
    #[inline(always)]
    fn nr() -> usize;
}

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
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("fma") {
            if a as usize % 32 != 0 {
                warn!("FMA kernel not use because a is not 32-byte aligned");
                return two_loops::two_loops::<fallback::Fallback>(
                    m, k, n, a, rsa, csa, b, rsb, csb, c, rsc, csc,
                );
            }
            if b as usize % 32 != 0 {
                warn!("FMA kernel not use because b is not 32-byte aligned");
                return two_loops::two_loops::<fallback::Fallback>(
                    m, k, n, a, rsa, csa, b, rsb, csb, c, rsc, csc,
                );
            }
            if c as usize % 32 != 0 {
                warn!("FMA kernel not use because c is not 32-byte aligned");
                return two_loops::two_loops::<fallback::Fallback>(
                    m, k, n, a, rsa, csa, b, rsb, csb, c, rsc, csc,
                );
            }
            return two_loops::two_loops::<haswell::KerFma16x6>(
                m, k, n, a, rsa, csa, b, rsb, csb, c, rsc, csc,
            );
        }
    }
    two_loops::two_loops::<fallback::Fallback>(m, k, n, a, rsa, csa, b, rsb, csb, c, rsc, csc)
}

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
        fn mat_mul((m, k, n, ref a, ref b) in strat(1)) {
            let mut expect = vec!(0.0f32; m*n);
            let mut found = vec!(0.0f32; m*n);
            unsafe {
                mat_mul_f32(m, k, n,
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
            use super::fallback::Fallback as K;
            let mut packed_a = vec!(0.0f32; two_loops::packed_a_panels(K::mr(), m) * K::mr() * k);
            two_loops::pack_a(packed_a.as_mut_ptr(), a.as_ptr(), k as isize, 1, K::mr(), m, k);

            let mut packed_b = vec!(0.0f32; two_loops::packed_b_panels(K::nr(), n) * K::nr() * k);
            two_loops::pack_b(packed_b.as_mut_ptr(), b.as_ptr(), n as isize, 1, K::nr(), n, k);

            let mut expect = vec!(0.0f32; m*n);
            let mut found = vec!(0.0f32; m*n);
            unsafe {
                two_loops::two_loops_prepacked::<K>(m, k, n, packed_a.as_ptr(), packed_b.as_ptr(), found.as_mut_ptr(), n as isize, 1);
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
        let a = vec![2.0];
        let b = vec![-1.0];
        let mut c = vec![0.0];
        mat_mul_f32(
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
