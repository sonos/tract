pub mod fallback;
pub mod haswell;
mod two_loops;

pub use self::two_loops::two_loops;

use ndarray::*;

pub trait Kernel {
    #[inline(always)]
    fn kernel(k: usize, a: *const f32, b: *const f32, c: *mut f32, rsc: usize);
    #[inline(always)]
    fn mr() -> usize;
    #[inline(always)]
    fn nr() -> usize;
}

pub fn mat_mul_ndarray_f32(a: &ArrayView2<f32>, b: &ArrayView2<f32>, c: &mut ArrayViewMut2<f32>) {
    assert_eq!(a.rows(), c.rows());
    assert_eq!(b.rows(), a.cols());
    assert_eq!(b.cols(), c.cols());
    mat_mul_f32(
        a.rows(),
        a.cols(),
        b.cols(),
        a.as_ptr() as *const f32,
        a.strides()[0],
        a.strides()[1],
        b.as_ptr() as *const f32,
        b.strides()[0],
        b.strides()[1],
        c.as_mut_ptr() as *mut f32,
        c.strides()[0],
        c.strides()[1],
    );
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
    if is_x86_feature_detected!("fma") {
        two_loops::two_loops::<haswell::KerFma16x6>(m, k, n, a, rsa, csa, b, rsb, csb, c, rsc, csc)
    } else {
        two_loops::two_loops::<fallback::Fallback>(m, k, n, a, rsa, csa, b, rsb, csb, c, rsc, csc)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use proptest::prelude::*;

    fn strat(k_fact: usize) -> BoxedStrategy<(usize, usize, usize, Vec<f32>, Vec<f32>)> {
        (0usize..35, 0usize..35, 0usize..35)
            .prop_flat_map(move |(m, n, k)| {
                (
                    Just(m),
                    Just(n),
                    Just(k * k_fact),
                    proptest::collection::vec((-10..10).prop_map(|a| a as f32), m * k * k_fact),
                    proptest::collection::vec((-10..10).prop_map(|a| a as f32), n * k * k_fact),
                )
            })
            .boxed()
    }

    proptest! {
        #[test]
        fn against_matrixmultiply((m,n,k, ref a, ref b) in strat(1)) {
            let mut expect = vec!(0.0f32; m*n);
            let mut found = vec!(0.0f32; m*n);
            unsafe {
                ::matrixmultiply::sgemm(  m, k, n,
                                        1.0, a.as_ptr(), k as _, 1,
                                        b.as_ptr(), n as _, 1,
                                        0.0, expect.as_mut_ptr(), n as _, 1);
                mat_mul_f32(m, k, n,
                            a.as_ptr(), k as isize, 1,
                            b.as_ptr(), n as isize, 1,
                            found.as_mut_ptr(), n as isize, 1);
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
