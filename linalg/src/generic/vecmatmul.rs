use crate::frame;

#[derive(Copy, Clone, Debug)]
pub struct SVecMatMul8;

impl frame::vecmatmul::VecMatMulKer<f32> for SVecMatMul8 {
    #[inline(always)]
    fn name() -> &'static str {
        "generic"
    }
    fn nr() -> usize {
        8
    }
    #[inline(always)]
    fn alignment_bytes_a() -> usize {
        4
    }
    #[inline(always)]
    fn alignment_bytes_b() -> usize {
        4
    }
    #[inline(always)]
    fn kernel(k: usize, a: *const f32, b: *const f32, c: *mut f32, sy: usize) {
        unsafe {
            let mut ab = [0.0f32; 8];
            for i in 0..k {
                let a = *a.offset(i as isize);
                let b = std::slice::from_raw_parts(b.offset(8 * i as isize), 8);
                for j in 0..8 {
                    ab[j] += a * b[j];
                }
            }
            for i in 0..8 {
                *c.offset((i * sy) as isize) = ab[i];
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::frame::vecmatmul::test::*;
    use crate::frame::PackedVecMatMul;
    use proptest::*;

    proptest! {
        #[test]
        fn vec_mat_mul_prepacked((k, n, ref a, ref b) in strat_vec_mat_mul()) {
            let mm = PackedVecMatMul::<SVecMatMul8, f32>::new(k, n);
            test_vec_mat_mul_prep_f32(mm, k, n, a, b)?
        }
    }

    #[test]
    fn test_1() {
        let mm = PackedVecMatMul::<SVecMatMul8, f32>::new(1, 5);
        test_vec_mat_mul_prep_f32(mm, 1, 5, &[1.0], &[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    }
}
