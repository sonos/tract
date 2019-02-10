use crate::frame;
use libc::size_t;

extern "C" {
    fn armv7neon_mm_s8x4(
        k: size_t,
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        rsc: size_t,
        csc: size_t,
    );
}

#[derive(Copy, Clone, Debug)]
pub struct SMatMul8x4;

impl frame::matmul::PackedMatMulKer<f32> for SMatMul8x4 {
    #[inline(always)]
    fn name() -> &'static str {
        "armv7neon"
    }
    #[inline(always)]
    fn mr() -> usize {
        8
    }
    #[inline(always)]
    fn nr() -> usize {
        4
    }
    fn alignment_bytes_a() -> usize {
        4
    }
    fn alignment_bytes_b() -> usize {
        4
    }
    #[inline(never)]
    fn kernel(k: usize, a: *const f32, b: *const f32, c: *mut f32, rsc: usize, csc: usize) {
        unsafe { armv7neon_mm_s8x4(k, a, b, c, rsc, csc) }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::arm32::has_neon;
    use crate::frame::matmul::test::*;
    use crate::frame::PackedMatMul;
    use proptest::*;

    proptest! {
        #[test]
        fn ker_mat_mul((k, ref a, ref b) in strat_ker_mat_mul::<SMatMul8x4>()) {
            if !has_neon() {
                return Ok(())
            }
            test_ker_mat_mul::<SMatMul8x4>(k, a, b)?
        }

        #[test]
        fn mat_mul_prepacked((m, k, n, ref a, ref b) in strat_mat_mul()) {
            if !has_neon() {
                return Ok(())
            }
            let mm = PackedMatMul::<SMatMul8x4, f32>::new(m, k, n);
            test_mat_mul_prep_f32(mm, m, k, n, a, b)?
        }
    }
}
