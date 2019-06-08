use crate::frame;
use libc::size_t;
use libc::ssize_t;

extern "C" {
    #[no_mangle]
    fn arm64simd_mm_s8x8(
        k: size_t,
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        rsc: size_t,
        csc: size_t,
    );

    fn arm64simd_conv_s8x8(
        k: size_t,
        a: *const f32,
        b_tops: *const *const f32,
        b_offsets: *const ssize_t,
        c: *mut f32,
        rsc: size_t,
        csc: size_t,
    );
}

#[derive(Copy, Clone, Debug)]
pub struct SMatMul8x8;

impl frame::matmul::PackedMatMulKer<f32> for SMatMul8x8 {
    #[inline(always)]
    fn name() -> &'static str {
        "arm64simd"
    }
    #[inline(always)]
    fn mr() -> usize {
        8
    }
    #[inline(always)]
    fn nr() -> usize {
        8
    }
    fn alignment_bytes_a() -> usize {
        16
    }
    fn alignment_bytes_b() -> usize {
        16
    }
    #[inline(never)]
    fn kernel(k: usize, a: *const f32, b: *const f32, c: *mut f32, rsc: usize, csc: usize) {
        unsafe { arm64simd_mm_s8x8(k, a, b, c, rsc, csc) }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct SConv8x8;

impl frame::conv::ConvKer<f32> for SConv8x8 {
    #[inline(always)]
    fn name() -> &'static str {
        "arm64simd"
    }
    #[inline(always)]
    fn mr() -> usize {
        8
    }
    #[inline(always)]
    fn nr() -> usize {
        8
    }
    fn alignment_bytes_a() -> usize {
        16
    }
    fn alignment_bytes_b() -> usize {
        4
    }
    #[inline(never)]
    fn kernel(
        k: usize,
        a: *const f32,
        b_tops: *const *const f32,
        b_offsets: *const isize,
        c: *mut f32,
        rsc: usize,
        csc: usize,
    ) {
        unsafe { arm64simd_conv_s8x8(k, a, b_tops, b_offsets, c, rsc, csc) }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::frame::conv::test::*;
    use crate::frame::matmul::test::*;
    use crate::frame::PackedConv;
    use crate::frame::PackedMatMul;
    use proptest::*;

    proptest! {
        #[test]
        fn ker_mat_mul((k, ref a, ref b) in strat_ker_mat_mul::<SMatMul8x8>()) {
            test_ker_mat_mul::<SMatMul8x8>(k, a, b)?
        }

        #[test]
        fn mat_mul_prepacked((m, k, n, ref a, ref b) in strat_mat_mul()) {
            let mm = PackedMatMul::<SMatMul8x8, f32>::new(m, k, n);
            test_mat_mul_prep_f32(mm, m, k, n, a, b)?
        }
    }

    proptest! {
        #[test]
        fn conv(pb in strat_conv_1d()) {
            let (kernel_offsets, data_offsets) = pb.offsets();
            let conv = PackedConv::<SConv8x8, f32>::new(pb.co, kernel_offsets, data_offsets);
            let found = pb.run(&conv);
            let expected = pb.expected();
            prop_assert_eq!(found, expected)
        }
    }
}
