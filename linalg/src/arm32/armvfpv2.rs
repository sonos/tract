use crate::frame;
use libc::size_t;
use libc::ssize_t;

extern "C" {
    fn arm_vfpv2_mm_s4x4(
        k: size_t,
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        rsc: size_t,
        csc: size_t,
    );

    fn arm_vfpv2_conv_s4x4(
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
pub struct SMatMul4x4;

impl frame::matmul::PackedMatMulKer<f32> for SMatMul4x4 {
    #[inline(always)]
    fn name() -> &'static str {
        "vfpv2"
    }
    #[inline(always)]
    fn mr() -> usize {
        4
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
        unsafe { arm_vfpv2_mm_s4x4(k, a, b, c, rsc, csc) }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct SConv4x4;

impl frame::conv::ConvKer<f32> for SConv4x4 {
    #[inline(always)]
    fn name() -> &'static str {
        "vfpv2"
    }
    #[inline(always)]
    fn mr() -> usize {
        4
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
    fn kernel(k: usize, a: *const f32, b_tops: *const *const f32, b_offsets: *const isize, c: *mut f32, rsc: usize, csc: usize) {
        unsafe { arm_vfpv2_conv_s4x4(k, a, b_tops, b_offsets, c, rsc, csc) }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::frame::*;
    use crate::frame::matmul::test::*;
    use crate::frame::conv::test::*;
    use proptest::*;

    proptest! {
        #[test]
        fn ker_mat_mul((k, ref a, ref b) in strat_ker_mat_mul::<SMatMul4x4>()) {
            test_ker_mat_mul::<SMatMul4x4>(k, a, b)?
        }

        #[test]
        fn mat_mul_prepacked((m, k, n, ref a, ref b) in strat_mat_mul()) {
            let mm = PackedMatMul::<SMatMul4x4, f32>::new(m, k, n);
            test_mat_mul_prep_f32(mm, m, k, n, a, b)?
        }
    }

    proptest! {
        #[test]
        fn conv(pb in strat_conv_1d()) {
            let (kernel_offsets, data_offsets) = pb.offsets();
            let conv = PackedConv::<SConv4x4, f32>::new(pb.co, kernel_offsets, data_offsets);
            let found = pb.run(&conv);
            let expected = pb.expected();
            prop_assert_eq!(found, expected)
        }
    }

    #[test]
    fn conv_4x4() {
        for t in 0..4 {
            for c in 0..4 {
                let filters = (0usize..4).map(|i| (i==c) as usize as f32).collect();
                let data = (0usize..4).map(|i| (i==t) as usize as f32).collect();
                let pb = ConvProblem { ci: 1, co: 4, kt: 1, stride: 1, dilation: 1, filters, data};
                let (kernel_offsets, data_offsets) = pb.offsets();
                let conv = PackedConv::<SConv4x4, f32>::new(pb.co, kernel_offsets, data_offsets);
                let found = pb.run(&conv);
                assert_eq!(found, pb.expected());
            }
        }
    }

    #[test]
    fn conv_1() {
        let pb = ConvProblem { ci: 1, co: 1, kt: 1, stride: 1, dilation: 1, filters: [0.0].to_vec(), data: [-7.0, -5.0, -9.0, -3.0].to_vec() };
        let (kernel_offsets, data_offsets) = pb.offsets();
        let conv = PackedConv::<SConv4x4, f32>::new(pb.co, kernel_offsets, data_offsets);
        let expected = pb.expected();
        for _ in 1..100000 {
            let found = pb.run(&conv);
            assert_eq!(found, expected)
        }
    }

    #[test]
    fn conv_2() {
        let pb = ConvProblem { ci: 1, co: 1, kt: 1, stride: 1, dilation: 1, filters: [0.0].to_vec(), data: [-7.0, 7.0, 8.0, -10.0].to_vec() };
        let (kernel_offsets, data_offsets) = pb.offsets();
        let conv = PackedConv::<SConv4x4, f32>::new(pb.co, kernel_offsets, data_offsets);
        let found = pb.run(&conv);
        let expected = pb.expected();
        assert_eq!(found, expected)
    }
}
