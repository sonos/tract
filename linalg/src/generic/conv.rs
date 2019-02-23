use crate::frame;

#[derive(Copy, Clone, Debug)]
pub struct SConv4x4;

impl frame::conv::ConvKer<f32> for SConv4x4 {
    #[inline(always)]
    fn name() -> &'static str {
        "generic"
    }
    #[inline(always)]
    fn mr() -> usize {
        4
    }
    #[inline(always)]
    fn nr() -> usize {
        4
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
    fn kernel(
        k: usize,
        a: *const f32,
        b_tops: *const *const f32,
        b_down_offsets: *const isize,
        c: *mut f32,
        rsc: usize,
        csc: usize,
    ) {
        unsafe {
            let mut ab = [[0.0f32; 4]; 4];
            let pb0 = *(b_tops.offset(0));
            let pb1 = *(b_tops.offset(1));
            let pb2 = *(b_tops.offset(2));
            let pb3 = *(b_tops.offset(3));
            for i in 0..k {
                let a = std::slice::from_raw_parts(a.offset(4 * i as isize), 4);
                let offset = *b_down_offsets.offset(i as isize);
                let b0 = *(pb0.offset(offset));
                let b1 = *(pb1.offset(offset));
                let b2 = *(pb2.offset(offset));
                let b3 = *(pb3.offset(offset));
                ab[0][0] += a[0] * b0;
                ab[0][1] += a[0] * b1;
                ab[0][2] += a[0] * b2;
                ab[0][3] += a[0] * b3;
                ab[1][0] += a[1] * b0;
                ab[1][1] += a[1] * b1;
                ab[1][2] += a[1] * b2;
                ab[1][3] += a[1] * b3;
                ab[2][0] += a[2] * b0;
                ab[2][1] += a[2] * b1;
                ab[2][2] += a[2] * b2;
                ab[2][3] += a[2] * b3;
                ab[3][0] += a[3] * b0;
                ab[3][1] += a[3] * b1;
                ab[3][2] += a[3] * b2;
                ab[3][3] += a[3] * b3;
            }
            let c = std::slice::from_raw_parts_mut(c, 1 + 3 * csc + 3 * rsc);
            c[0 * csc + 0 * rsc] = ab[0][0];
            c[1 * csc + 0 * rsc] = ab[0][1];
            c[2 * csc + 0 * rsc] = ab[0][2];
            c[3 * csc + 0 * rsc] = ab[0][3];
            c[0 * csc + 1 * rsc] = ab[1][0];
            c[1 * csc + 1 * rsc] = ab[1][1];
            c[2 * csc + 1 * rsc] = ab[1][2];
            c[3 * csc + 1 * rsc] = ab[1][3];
            c[0 * csc + 2 * rsc] = ab[2][0];
            c[1 * csc + 2 * rsc] = ab[2][1];
            c[2 * csc + 2 * rsc] = ab[2][2];
            c[3 * csc + 2 * rsc] = ab[2][3];
            c[0 * csc + 3 * rsc] = ab[3][0];
            c[1 * csc + 3 * rsc] = ab[3][1];
            c[2 * csc + 3 * rsc] = ab[3][2];
            c[3 * csc + 3 * rsc] = ab[3][3];
        }
    }
}

#[cfg(test)]
mod test {
    use crate::frame::conv::test::*;
    use crate::frame::PackedConv;
    use proptest::*;

    proptest! {
        #[test]
        fn conv_prepacked((ci, co, kt, stride, dilation, ref filters, ref data) in strat_conv_1d()) {
            let kernel_field = dilation * (kt - 1) + 1;
            let t = data.len() / ci;
            let n = (t - kernel_field) / stride + 1;
            let data_offsets:Vec<isize> = (0..n).map(|i| (i * stride) as isize).collect();
            let kernel_offsets:Vec<isize> = (0..ci)
                .flat_map(move |ici| (0..kt).map(move |ikt| (ikt * dilation + ici * t) as isize))
                .collect();
            assert!(data_offsets.iter().max().unwrap() + kernel_offsets.iter().max().unwrap() <= data.len() as isize);

            let conv = PackedConv::<crate::generic::conv::SConv4x4, f32>::new(co, kernel_offsets, data_offsets);
            test_conv_1d_f32(conv, ci, co, kt, stride, dilation, filters, data)?
        }
    }
}
