use crate::frame::tiling_kernel::*;

extern "C" {
    #[no_mangle]
    fn fma_stile16x6(op: *const TileOpSpec<f32>) -> isize;
}

#[derive(Copy, Clone, Debug)]
pub struct STile16x6;

impl TilingKer<f32> for STile16x6 {
    #[inline(always)]
    fn name() -> &'static str {
        "fma"
    }
    #[inline(always)]
    fn mr() -> usize {
        16
    }
    #[inline(always)]
    fn nr() -> usize {
        6
    }
    fn alignment_bytes_packed_a() -> usize {
        32
    }
    fn alignment_bytes_packed_b() -> usize {
        4
    }
    #[inline(never)]
    fn kernel(spec: &TileOpSpec<f32>) -> isize {
        unsafe { fma_stile16x6(spec) }
    }
}

#[cfg(test)]
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"),))]
mod test {
    tile_frame_tests!(is_x86_feature_detected!("fma"), crate::x86_64_fma::tile::STile16x6);
    tile_kernel_tests!(is_x86_feature_detected!("fma"), crate::x86_64_fma::tile::STile16x6, f32);
}
