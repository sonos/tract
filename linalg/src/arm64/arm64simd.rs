use crate::frame::tiling_kernel::*;

extern "C" {
    #[no_mangle]
    fn arm64simd_stile8x8(op: *const TileOpSpec<f32>) -> isize;
}

#[derive(Copy, Clone, Debug)]
pub struct STile8x8;

impl TilingKer<f32> for STile8x8 {
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
    fn alignment_bytes_packed_a() -> usize {
        16
    }
    fn alignment_bytes_packed_b() -> usize {
        16
    }
    #[inline(never)]
    fn kernel(op: &TileOpSpec<f32>) -> isize {
        unsafe { arm64simd_stile8x8(op) }
    }
}

#[cfg(test)]
mod test {
    tile_kernel_tests!(true, crate::arm64::arm64simd::STile8x8, f32);
    tile_frame_tests!(true, crate::arm64::arm64simd::STile8x8);
}
