use crate::frame::tiling_kernel::*;

extern "C" {
    #[no_mangle]
    fn neon_stile8x4(op: *const TileOpSpec<f32>) -> isize;
}

#[derive(Copy, Clone, Debug)]
pub struct STile8x4;

impl TilingKer<f32> for STile8x4 {
    #[inline(always)]
    fn name() -> &'static str {
        "neon"
    }
    #[inline(always)]
    fn mr() -> usize {
        8
    }
    #[inline(always)]
    fn nr() -> usize {
        4
    }
    fn alignment_bytes_packed_a() -> usize {
        4
    }
    fn alignment_bytes_packed_b() -> usize {
        4
    }
    #[inline(never)]
    fn kernel(spec: &TileOpSpec<f32>) -> isize {
        unsafe { neon_stile8x4(spec) }
    }
}

#[cfg(test)]
mod test {
    tile_kernel_tests!(crate::arm32::has_neon(), crate::arm32::armv7neon::STile8x4, f32);
    tile_frame_tests!(crate::arm32::has_neon(), crate::arm32::armv7neon::STile8x4);
}
