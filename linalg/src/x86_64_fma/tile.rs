use crate::frame;
use crate::frame::tiling::*;

extern "C" {
    #[no_mangle]
    fn fma_stile16x6(op: *const TileOpSpec<f32>) -> isize;
}

#[derive(Copy, Clone, Debug)]
pub struct STile16x6;

impl frame::tiling::TilingKer<f32> for STile16x6 {
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
    use super::*;

    tile_tests!(is_x86_feature_detected!("fma"), STile16x6);

    #[test]
    fn mat_mul_1() {
        test_mat_mul_prep_f32::<STile16x6>(1, 1, 1, &[1f32], &[1f32]).unwrap();
    }
}
