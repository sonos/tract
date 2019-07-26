use crate::frame;
use crate::frame::tiling::LinearSpec::*;
use crate::frame::tiling::TileStorageSpec::*;
use crate::frame::tiling::*;

#[repr(align(32))]
struct SixteenAlignedF32([f32; 16]);

#[derive(Copy, Clone, Debug)]
pub struct STile16x6;

#[target_feature(enable = "fma")]
unsafe fn kernel(spec: &TileOpSpec<f32>) -> isize {
    use std::arch::x86_64::*;
    let mut ab1 = [_mm256_setzero_ps(); 6];
    let mut ab2 = [_mm256_setzero_ps(); 6];
    match (*spec.a, *spec.b, *spec.linear) {
        (Packed { ptr: a }, Packed { ptr: b }, Mul { k }) => {
            for i in 0..k {
                let ar1 = _mm256_load_ps(a.offset((i * 16) as isize));
                let ar2 = _mm256_load_ps(a.offset((i * 16 + 8) as isize));
                for j in 0usize..6 {
                    let br = _mm256_set1_ps(*b.offset((i * 6 + j) as isize));
                    ab1[j] = _mm256_fmadd_ps(ar1, br, ab1[j]);
                    ab2[j] = _mm256_fmadd_ps(ar2, br, ab2[j]);
                }
            }
        }
        (Packed { ptr: a }, OffsetsAndPtrs { row_byte_offsets, col_ptrs }, Mul { k }) => {
            for i in 0..k {
                let down_offset = *row_byte_offsets.offset(i as isize) >> 2;
                let ar1 = _mm256_load_ps(a.offset((i * 16) as isize));
                let ar2 = _mm256_load_ps(a.offset((i * 16 + 8) as isize));
                for j in 0usize..6 {
                    let bp = (*col_ptrs.offset(j as isize)).offset(down_offset);
                    let br = _mm256_set1_ps(*bp);
                    ab1[j] = _mm256_fmadd_ps(ar1, br, ab1[j]);
                    ab2[j] = _mm256_fmadd_ps(ar2, br, ab2[j]);
                }
            }
        }
        _ => return 1,
    }
    let mut pnl = spec.non_linear;
    loop {
        if pnl.is_null() || *pnl == NonLinearSpec::Done {
            break;
        }
        pnl = pnl.add(1);
    }
    match *spec.c {
        Strides { ptr: c, row_byte_stride, col_byte_stride } => {
            let rsc = (row_byte_stride >> 2) as isize;
            let csc = (col_byte_stride >> 2) as isize;
            for x in 0..6 {
                let mut col = SixteenAlignedF32([0f32; 16]);
                _mm256_store_ps(col.0.as_mut_ptr(), ab1[x]);
                _mm256_store_ps(col.0.as_mut_ptr().offset(8), ab2[x]);
                for y in 0..16 {
                    *c.offset(y as isize * rsc + x as isize * csc) = col.0[y];
                }
            }
        }
        _ => return 1,
    }
    return 0
}

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
        unsafe { kernel(spec) }
    }
}

#[cfg(test)]
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"),))]
mod test {
    use super::*;

    tile_tests!(is_x86_feature_detected!("fma"), STile16x6);
}
