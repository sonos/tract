use crate::frame;

#[repr(align(32))]
struct SixteenAlignedF32([f32; 16]);

#[derive(Copy, Clone, Debug)]
pub struct KerFma16x6;

#[target_feature(enable = "fma")]
unsafe fn fma(k: usize, a: *const f32, b: *const f32, c: *mut f32, rsc: usize, csc: usize) {
    use std::arch::x86_64::*;
    assert!(a as usize % 32 == 0);
    assert!(b as usize % 4 == 0);
    assert!(c as usize % 4 == 0);
    let mut ab1 = [_mm256_setzero_ps(); 6];
    let mut ab2 = [_mm256_setzero_ps(); 6];
    for i in 0..k {
        let ar1 = _mm256_load_ps(a.offset((i * 16) as isize));
        let ar2 = _mm256_load_ps(a.offset((i * 16 + 8) as isize));
        for j in 0usize..6 {
            let br = _mm256_set1_ps(*b.offset((i * 6 + j) as isize));
            ab1[j] = _mm256_fmadd_ps(ar1, br, ab1[j]);
            ab2[j] = _mm256_fmadd_ps(ar2, br, ab2[j]);
        }
    }
    for x in 0..6 {
        let mut col = SixteenAlignedF32([0f32; 16]);
        _mm256_store_ps(col.0.as_mut_ptr(), ab1[x]);
        _mm256_store_ps(col.0.as_mut_ptr().offset(8), ab2[x]);
        for y in 0..16 {
            *c.offset((y * rsc + x * csc) as isize) = col.0[y];
        }
    }
}

impl frame::matmul::PackedMatMulKer<f32> for KerFma16x6 {
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
    fn alignment_bytes_a() -> usize {
        32
    }
    fn alignment_bytes_b() -> usize {
        4
    }
    #[inline(always)]
    fn kernel(k: usize, a: *const f32, b: *const f32, c: *mut f32, rsc: usize, csc: usize) {
        unsafe { fma(k, a, b, c, rsc, csc) }
    }
}

#[cfg(test)]
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
))]
mod test {
    use super::*;
    use crate::frame::matmul::test::*;
    use crate::frame::PackedMatMul;
    use proptest::*;

    proptest! {
        #[test]
        fn mat_mul_prepacked((m, k, n, ref a, ref b) in strat_mat_mul()) {
            if !is_x86_feature_detected!("fma") {
                return Ok(())
            }
            let mm = PackedMatMul::<KerFma16x6, f32>::new(m, k, n);
            test_mat_mul_prep_f32(mm, m, k, n, a, b)?
        }
    }
}
