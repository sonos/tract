#![allow(dead_code)]

use super::Kernel;

pub struct KerFma8x8;

impl Kernel for KerFma8x8 {
    #[inline(always)]
    fn mr() -> usize {
        8
    }
    #[inline(always)]
    fn nr() -> usize {
        8
    }
    #[inline(never)]
    fn kernel(k: usize, a: *const f32, b: *const f32, c: *mut f32, rsc: usize) {
        unsafe {
            use std::arch::x86_64::*;
            let mut ab = [_mm256_setzero_ps(); 8];
            for i in 0..k {
                let ar = _mm256_load_ps(a.offset((i * 8) as isize));
                for j in 0usize..8 {
                    let br = _mm256_set1_ps(*b.offset((i * 8 + j) as isize));
                    ab[j] = _mm256_fmadd_ps(ar, br, ab[j]);
                }
            }
            for x in 0..8 {
                let mut col = [0.0; 8];
                _mm256_storeu_ps(col.as_mut_ptr(), ab[x]);
                for y in 0..8 {
                    *c.offset((y * rsc + x) as isize) = col[y];
                }
            }
        }
    }
}

pub struct KerFma16x6;

impl KerFma16x6 {
    #[target_feature(enable = "fma")]
    unsafe fn fma(k: usize, a: *const f32, b: *const f32, c: *mut f32, rsc: usize) {
        use std::arch::x86_64::*;
        let mut ab1 = [_mm256_setzero_ps(); 6];
        let mut ab2 = [_mm256_setzero_ps(); 6];
        for i in 0..k {
            let ar1 = _mm256_load_ps(a.offset((i * 16) as isize));
            let ar2 = _mm256_load_ps(a.offset((i * 16 + 8) as isize));
            for j in 0usize..6 {
                let br = _mm256_set1_ps(*b.offset((i * 6 + j) as isize));
                if j == 0 {
                }
                ab1[j] = _mm256_fmadd_ps(ar1, br, ab1[j]);
                if j == 0 {
                }
                ab2[j] = _mm256_fmadd_ps(ar2, br, ab2[j]);
            }
        }
        for x in 0..6 {
            let mut col = [0.0; 16];
            _mm256_store_ps(col.as_mut_ptr(), ab1[x]);
            _mm256_store_ps(col.as_mut_ptr().offset(8), ab2[x]);
            for y in 0..16 {
                *c.offset((y * rsc + x) as isize) = col[y];
            }
        }
    }
}

impl Kernel for KerFma16x6 {
    #[inline(always)]
    fn mr() -> usize {
        16
    }
    #[inline(always)]
    fn nr() -> usize {
        6
    }
    #[inline(always)]
    fn kernel(k: usize, a: *const f32, b: *const f32, c: *mut f32, rsc: usize) {
        unsafe { Self::fma(k, a, b, c, rsc) }
    }
}

pub struct KerFma16x7;

impl Kernel for KerFma16x7 {
    #[inline(always)]
    fn mr() -> usize {
        16
    }
    #[inline(always)]
    fn nr() -> usize {
        7
    }
    #[inline(never)]
    fn kernel(k: usize, a: *const f32, b: *const f32, c: *mut f32, rsc: usize) {
        unsafe {
            use std::arch::x86_64::*;
            let mut ab1 = [_mm256_setzero_ps(); 7];
            let mut ab2 = [_mm256_setzero_ps(); 7];
            for i in 0..k {
                let ar1 = _mm256_load_ps(a.offset((i * 16) as isize));
                let ar2 = _mm256_load_ps(a.offset((i * 16 + 8) as isize));
                for j in 0usize..7 {
                    let br = _mm256_set1_ps(*b.offset((i * 7 + j) as isize));
                    ab1[j] = _mm256_fmadd_ps(ar1, br, ab1[j]);
                    ab2[j] = _mm256_fmadd_ps(ar2, br, ab2[j]);
                }
            }
            for x in 0..7 {
                let mut col = [0.0; 16];
                _mm256_storeu_ps(col.as_mut_ptr(), ab1[x]);
                _mm256_storeu_ps(col.as_mut_ptr().offset(8), ab2[x]);
                for y in 0..16 {
                    *c.offset((y * rsc + x) as isize) = col[y];
                }
            }
        }
    }
}

#[cfg(test)]
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"),
  target_feature = "fma"))]
mod test {
    use super::*;

    #[test]
    fn t_kernel_16x6() {
        let mut a = vec!(0.0; 16);
        a[0] = 2.0;
        let mut b = vec!(0.0; 6);
        b[0] = -1.0;
        let mut c = vec!(0.0; 16*6);
        KerFma16x6::kernel(1, a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), 6);
        let mut exp = vec!(0.0; 16*6);
        exp[0] = -2.0;
        assert_eq!(c, exp);
    }
}
