// AVX-512 f16 element-wise activations. Each kernel chunks f16 -> 64-byte-aligned
// f32 scratch via vcvtph2ps (cvt_f16_to_f32 below), runs the matching f32
// AVX-512 kernel (or the avx512_sigmoid_f32 / avx512_tanh_f32 wrappers from
// x86_64_fma.rs), and converts back to f16 via vcvtps2ph (cvt_f32_to_f16).
// Conversion is driven through std::arch intrinsics directly because the
// scalar f16::to_f32 / f16::from_f32 loops are not autovectorized by
// rustc + LLVM (branches / call overhead in the half crate's methods),
// which leaves a naive port stuck around 7 Melem/s.
//
// The f32 AVX-512 activation kernels assume 64-byte aligned input (alignment
// bytes = nr * 4 for nr >= 16). The local scratch is wrapped in
// #[repr(C, align(64))] so the contained [f32; 256] sits at a 64-byte boundary.
//
// Validated against the generic f16 reference (HHardSwish8 / HLeakyRelu8 /
// HSigmoid8 / HTanh8 / HSiLU8 / HGelu8) via the existing *_frame_tests!
// macros at SuperApproximate tolerance, which covers the precision delta
// between scalar f16 arithmetic and f32-internal computation.

use tract_data::internal::f16;

#[repr(C, align(64))]
struct AlignedScratch([f32; 256]);

impl AlignedScratch {
    fn new() -> Self {
        Self([0f32; 256])
    }
}

const CHUNK: usize = 256;

// Vectorized f16 <-> f32 helpers using vcvtph2ps / vcvtps2ph. Rustc + LLVM
// do NOT autovectorize the scalar `.to_f32()` loop (the half crate's method
// has branches / function-call overhead), so we drive the conversion with
// intrinsics directly. Both helpers process 16 lanes per iteration; the tail
// (which only fires for the 1-15 leftover lanes inside a CHUNK = 256 batch)
// falls back to scalar.
#[target_feature(enable = "avx512f")]
unsafe fn cvt_f16_to_f32(src: &[f16], dst: &mut [f32]) {
    use core::arch::x86_64::*;
    let n = src.len();
    debug_assert!(dst.len() >= n);
    let chunks = n / 16;
    unsafe {
        for k in 0..chunks {
            let m = _mm256_loadu_si256(src.as_ptr().add(k * 16) as *const __m256i);
            let z = _mm512_cvtph_ps(m);
            _mm512_storeu_ps(dst.as_mut_ptr().add(k * 16), z);
        }
        for k in (chunks * 16)..n {
            *dst.get_unchecked_mut(k) = src.get_unchecked(k).to_f32();
        }
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn cvt_f32_to_f16(src: &[f32], dst: &mut [f16]) {
    use core::arch::x86_64::*;
    let n = src.len();
    debug_assert!(dst.len() >= n);
    let chunks = n / 16;
    unsafe {
        for k in 0..chunks {
            let z = _mm512_loadu_ps(src.as_ptr().add(k * 16));
            // _MM_FROUND_TO_NEAREST_INT == 0 (round-to-nearest-even, matches f16::from_f32)
            let m = _mm512_cvtps_ph::<0>(z);
            _mm256_storeu_si256(dst.as_mut_ptr().add(k * 16) as *mut __m256i, m);
        }
        for k in (chunks * 16)..n {
            *dst.get_unchecked_mut(k) = f16::from_f32(*src.get_unchecked(k));
        }
    }
}

// hardswish_f16
ew_impl_wrap!(
    f16,
    x86_64_avx512_hardswish_f16_64n,
    64,
    32,
    (),
    #[inline(never)]
    fn run(buf: &mut [f16], _: ()) {
        debug_assert!(buf.len() % Self::nr() == 0);
        debug_assert!(buf.as_ptr() as usize % Self::alignment_bytes() == 0);
        if buf.is_empty() {
            return;
        }
        let mut scratch = AlignedScratch::new();
        let s = &mut scratch.0;
        let mut i = 0;
        while i < buf.len() {
            let n = (CHUNK).min(buf.len() - i);
            unsafe { cvt_f16_to_f32(&buf[i..i + n], &mut s[..n]) };
            super::act::x86_64_avx512_hardswish_f32_64n::run(&mut s[..n], ());
            unsafe { cvt_f32_to_f16(&s[..n], &mut buf[i..i + n]) };
            i += n;
        }
    }
);

#[cfg(test)]
pub mod test_x86_64_avx512_hardswish_f16_64n {
    use super::*;
    hardswish_frame_tests!(
        is_x86_feature_detected!("avx512f"),
        f16,
        x86_64_avx512_hardswish_f16_64n
    );
}

// leaky_relu_f16  (parameter: alpha as f16)
ew_impl_wrap!(
    f16,
    x86_64_avx512_leaky_relu_f16_64n,
    64,
    32,
    f16,
    #[inline(never)]
    fn run(buf: &mut [f16], alpha: f16) {
        debug_assert!(buf.len() % Self::nr() == 0);
        debug_assert!(buf.as_ptr() as usize % Self::alignment_bytes() == 0);
        if buf.is_empty() {
            return;
        }
        let alpha_f32 = alpha.to_f32();
        let mut scratch = AlignedScratch::new();
        let s = &mut scratch.0;
        let mut i = 0;
        while i < buf.len() {
            let n = (CHUNK).min(buf.len() - i);
            unsafe { cvt_f16_to_f32(&buf[i..i + n], &mut s[..n]) };
            super::act::x86_64_avx512_leaky_relu_f32_64n::run(&mut s[..n], alpha_f32);
            unsafe { cvt_f32_to_f16(&s[..n], &mut buf[i..i + n]) };
            i += n;
        }
    }
);

#[cfg(test)]
pub mod test_x86_64_avx512_leaky_relu_f16_64n {
    use super::*;
    leaky_relu_frame_tests!(
        is_x86_feature_detected!("avx512f"),
        f16,
        x86_64_avx512_leaky_relu_f16_64n
    );
}

// sigmoid_f16  (calls the avx512_sigmoid_f32 wrapper from x86_64_fma.rs;
// its nr() is 16 so CHUNK=256 is always a clean multiple)
ew_impl_wrap!(
    f16,
    x86_64_avx512_sigmoid_f16_16n,
    16,
    16,
    (),
    #[inline(never)]
    fn run(buf: &mut [f16], _: ()) {
        debug_assert!(buf.len() % Self::nr() == 0);
        debug_assert!(buf.as_ptr() as usize % Self::alignment_bytes() == 0);
        if buf.is_empty() {
            return;
        }
        let mut scratch = AlignedScratch::new();
        let s = &mut scratch.0;
        let mut i = 0;
        while i < buf.len() {
            let n = (CHUNK).min(buf.len() - i);
            unsafe { cvt_f16_to_f32(&buf[i..i + n], &mut s[..n]) };
            super::avx512_sigmoid_f32::run(&mut s[..n], ());
            unsafe { cvt_f32_to_f16(&s[..n], &mut buf[i..i + n]) };
            i += n;
        }
    }
);

#[cfg(test)]
pub mod test_x86_64_avx512_sigmoid_f16_16n {
    use super::*;
    sigmoid_frame_tests!(is_x86_feature_detected!("avx512f"), f16, x86_64_avx512_sigmoid_f16_16n);
}

// tanh_f16
ew_impl_wrap!(
    f16,
    x86_64_avx512_tanh_f16_16n,
    16,
    16,
    (),
    #[inline(never)]
    fn run(buf: &mut [f16], _: ()) {
        debug_assert!(buf.len() % Self::nr() == 0);
        debug_assert!(buf.as_ptr() as usize % Self::alignment_bytes() == 0);
        if buf.is_empty() {
            return;
        }
        let mut scratch = AlignedScratch::new();
        let s = &mut scratch.0;
        let mut i = 0;
        while i < buf.len() {
            let n = (CHUNK).min(buf.len() - i);
            unsafe { cvt_f16_to_f32(&buf[i..i + n], &mut s[..n]) };
            super::avx512_tanh_f32::run(&mut s[..n], ());
            unsafe { cvt_f32_to_f16(&s[..n], &mut buf[i..i + n]) };
            i += n;
        }
    }
);

#[cfg(test)]
pub mod test_x86_64_avx512_tanh_f16_16n {
    use super::*;
    tanh_frame_tests!(is_x86_feature_detected!("avx512f"), f16, x86_64_avx512_tanh_f16_16n);
}

// silu_f16: x * sigmoid(x).  Mirror the f32 silu pattern: save the input
// (in f32), run sigmoid in place on the scratch, then multiply back.
ew_impl_wrap!(
    f16,
    x86_64_avx512_silu_f16_16n,
    16,
    16,
    (),
    #[inline(never)]
    fn run(buf: &mut [f16], _: ()) {
        debug_assert!(buf.len() % Self::nr() == 0);
        debug_assert!(buf.as_ptr() as usize % Self::alignment_bytes() == 0);
        if buf.is_empty() {
            return;
        }
        let mut work = AlignedScratch::new();
        let mut save = AlignedScratch::new();
        let w = &mut work.0;
        let v = &mut save.0;
        let mut i = 0;
        while i < buf.len() {
            let n = (CHUNK).min(buf.len() - i);
            unsafe { cvt_f16_to_f32(&buf[i..i + n], &mut w[..n]) };
            v[..n].copy_from_slice(&w[..n]);
            super::avx512_sigmoid_f32::run(&mut w[..n], ());
            for j in 0..n {
                w[j] *= v[j];
            }
            unsafe { cvt_f32_to_f16(&w[..n], &mut buf[i..i + n]) };
            i += n;
        }
    }
);

#[cfg(test)]
pub mod test_x86_64_avx512_silu_f16_16n {
    use super::*;
    silu_frame_tests!(is_x86_feature_detected!("avx512f"), f16, x86_64_avx512_silu_f16_16n);
}

// Tanh-form GELU (matches tract's GeluApproximate, pow=3, see act.rs gelu_f32):
//   gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
ew_impl_wrap!(
    f16,
    x86_64_avx512_gelu_f16_16n,
    16,
    16,
    (),
    #[inline(never)]
    fn run(buf: &mut [f16], _: ()) {
        debug_assert!(buf.len() % Self::nr() == 0);
        debug_assert!(buf.as_ptr() as usize % Self::alignment_bytes() == 0);
        if buf.is_empty() {
            return;
        }
        const SQRT_2_OVER_PI: f32 = 0.7978845608028654;
        const COEF: f32 = 0.044715;
        let mut work = AlignedScratch::new();
        let mut save = AlignedScratch::new();
        let w = &mut work.0;
        let v = &mut save.0;
        let mut i = 0;
        while i < buf.len() {
            let n = (CHUNK).min(buf.len() - i);
            unsafe { cvt_f16_to_f32(&buf[i..i + n], &mut v[..n]) };
            for j in 0..n {
                let x = v[j];
                w[j] = SQRT_2_OVER_PI * (x + COEF * x * x * x);
            }
            super::avx512_tanh_f32::run(&mut w[..n], ());
            for j in 0..n {
                w[j] = 0.5 * v[j] * (1.0 + w[j]);
            }
            unsafe { cvt_f32_to_f16(&w[..n], &mut buf[i..i + n]) };
            i += n;
        }
    }
);

#[cfg(test)]
pub mod test_x86_64_avx512_gelu_f16_16n {
    use super::*;
    gelu_frame_tests!(is_x86_feature_detected!("avx512f"), f16, x86_64_avx512_gelu_f16_16n);
}
