// NEON (aarch64, 128-bit, 4 f32 lanes) fused row-wise RmsNorm.
//
// Mirrors the AVX-512 kernel structure from `x86_64_fma/rms_norm.rs`:
//
//   Pass 1 (sum of squares):  acc += x² over 4 v-registers (4 lanes each →
//                             16 f32 / iter), horizontal reduce to a scalar,
//                             then rsqrt(mean + eps) in scalar.
//   Pass 2 (multiply-back):   broadcast inv_std into v0, multiply each
//                             4-v-register chunk in place.
//   Scalar tail handles the (len % 16 != 0) remainder.
//
// Drops into `Ops::rms_norm_f32` (added by the parent PR) — the core-side
// dispatcher in `core::ops::nn::RmsNorm::eval` is already arch-neutral and
// will pick this up automatically.

use tract_data::internal::f16;

#[target_feature(enable = "neon")]
unsafe fn rms_norm_f32_inner(buf: &mut [f32], eps: f32) {
    use std::arch::aarch64::*;
    let n = buf.len();
    let chunks = n / 16;
    let tail_start = chunks * 16;
    let ptr = buf.as_mut_ptr();

    // --- Pass 1: sum of squares ---
    let mut sum_sq: f32 = 0.0;
    if chunks > 0 {
        let p = ptr;
        let c = chunks;
        let mut sum_v: float32x4_t = vdupq_n_f32(0.0);
        unsafe {
            std::arch::asm!("
                movi v1.4s, 0
                movi v2.4s, 0
                movi v3.4s, 0
                2:
                    ld1 {{v4.4s, v5.4s, v6.4s, v7.4s}}, [{p}], 64
                    fmla v0.4s, v4.4s, v4.4s
                    fmla v1.4s, v5.4s, v5.4s
                    fmla v2.4s, v6.4s, v6.4s
                    fmla v3.4s, v7.4s, v7.4s
                    subs {c}, {c}, 1
                    bne 2b
                fadd v0.4s, v0.4s, v1.4s
                fadd v2.4s, v2.4s, v3.4s
                fadd v0.4s, v0.4s, v2.4s
                ",
                p = inout(reg) p => _,
                c = inout(reg) c => _,
                inout("v0") sum_v,
                out("v1") _, out("v2") _, out("v3") _,
                out("v4") _, out("v5") _, out("v6") _, out("v7") _,
            );
        }
        // horizontal sum across the 4 surviving lanes
        sum_sq = vaddvq_f32(sum_v);
    }
    // scalar tail
    for i in tail_start..n {
        let x = unsafe { *buf.get_unchecked(i) };
        sum_sq += x * x;
    }

    // --- Compute inv_std (scalar) ---
    let mean_sq = sum_sq / (n as f32);
    let inv_std = (mean_sq + eps).sqrt().recip();

    // --- Pass 2: multiply by inv_std ---
    if chunks > 0 {
        let p = ptr;
        let c = chunks;
        let inv_v: float32x4_t = vdupq_n_f32(inv_std);
        unsafe {
            std::arch::asm!("
                2:
                    ld1 {{v4.4s, v5.4s, v6.4s, v7.4s}}, [{p}]
                    fmul v4.4s, v4.4s, v0.4s
                    fmul v5.4s, v5.4s, v0.4s
                    fmul v6.4s, v6.4s, v0.4s
                    fmul v7.4s, v7.4s, v0.4s
                    st1 {{v4.4s, v5.4s, v6.4s, v7.4s}}, [{p}], 64
                    subs {c}, {c}, 1
                    bne 2b
                ",
                p = inout(reg) p => _,
                c = inout(reg) c => _,
                in("v0") inv_v,
                out("v4") _, out("v5") _, out("v6") _, out("v7") _,
            );
        }
    }
    for i in tail_start..n {
        unsafe {
            *buf.get_unchecked_mut(i) *= inv_std;
        }
    }
}

pub fn rms_norm_f32(buf: &mut [f32], eps: f32) {
    if buf.is_empty() {
        return;
    }
    unsafe { rms_norm_f32_inner(buf, eps) }
}

// f16 rows, arithmetic unchanged from `rms_norm_f32_inner`: FCVTL widens each
// 16-element group into the same four f32 registers before the same FMLA chain,
// and FCVTN rounds to nearest-even on the way out, matching `f16::from_f32`.
#[target_feature(enable = "neon")]
unsafe fn rms_norm_f16_inner(buf: &mut [f16], eps: f32) {
    use std::arch::aarch64::*;
    let n = buf.len();
    let chunks = n / 16;
    let tail_start = chunks * 16;
    let ptr = buf.as_mut_ptr();

    let mut sum_sq: f32 = 0.0;
    if chunks > 0 {
        let p = ptr;
        let c = chunks;
        let mut sum_v: float32x4_t = vdupq_n_f32(0.0);
        unsafe {
            std::arch::asm!("
                movi v1.4s, 0
                movi v2.4s, 0
                movi v3.4s, 0
                2:
                    ld1 {{v16.8h, v17.8h}}, [{p}], 32
                    fcvtl  v4.4s, v16.4h
                    fcvtl2 v5.4s, v16.8h
                    fcvtl  v6.4s, v17.4h
                    fcvtl2 v7.4s, v17.8h
                    fmla v0.4s, v4.4s, v4.4s
                    fmla v1.4s, v5.4s, v5.4s
                    fmla v2.4s, v6.4s, v6.4s
                    fmla v3.4s, v7.4s, v7.4s
                    subs {c}, {c}, 1
                    bne 2b
                fadd v0.4s, v0.4s, v1.4s
                fadd v2.4s, v2.4s, v3.4s
                fadd v0.4s, v0.4s, v2.4s
                ",
                p = inout(reg) p => _,
                c = inout(reg) c => _,
                inout("v0") sum_v,
                out("v1") _, out("v2") _, out("v3") _,
                out("v4") _, out("v5") _, out("v6") _, out("v7") _,
                out("v16") _, out("v17") _,
            );
        }
        sum_sq = vaddvq_f32(sum_v);
    }
    for i in tail_start..n {
        let x = unsafe { buf.get_unchecked(i).to_f32() };
        sum_sq += x * x;
    }

    let mean_sq = sum_sq / (n as f32);
    let inv_std = (mean_sq + eps).sqrt().recip();

    if chunks > 0 {
        let p = ptr;
        let c = chunks;
        let inv_v: float32x4_t = vdupq_n_f32(inv_std);
        unsafe {
            std::arch::asm!("
                2:
                    ld1 {{v16.8h, v17.8h}}, [{p}]
                    fcvtl  v4.4s, v16.4h
                    fcvtl2 v5.4s, v16.8h
                    fcvtl  v6.4s, v17.4h
                    fcvtl2 v7.4s, v17.8h
                    fmul v4.4s, v4.4s, v0.4s
                    fmul v5.4s, v5.4s, v0.4s
                    fmul v6.4s, v6.4s, v0.4s
                    fmul v7.4s, v7.4s, v0.4s
                    fcvtn  v16.4h, v4.4s
                    fcvtn2 v16.8h, v5.4s
                    fcvtn  v17.4h, v6.4s
                    fcvtn2 v17.8h, v7.4s
                    st1 {{v16.8h, v17.8h}}, [{p}], 32
                    subs {c}, {c}, 1
                    bne 2b
                ",
                p = inout(reg) p => _,
                c = inout(reg) c => _,
                in("v0") inv_v,
                out("v4") _, out("v5") _, out("v6") _, out("v7") _,
                out("v16") _, out("v17") _,
            );
        }
    }
    for i in tail_start..n {
        unsafe {
            let x = buf.get_unchecked(i).to_f32();
            *buf.get_unchecked_mut(i) = f16::from_f32(x * inv_std);
        }
    }
}

pub fn rms_norm_f16(buf: &mut [f16], eps: f32) {
    if buf.is_empty() {
        return;
    }
    unsafe { rms_norm_f16_inner(buf, eps) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ref_rms_norm(buf: &mut [f32], eps: f32) {
        let n = buf.len() as f32;
        let sum_sq: f32 = buf.iter().map(|x| x * x).sum();
        let mean_sq = sum_sq / n;
        let inv_std = (mean_sq + eps).sqrt().recip();
        for x in buf.iter_mut() {
            *x *= inv_std;
        }
    }

    fn close_enough(got: &[f32], want: &[f32], tol: f32) {
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            let diff = (g - w).abs();
            assert!(diff <= tol, "lane {i}: got {g}, want {w}, diff {diff}");
        }
    }

    #[test]
    fn matches_reference_16() {
        // 16 = exactly one inner iteration, no tail.
        let mut x: Vec<f32> = (0..16).map(|i| (i as f32 * 0.13).sin() * 5.0).collect();
        let mut y = x.clone();
        rms_norm_f32(&mut x, 1e-5);
        ref_rms_norm(&mut y, 1e-5);
        close_enough(&x, &y, 1e-5);
    }

    #[test]
    fn matches_reference_1024_with_tail() {
        // 1024 + 7 = exercises the scalar tail loop (len % 16 = 7).
        let n = 1024 + 7;
        let mut x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.07).cos() * 3.0).collect();
        let mut y = x.clone();
        rms_norm_f32(&mut x, 1e-5);
        ref_rms_norm(&mut y, 1e-5);
        close_enough(&x, &y, 1e-4);
    }

    #[test]
    fn matches_reference_short_below_chunk() {
        // 8 elements — shorter than one NEON iteration; all scalar tail.
        let mut x: Vec<f32> = vec![0.5, -1.5, 2.5, -3.5, 0.0, 4.0, -4.0, 1.0];
        let mut y = x.clone();
        rms_norm_f32(&mut x, 1e-5);
        ref_rms_norm(&mut y, 1e-5);
        close_enough(&x, &y, 1e-5);
    }

    #[test]
    fn empty_is_noop() {
        let mut x: Vec<f32> = vec![];
        rms_norm_f32(&mut x, 1e-5);
        assert!(x.is_empty());
        let mut h: Vec<f16> = vec![];
        rms_norm_f16(&mut h, 1e-5);
        assert!(h.is_empty());
    }

    #[test]
    fn f16_matches_widening_the_row_to_f32() {
        for len in [1usize, 7, 15, 16, 17, 33, 64, 129, 1024, 4096] {
            let row: Vec<f16> =
                (0..len).map(|i| f16::from_f32((i as f32 * 0.13).sin() * 5.0)).collect();

            let mut got = row.clone();
            rms_norm_f16(&mut got, 1e-5);

            let mut widened: Vec<f32> = row.iter().map(|x| x.to_f32()).collect();
            rms_norm_f32(&mut widened, 1e-5);
            let want: Vec<f16> = widened.into_iter().map(f16::from_f32).collect();

            let mismatch = got.iter().zip(&want).position(|(a, b)| a.to_bits() != b.to_bits());
            assert_eq!(mismatch, None, "len {len}");
        }
    }
}
