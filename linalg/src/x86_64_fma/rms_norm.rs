// Fused AVX-512 RmsNorm (single contiguous row):
//   out[i] = x[i] * rsqrt(mean(x[i]²) + eps)
//
// Two passes over the row:
//   Pass 1 (sum of squares):   acc += x² over 4 zmm accumulators, then reduce
//                              to a scalar f32; scalar tail handles the
//                              (len % 64) remainder.
//   Pass 2 (multiply-back):    broadcast inv_std into zmm0, multiply each
//                              4-zmm chunk in place; scalar tail.
//
// Uses unaligned loads/stores (vmovups) — the caller hands per-row slices of a
// possibly-misaligned tensor, so we can't assume 64-byte alignment. On Cascade
// Lake the unaligned penalty is negligible for cache-resident rows.

#[target_feature(enable = "avx512f")]
unsafe fn rms_norm_f32_inner(buf: &mut [f32], eps: f32) {
    let n = buf.len();
    let chunks = n / 64;
    let tail_start = chunks * 64;
    let ptr = buf.as_mut_ptr();

    // --- Pass 1: sum of squares ---
    let mut sum_sq: f32 = 0.0;
    if chunks > 0 {
        let p = ptr;
        let c = chunks;
        unsafe {
            std::arch::asm!("
                vpxord zmm0, zmm0, zmm0
                vpxord zmm1, zmm1, zmm1
                vpxord zmm2, zmm2, zmm2
                vpxord zmm3, zmm3, zmm3
                2:
                    vmovups zmm4, [{p}]
                    vmovups zmm5, [{p} + 64]
                    vmovups zmm6, [{p} + 128]
                    vmovups zmm7, [{p} + 192]
                    vfmadd231ps zmm0, zmm4, zmm4
                    vfmadd231ps zmm1, zmm5, zmm5
                    vfmadd231ps zmm2, zmm6, zmm6
                    vfmadd231ps zmm3, zmm7, zmm7
                    add {p}, 256
                    sub {c}, 1
                    jnz 2b

                vaddps zmm0, zmm0, zmm1
                vaddps zmm2, zmm2, zmm3
                vaddps zmm0, zmm0, zmm2
                vextractf64x4 ymm1, zmm0, 1
                vaddps ymm0, ymm0, ymm1
                vextractf128 xmm1, ymm0, 1
                vaddps xmm0, xmm0, xmm1
                vpermilps xmm1, xmm0, 2 + (3 << 2)
                vaddps xmm0, xmm0, xmm1
                vpermilps xmm1, xmm0, 1
                vaddps xmm0, xmm0, xmm1
                ",
                p = inout(reg) p => _,
                c = inout(reg) c => _,
                out("xmm0") sum_sq,
                out("zmm1") _, out("zmm2") _, out("zmm3") _,
                out("zmm4") _, out("zmm5") _, out("zmm6") _, out("zmm7") _,
            );
        }
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
        let inv = inv_std;
        unsafe {
            std::arch::asm!("
                vbroadcastss zmm0, xmm0
                2:
                    vmovups zmm1, [{p}]
                    vmovups zmm2, [{p} + 64]
                    vmovups zmm3, [{p} + 128]
                    vmovups zmm4, [{p} + 192]
                    vmulps zmm1, zmm1, zmm0
                    vmulps zmm2, zmm2, zmm0
                    vmulps zmm3, zmm3, zmm0
                    vmulps zmm4, zmm4, zmm0
                    vmovups [{p}],       zmm1
                    vmovups [{p} + 64],  zmm2
                    vmovups [{p} + 128], zmm3
                    vmovups [{p} + 192], zmm4
                    add {p}, 256
                    sub {c}, 1
                    jnz 2b
                ",
                p = inout(reg) p => _,
                c = inout(reg) c => _,
                inout("xmm0") inv => _,
                out("zmm1") _, out("zmm2") _, out("zmm3") _, out("zmm4") _,
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
    fn matches_reference_64() {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }
        let mut x: Vec<f32> = (0..64).map(|i| (i as f32 * 0.13).sin() * 5.0).collect();
        let mut y = x.clone();
        rms_norm_f32(&mut x, 1e-5);
        ref_rms_norm(&mut y, 1e-5);
        close_enough(&x, &y, 1e-5);
    }

    #[test]
    fn matches_reference_1024_with_tail() {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }
        // 1024 + 17 = a row that exercises the scalar tail loop.
        let n = 1024 + 17;
        let mut x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.07).cos() * 3.0).collect();
        let mut y = x.clone();
        rms_norm_f32(&mut x, 1e-5);
        ref_rms_norm(&mut y, 1e-5);
        close_enough(&x, &y, 1e-4);
    }

    #[test]
    fn matches_reference_short_below_chunk() {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }
        // Shorter than 64 -> all scalar tail.
        let mut x: Vec<f32> = vec![0.5, -1.5, 2.5, -3.5, 0.0, 4.0, -4.0, 1.0];
        let mut y = x.clone();
        rms_norm_f32(&mut x, 1e-5);
        ref_rms_norm(&mut y, 1e-5);
        close_enough(&x, &y, 1e-5);
    }

    #[test]
    fn empty_is_noop() {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }
        let mut x: Vec<f32> = vec![];
        rms_norm_f32(&mut x, 1e-5);
        assert!(x.is_empty());
    }
}
