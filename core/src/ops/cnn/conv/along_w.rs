//! Vectorised convolution along a contiguous output-W axis (NCHW).
//!
//! Same inner loop ORT/MLAS uses for NCHW spatial conv: splat each tap, FMA a
//! SIMD load of X. Depthwise 3×3/5×5 is N=9/25; a grouped-1 stem 3×3×Cin is
//! N=Cin·9. Short rows and non-SIMD targets stay on the const-N monomorph in
//! `depth_wise.rs` — see `MIN_ALONG_W_LEN` and `HAS_SIMD_KERNEL`.

/// `conv_along_w_*` is only a win where it has a SIMD kernel. On every other
/// target it degrades to `scalar()`, a runtime-length serial `fmul`/`fadd`
/// chain, which loses to the const-N monomorph's four independent accumulators
/// (armv7 cortex-a7/a9, riscv64, wasm simd128).
pub const HAS_SIMD_KERNEL: bool = cfg!(any(target_arch = "aarch64", target_arch = "x86_64"));

/// Minimum contiguous outputs before the along-W handoff pays for itself. The
/// call is out-of-line and re-splats each tap per 8-output block, so a short
/// row (MobileNet 14×14 and 7×7 stages) is cheaper on the const-N body, which
/// keeps taps in registers and unrolls four outputs.
pub const MIN_ALONG_W_LEN: usize = 32;

/// F32 FIR along W. `in_stride` is the input step for one output step (1 for
/// stride-1, 2/3 for the DPDFNet encoder). `relu` applies `max(0, ·)` at the
/// store, matching ORT `FusedConv`.
///
/// # Safety
/// `iptr`/`optr` cover `len` outputs; `k`/`ioffset` are `n_taps` long; each
/// `iptr.offset(ioffset[n]).offset(i * in_stride)` for `i in 0..len` is in-bounds.
pub unsafe fn conv_along_w_f32(
    iptr: *const f32,
    optr: *mut f32,
    k: &[f32],
    ioffset: &[isize],
    bias: f32,
    len: usize,
    in_stride: isize,
    relu: bool,
) {
    debug_assert_eq!(k.len(), ioffset.len());
    #[cfg(target_arch = "aarch64")]
    unsafe {
        neon(iptr, optr, k, ioffset, bias, len, in_stride, relu);
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        if is_x86_feature_detected!("fma") && is_x86_feature_detected!("avx2") {
            avx2_fma(iptr, optr, k, ioffset, bias, len, in_stride, relu);
        } else {
            scalar(iptr, optr, k, ioffset, bias, len, in_stride, relu);
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    unsafe {
        scalar(iptr, optr, k, ioffset, bias, len, in_stride, relu);
    }
}

#[inline(always)]
fn maybe_relu(v: f32, relu: bool) -> f32 {
    if relu { v.max(0.0) } else { v }
}

#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
unsafe fn scalar(
    iptr: *const f32,
    optr: *mut f32,
    k: &[f32],
    ioffset: &[isize],
    bias: f32,
    len: usize,
    in_stride: isize,
    relu: bool,
) {
    unsafe {
        for i in 0..len {
            let mut sum = bias;
            for n in 0..k.len() {
                sum += k[n] * *iptr.offset(ioffset[n]).offset(i as isize * in_stride);
            }
            *optr.add(i) = maybe_relu(sum, relu);
        }
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn neon(
    iptr: *const f32,
    optr: *mut f32,
    k: &[f32],
    ioffset: &[isize],
    bias: f32,
    len: usize,
    in_stride: isize,
    relu: bool,
) {
    unsafe {
        use std::arch::aarch64::*;
        let n_taps = k.len();
        let biasv = vdupq_n_f32(bias);
        let z = vdupq_n_f32(0.0);
        let store = |p: *mut f32, mut acc: float32x4_t| {
            if relu {
                acc = vmaxq_f32(acc, z);
            }
            vst1q_f32(p, acc);
        };
        let mut i = 0usize;
        if in_stride == 1 {
            while i + 8 <= len {
                let mut acc0 = biasv;
                let mut acc1 = biasv;
                for n in 0..n_taps {
                    let kn = vdupq_n_f32(k[n]);
                    let p = iptr.offset(ioffset[n]).add(i);
                    acc0 = vfmaq_f32(acc0, vld1q_f32(p), kn);
                    acc1 = vfmaq_f32(acc1, vld1q_f32(p.add(4)), kn);
                }
                store(optr.add(i), acc0);
                store(optr.add(i + 4), acc1);
                i += 8;
            }
            while i + 4 <= len {
                let mut acc = biasv;
                for n in 0..n_taps {
                    let kn = vdupq_n_f32(k[n]);
                    acc = vfmaq_f32(acc, vld1q_f32(iptr.offset(ioffset[n]).add(i)), kn);
                }
                store(optr.add(i), acc);
                i += 4;
            }
        } else if in_stride == 2 {
            while i + 8 <= len {
                let mut acc0 = biasv;
                let mut acc1 = biasv;
                for n in 0..n_taps {
                    let kn = vdupq_n_f32(k[n]);
                    let p = iptr.offset(ioffset[n]).offset(i as isize * 2);
                    let a = vld2q_f32(p);
                    let b = vld2q_f32(p.add(8));
                    acc0 = vfmaq_f32(acc0, a.0, kn);
                    acc1 = vfmaq_f32(acc1, b.0, kn);
                }
                store(optr.add(i), acc0);
                store(optr.add(i + 4), acc1);
                i += 8;
            }
            while i + 4 <= len {
                let mut acc = biasv;
                for n in 0..n_taps {
                    let kn = vdupq_n_f32(k[n]);
                    let a = vld2q_f32(iptr.offset(ioffset[n]).offset(i as isize * 2));
                    acc = vfmaq_f32(acc, a.0, kn);
                }
                store(optr.add(i), acc);
                i += 4;
            }
        } else if in_stride == 3 {
            while i + 8 <= len {
                let mut acc0 = biasv;
                let mut acc1 = biasv;
                for n in 0..n_taps {
                    let kn = vdupq_n_f32(k[n]);
                    let p = iptr.offset(ioffset[n]).offset(i as isize * 3);
                    let a = vld3q_f32(p);
                    let b = vld3q_f32(p.add(12));
                    acc0 = vfmaq_f32(acc0, a.0, kn);
                    acc1 = vfmaq_f32(acc1, b.0, kn);
                }
                store(optr.add(i), acc0);
                store(optr.add(i + 4), acc1);
                i += 8;
            }
            while i + 4 <= len {
                let mut acc = biasv;
                for n in 0..n_taps {
                    let kn = vdupq_n_f32(k[n]);
                    let a = vld3q_f32(iptr.offset(ioffset[n]).offset(i as isize * 3));
                    acc = vfmaq_f32(acc, a.0, kn);
                }
                store(optr.add(i), acc);
                i += 4;
            }
        }
        while i < len {
            let mut sum = bias;
            for n in 0..n_taps {
                sum += k[n] * *iptr.offset(ioffset[n]).offset(i as isize * in_stride);
            }
            *optr.add(i) = maybe_relu(sum, relu);
            i += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn avx2_load8(p: *const f32, stride: isize) -> std::arch::x86_64::__m256 {
    unsafe {
        use std::arch::x86_64::*;
        if stride == 1 {
            _mm256_loadu_ps(p)
        } else if stride == 2 {
            // 16 consecutive → 8 evens: [0,2,4,6,8,10,12,14]
            let a = _mm256_loadu_ps(p);
            let b = _mm256_loadu_ps(p.add(8));
            let t0 = _mm256_permute2f128_ps(a, b, 0x20);
            let t1 = _mm256_permute2f128_ps(a, b, 0x31);
            _mm256_shuffle_ps(t0, t1, 0x88)
        } else {
            let s = stride as i32;
            let idx = _mm256_setr_epi32(0, s, 2 * s, 3 * s, 4 * s, 5 * s, 6 * s, 7 * s);
            _mm256_i32gather_ps(p, idx, 4)
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn avx2_fma(
    iptr: *const f32,
    optr: *mut f32,
    k: &[f32],
    ioffset: &[isize],
    bias: f32,
    len: usize,
    in_stride: isize,
    relu: bool,
) {
    unsafe {
        use std::arch::x86_64::*;
        let n_taps = k.len();
        let mut i = 0usize;
        if in_stride == 1 || in_stride == 2 || in_stride == 3 {
            let biasv = _mm256_set1_ps(bias);
            let z = _mm256_setzero_ps();
            while i + 8 <= len {
                let mut acc = biasv;
                for n in 0..n_taps {
                    let kn = _mm256_set1_ps(k[n]);
                    let x = avx2_load8(
                        iptr.offset(ioffset[n]).offset(i as isize * in_stride),
                        in_stride,
                    );
                    acc = _mm256_fmadd_ps(x, kn, acc);
                }
                if relu {
                    acc = _mm256_max_ps(acc, z);
                }
                _mm256_storeu_ps(optr.add(i), acc);
                i += 8;
            }
        }
        while i < len {
            let mut sum = bias;
            for n in 0..n_taps {
                sum += k[n] * *iptr.offset(ioffset[n]).offset(i as isize * in_stride);
            }
            *optr.add(i) = maybe_relu(sum, relu);
            i += 1;
        }
    }
}

/// Four output channels sharing the same X loads. `k` is `4 * n_taps`,
/// channel-major (`k[o * n_taps + t]`). `optr` is channel 0; the next
/// three sit `oc_stride` apart (NCHW: `H*W`).
///
/// # Safety
/// Same as `conv_along_w_f32`, plus `optr.offset(o * oc_stride)` valid for
/// `o in 0..4`.
pub unsafe fn conv_along_w_oc4_f32(
    iptr: *const f32,
    optr: *mut f32,
    k: &[f32],
    ioffset: &[isize],
    bias: &[f32; 4],
    len: usize,
    in_stride: isize,
    oc_stride: isize,
    relu: bool,
) {
    debug_assert_eq!(k.len(), 4 * ioffset.len());
    #[cfg(target_arch = "aarch64")]
    unsafe {
        neon_oc4(iptr, optr, k, ioffset, bias, len, in_stride, oc_stride, relu);
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        if is_x86_feature_detected!("fma") && is_x86_feature_detected!("avx2") {
            avx2_oc4(iptr, optr, k, ioffset, bias, len, in_stride, oc_stride, relu);
        } else {
            scalar_oc4(iptr, optr, k, ioffset, bias, len, in_stride, oc_stride, relu);
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    unsafe {
        scalar_oc4(iptr, optr, k, ioffset, bias, len, in_stride, oc_stride, relu);
    }
}

#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
unsafe fn scalar_oc4(
    iptr: *const f32,
    optr: *mut f32,
    k: &[f32],
    ioffset: &[isize],
    bias: &[f32; 4],
    len: usize,
    in_stride: isize,
    oc_stride: isize,
    relu: bool,
) {
    unsafe {
        let n_taps = ioffset.len();
        for i in 0..len {
            let ipi = iptr.offset(i as isize * in_stride);
            for o in 0..4 {
                let mut sum = bias[o];
                let ko = o * n_taps;
                for t in 0..n_taps {
                    sum += k[ko + t] * *ipi.offset(ioffset[t]);
                }
                *optr.offset(o as isize * oc_stride).add(i) = maybe_relu(sum, relu);
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn neon_oc4(
    iptr: *const f32,
    optr: *mut f32,
    k: &[f32],
    ioffset: &[isize],
    bias: &[f32; 4],
    len: usize,
    in_stride: isize,
    oc_stride: isize,
    relu: bool,
) {
    unsafe {
        use std::arch::aarch64::*;
        let n_taps = ioffset.len();
        let z = vdupq_n_f32(0.0);
        let b0 = vdupq_n_f32(bias[0]);
        let b1 = vdupq_n_f32(bias[1]);
        let b2 = vdupq_n_f32(bias[2]);
        let b3 = vdupq_n_f32(bias[3]);
        let store4 = |base: *mut f32, i: usize, mut a0, mut a1, mut a2, mut a3| {
            if relu {
                a0 = vmaxq_f32(a0, z);
                a1 = vmaxq_f32(a1, z);
                a2 = vmaxq_f32(a2, z);
                a3 = vmaxq_f32(a3, z);
            }
            vst1q_f32(base.add(i), a0);
            vst1q_f32(base.offset(oc_stride).add(i), a1);
            vst1q_f32(base.offset(2 * oc_stride).add(i), a2);
            vst1q_f32(base.offset(3 * oc_stride).add(i), a3);
        };
        let mut i = 0usize;
        if in_stride == 1 {
            while i + 8 <= len {
                let mut a0 = b0;
                let mut a0b = b0;
                let mut a1 = b1;
                let mut a1b = b1;
                let mut a2 = b2;
                let mut a2b = b2;
                let mut a3 = b3;
                let mut a3b = b3;
                for t in 0..n_taps {
                    let p = iptr.offset(ioffset[t]).add(i);
                    let x0 = vld1q_f32(p);
                    let x1 = vld1q_f32(p.add(4));
                    let k0 = vdupq_n_f32(k[t]);
                    let k1 = vdupq_n_f32(k[n_taps + t]);
                    let k2 = vdupq_n_f32(k[2 * n_taps + t]);
                    let k3 = vdupq_n_f32(k[3 * n_taps + t]);
                    a0 = vfmaq_f32(a0, x0, k0);
                    a0b = vfmaq_f32(a0b, x1, k0);
                    a1 = vfmaq_f32(a1, x0, k1);
                    a1b = vfmaq_f32(a1b, x1, k1);
                    a2 = vfmaq_f32(a2, x0, k2);
                    a2b = vfmaq_f32(a2b, x1, k2);
                    a3 = vfmaq_f32(a3, x0, k3);
                    a3b = vfmaq_f32(a3b, x1, k3);
                }
                store4(optr, i, a0, a1, a2, a3);
                store4(optr, i + 4, a0b, a1b, a2b, a3b);
                i += 8;
            }
            while i + 4 <= len {
                let mut a0 = b0;
                let mut a1 = b1;
                let mut a2 = b2;
                let mut a3 = b3;
                for t in 0..n_taps {
                    let x = vld1q_f32(iptr.offset(ioffset[t]).add(i));
                    a0 = vfmaq_f32(a0, x, vdupq_n_f32(k[t]));
                    a1 = vfmaq_f32(a1, x, vdupq_n_f32(k[n_taps + t]));
                    a2 = vfmaq_f32(a2, x, vdupq_n_f32(k[2 * n_taps + t]));
                    a3 = vfmaq_f32(a3, x, vdupq_n_f32(k[3 * n_taps + t]));
                }
                store4(optr, i, a0, a1, a2, a3);
                i += 4;
            }
        } else if in_stride == 2 {
            while i + 8 <= len {
                let mut a0 = b0;
                let mut a0b = b0;
                let mut a1 = b1;
                let mut a1b = b1;
                let mut a2 = b2;
                let mut a2b = b2;
                let mut a3 = b3;
                let mut a3b = b3;
                for t in 0..n_taps {
                    let p = iptr.offset(ioffset[t]).offset(i as isize * 2);
                    let x0 = vld2q_f32(p).0;
                    let x1 = vld2q_f32(p.add(8)).0;
                    let k0 = vdupq_n_f32(k[t]);
                    let k1 = vdupq_n_f32(k[n_taps + t]);
                    let k2 = vdupq_n_f32(k[2 * n_taps + t]);
                    let k3 = vdupq_n_f32(k[3 * n_taps + t]);
                    a0 = vfmaq_f32(a0, x0, k0);
                    a0b = vfmaq_f32(a0b, x1, k0);
                    a1 = vfmaq_f32(a1, x0, k1);
                    a1b = vfmaq_f32(a1b, x1, k1);
                    a2 = vfmaq_f32(a2, x0, k2);
                    a2b = vfmaq_f32(a2b, x1, k2);
                    a3 = vfmaq_f32(a3, x0, k3);
                    a3b = vfmaq_f32(a3b, x1, k3);
                }
                store4(optr, i, a0, a1, a2, a3);
                store4(optr, i + 4, a0b, a1b, a2b, a3b);
                i += 8;
            }
            while i + 4 <= len {
                let mut a0 = b0;
                let mut a1 = b1;
                let mut a2 = b2;
                let mut a3 = b3;
                for t in 0..n_taps {
                    let x = vld2q_f32(iptr.offset(ioffset[t]).offset(i as isize * 2)).0;
                    a0 = vfmaq_f32(a0, x, vdupq_n_f32(k[t]));
                    a1 = vfmaq_f32(a1, x, vdupq_n_f32(k[n_taps + t]));
                    a2 = vfmaq_f32(a2, x, vdupq_n_f32(k[2 * n_taps + t]));
                    a3 = vfmaq_f32(a3, x, vdupq_n_f32(k[3 * n_taps + t]));
                }
                store4(optr, i, a0, a1, a2, a3);
                i += 4;
            }
        }
        if i < len {
            scalar_oc4(
                iptr.offset(i as isize * in_stride),
                optr.add(i),
                k,
                ioffset,
                bias,
                len - i,
                in_stride,
                oc_stride,
                relu,
            );
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn avx2_oc4(
    iptr: *const f32,
    optr: *mut f32,
    k: &[f32],
    ioffset: &[isize],
    bias: &[f32; 4],
    len: usize,
    in_stride: isize,
    oc_stride: isize,
    relu: bool,
) {
    unsafe {
        use std::arch::x86_64::*;
        let n_taps = ioffset.len();
        let mut i = 0usize;
        if in_stride == 1 || in_stride == 2 || in_stride == 3 {
            let z = _mm256_setzero_ps();
            let b0 = _mm256_set1_ps(bias[0]);
            let b1 = _mm256_set1_ps(bias[1]);
            let b2 = _mm256_set1_ps(bias[2]);
            let b3 = _mm256_set1_ps(bias[3]);
            while i + 8 <= len {
                let mut a0 = b0;
                let mut a1 = b1;
                let mut a2 = b2;
                let mut a3 = b3;
                for t in 0..n_taps {
                    let x = avx2_load8(
                        iptr.offset(ioffset[t]).offset(i as isize * in_stride),
                        in_stride,
                    );
                    a0 = _mm256_fmadd_ps(x, _mm256_set1_ps(k[t]), a0);
                    a1 = _mm256_fmadd_ps(x, _mm256_set1_ps(k[n_taps + t]), a1);
                    a2 = _mm256_fmadd_ps(x, _mm256_set1_ps(k[2 * n_taps + t]), a2);
                    a3 = _mm256_fmadd_ps(x, _mm256_set1_ps(k[3 * n_taps + t]), a3);
                }
                if relu {
                    a0 = _mm256_max_ps(a0, z);
                    a1 = _mm256_max_ps(a1, z);
                    a2 = _mm256_max_ps(a2, z);
                    a3 = _mm256_max_ps(a3, z);
                }
                _mm256_storeu_ps(optr.add(i), a0);
                _mm256_storeu_ps(optr.offset(oc_stride).add(i), a1);
                _mm256_storeu_ps(optr.offset(2 * oc_stride).add(i), a2);
                _mm256_storeu_ps(optr.offset(3 * oc_stride).add(i), a3);
                i += 8;
            }
        }
        if i < len {
            scalar_oc4(
                iptr.offset(i as isize * in_stride),
                optr.add(i),
                k,
                ioffset,
                bias,
                len - i,
                in_stride,
                oc_stride,
                relu,
            );
        }
    }
}
