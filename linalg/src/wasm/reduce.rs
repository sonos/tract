// SIMD reductions. LLVM does not vectorize these on wasm32: it will not
// reassociate a floating-point accumulator, so the generic forms keep a serial
// dependency chain one element long. Independent accumulators plus a
// tree-shaped horizontal reduction remove it. The max and min lanes are
// combined with `total_cmp` so the result matches the generic kernels exactly,
// NaN ordering included.

routine_reduce_rust!(wasm32;
    f32,
    wasm_max_f32_32n,
    32,
    4,
    #[inline(never)]
    fn run(x: &[f32], _: ()) -> f32 {
        use std::arch::wasm32::*;
        {
            let mut acc = [f32x4_splat(f32::NEG_INFINITY); 8];
            let mut chunks = x.chunks_exact(32);
            for c in &mut chunks {
                for (j, a) in acc.iter_mut().enumerate() {
                    let k = j * 4;
                    *a = f32x4_pmax(*a, f32x4(c[k], c[k + 1], c[k + 2], c[k + 3]));
                }
            }
            let tail = chunks.remainder();
            let a01 = f32x4_pmax(acc[0], acc[1]);
            let a23 = f32x4_pmax(acc[2], acc[3]);
            let a45 = f32x4_pmax(acc[4], acc[5]);
            let a67 = f32x4_pmax(acc[6], acc[7]);
            let s = f32x4_pmax(f32x4_pmax(a01, a23), f32x4_pmax(a45, a67));
            let mut m = f32x4_extract_lane::<0>(s);
            for v in
                [f32x4_extract_lane::<1>(s), f32x4_extract_lane::<2>(s), f32x4_extract_lane::<3>(s)]
            {
                if v.total_cmp(&m) == std::cmp::Ordering::Greater {
                    m = v;
                }
            }
            for &v in tail {
                if v.total_cmp(&m) == std::cmp::Ordering::Greater {
                    m = v;
                }
            }
            m
        }
    },
    op(Max)
);

routine_reduce_rust!(wasm32;
    f32,
    wasm_min_f32_32n,
    32,
    4,
    #[inline(never)]
    fn run(x: &[f32], _: ()) -> f32 {
        use std::arch::wasm32::*;
        {
            let mut acc = [f32x4_splat(f32::INFINITY); 8];
            let mut chunks = x.chunks_exact(32);
            for c in &mut chunks {
                for (j, a) in acc.iter_mut().enumerate() {
                    let k = j * 4;
                    *a = f32x4_pmin(*a, f32x4(c[k], c[k + 1], c[k + 2], c[k + 3]));
                }
            }
            let tail = chunks.remainder();
            let a01 = f32x4_pmin(acc[0], acc[1]);
            let a23 = f32x4_pmin(acc[2], acc[3]);
            let a45 = f32x4_pmin(acc[4], acc[5]);
            let a67 = f32x4_pmin(acc[6], acc[7]);
            let s = f32x4_pmin(f32x4_pmin(a01, a23), f32x4_pmin(a45, a67));
            let mut m = f32x4_extract_lane::<0>(s);
            for v in
                [f32x4_extract_lane::<1>(s), f32x4_extract_lane::<2>(s), f32x4_extract_lane::<3>(s)]
            {
                if v.total_cmp(&m) == std::cmp::Ordering::Less {
                    m = v;
                }
            }
            for &v in tail {
                if v.total_cmp(&m) == std::cmp::Ordering::Less {
                    m = v;
                }
            }
            m
        }
    },
    op(Min)
);

routine_reduce_rust!(wasm32;
    f32,
    wasm_sum_f32_32n,
    32,
    4,
    #[inline(never)]
    fn run(x: &[f32], _: ()) -> f32 {
        use std::arch::wasm32::*;
        {
            let mut acc = [f32x4_splat(0f32); 8];
            let mut chunks = x.chunks_exact(32);
            for c in &mut chunks {
                for (j, a) in acc.iter_mut().enumerate() {
                    let k = j * 4;
                    *a = f32x4_add(*a, f32x4(c[k], c[k + 1], c[k + 2], c[k + 3]));
                }
            }
            let tail = chunks.remainder();
            let a01 = f32x4_add(acc[0], acc[1]);
            let a23 = f32x4_add(acc[2], acc[3]);
            let a45 = f32x4_add(acc[4], acc[5]);
            let a67 = f32x4_add(acc[6], acc[7]);
            let s = f32x4_add(f32x4_add(a01, a23), f32x4_add(a45, a67));
            let mut sum = f32x4_extract_lane::<0>(s)
                + f32x4_extract_lane::<1>(s)
                + f32x4_extract_lane::<2>(s)
                + f32x4_extract_lane::<3>(s);
            for &v in tail {
                sum += v;
            }
            sum
        }
    },
    op(Sum)
);

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use std::arch::wasm32::*;
use tract_data::internal::f16;

// f16 orders correctly as sign-magnitude, so this monotone integer mapping lets i16x8_max
// reduce the lanes without any conversion to f32.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
fn mono(v: v128) -> v128 {
    v128_xor(v, v128_and(i16x8_shr(v, 15), u16x8_splat(0x7fff)))
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
fn load8_f16(c: &[f16]) -> v128 {
    i16x8(
        c[0].to_bits() as i16,
        c[1].to_bits() as i16,
        c[2].to_bits() as i16,
        c[3].to_bits() as i16,
        c[4].to_bits() as i16,
        c[5].to_bits() as i16,
        c[6].to_bits() as i16,
        c[7].to_bits() as i16,
    )
}

// f16 lanes widened to f32 by bit surgery; wasm32 has no f16 conversion instruction and the
// f16x8 proposal is not exposed by stable Rust.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
fn widen_f16(h: v128) -> v128 {
    let sign = v128_and(u32x4_shl(h, 16), u32x4_splat(0x8000_0000));
    let exp = v128_and(u32x4_shr(h, 10), u32x4_splat(0x1f));
    let man = v128_and(h, u32x4_splat(0x3ff));
    let is_zero = u32x4_eq(v128_or(exp, man), u32x4_splat(0));
    let normal = v128_or(u32x4_shl(u32x4_add(exp, u32x4_splat(112)), 23), u32x4_shl(man, 13));
    v128_or(sign, v128_andnot(normal, is_zero))
}

routine_reduce_rust!(wasm32;
    f16,
    wasm_max_f16_32n,
    32,
    8,
    #[inline(never)]
    fn run(x: &[f16], _: ()) -> f16 {
        use std::arch::wasm32::*;
        let mut acc = [i16x8_splat(i16::MIN); 4];
        let mut rest = x;
        for &width in &[32, 16, 8] {
            let mut chunks = rest.chunks_exact(width);
            for c in &mut chunks {
                for (i, a) in acc.iter_mut().take(width / 8).enumerate() {
                    *a = i16x8_max(*a, mono(load8_f16(&c[i * 8..i * 8 + 8])));
                }
            }
            rest = chunks.remainder();
        }
        let a = i16x8_max(i16x8_max(acc[0], acc[1]), i16x8_max(acc[2], acc[3]));
        let best = [
            i16x8_extract_lane::<0>(a),
            i16x8_extract_lane::<1>(a),
            i16x8_extract_lane::<2>(a),
            i16x8_extract_lane::<3>(a),
            i16x8_extract_lane::<4>(a),
            i16x8_extract_lane::<5>(a),
            i16x8_extract_lane::<6>(a),
            i16x8_extract_lane::<7>(a),
        ]
        .into_iter()
        .fold(i16::MIN, i16::max);
        let mut out = f16::from_bits((best ^ ((best >> 15) & 0x7fff)) as u16);
        for v in rest {
            if v.total_cmp(&out) == std::cmp::Ordering::Greater {
                out = *v;
            }
        }
        out
    },
    op(Max)
);

routine_reduce_rust!(wasm32;
    f16,
    wasm_sum_f16_32n,
    32,
    8,
    #[inline(never)]
    fn run(x: &[f16], _: ()) -> f16 {
        use std::arch::wasm32::*;

        let mut a = [f32x4_splat(0.0); 8];
        let mut chunks = x.chunks_exact(8);
        for (idx, c) in chunks.by_ref().enumerate() {
            let v = i16x8(
                c[0].to_bits() as i16,
                c[1].to_bits() as i16,
                c[2].to_bits() as i16,
                c[3].to_bits() as i16,
                c[4].to_bits() as i16,
                c[5].to_bits() as i16,
                c[6].to_bits() as i16,
                c[7].to_bits() as i16,
            );
            let ai = (idx & 3) * 2;
            a[ai] = f32x4_add(a[ai], widen_f16(u32x4_extend_low_u16x8(v)));
            a[ai + 1] = f32x4_add(a[ai + 1], widen_f16(u32x4_extend_high_u16x8(v)));
        }
        let s = f32x4_add(
            f32x4_add(f32x4_add(a[0], a[1]), f32x4_add(a[2], a[3])),
            f32x4_add(f32x4_add(a[4], a[5]), f32x4_add(a[6], a[7])),
        );
        let mut out = f32x4_extract_lane::<0>(s)
            + f32x4_extract_lane::<1>(s)
            + f32x4_extract_lane::<2>(s)
            + f32x4_extract_lane::<3>(s);
        for v in chunks.remainder() {
            out += v.to_f32();
        }
        f16::from_f32(out)
    },
    op(Sum)
);

/// RMS-normalises `buf` in place: each element is divided by the root of the
/// mean square plus `eps`. The sum of squares keeps sixteen independent
/// accumulators so the multiply-accumulate latency does not serialise, which is
/// what the generic scalar form cannot avoid.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub fn rms_norm_f32(buf: &mut [f32], eps: f32) {
    use std::arch::wasm32::*;
    if buf.is_empty() {
        return;
    }
    {
        let len = buf.len();
        let mut acc = [f32x4_splat(0f32); 16];
        let mut chunks = buf.chunks_exact(64);
        for c in &mut chunks {
            for (j, a) in acc.iter_mut().enumerate() {
                let k = j * 4;
                let v = f32x4(c[k], c[k + 1], c[k + 2], c[k + 3]);
                *a = madd_f32x4!(*a, v, v);
            }
        }
        let mut pairs = [f32x4_splat(0f32); 8];
        for (k, p) in pairs.iter_mut().enumerate() {
            *p = f32x4_add(acc[2 * k], acc[2 * k + 1]);
        }
        let q0 = f32x4_add(f32x4_add(pairs[0], pairs[1]), f32x4_add(pairs[2], pairs[3]));
        let q1 = f32x4_add(f32x4_add(pairs[4], pairs[5]), f32x4_add(pairs[6], pairs[7]));
        let s = f32x4_add(q0, q1);
        let mut sum = f32x4_extract_lane::<0>(s)
            + f32x4_extract_lane::<1>(s)
            + f32x4_extract_lane::<2>(s)
            + f32x4_extract_lane::<3>(s);
        for &v in chunks.remainder() {
            sum += v * v;
        }

        let scale = 1f32 / (sum / len as f32 + eps).sqrt();
        let scale_v = f32x4_splat(scale);
        for c in buf.chunks_exact_mut(64) {
            for j in 0..16 {
                let k = j * 4;
                let r = f32x4_mul(f32x4(c[k], c[k + 1], c[k + 2], c[k + 3]), scale_v);
                c[k] = f32x4_extract_lane::<0>(r);
                c[k + 1] = f32x4_extract_lane::<1>(r);
                c[k + 2] = f32x4_extract_lane::<2>(r);
                c[k + 3] = f32x4_extract_lane::<3>(r);
            }
        }
        let tail_start = len - len % 64;
        for v in buf[tail_start..].iter_mut() {
            *v *= scale;
        }
    }
}

bail_stub!(wasm32; pub fn rms_norm_f32(&mut [f32], f32));

submit_routine!(wasm32; RmsNormF32, RmsNorm, "wasm_rms_norm_f32", rms_norm_f32);
