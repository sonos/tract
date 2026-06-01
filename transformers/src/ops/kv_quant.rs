//! KIVI-style KV-cache quantization (training-free): store the cache in low precision to
//! shrink memory **near-losslessly**, keeping every token (a gentler trade than evicting).
//!
//! The key asymmetry (Liu et al. 2024, KIVI): **Keys are quantized PER-CHANNEL** (each
//! head-dim channel gets its own scale — Keys have large-magnitude *outlier channels* that
//! would wreck a shared scale) and **Values PER-TOKEN**. Works for any model, no training.
//! (CommVQ's RoPE-commutative codebook is a fancier, model-specific follow-on.)
//!
//! This module is the quantize→dequantize math + its quality, validated here and on real
//! activations (`harness/kv_quant_real.py`). Packed storage / a cache op is the follow-on;
//! memory saving is analytic: `bits/16` of the f16 cache (int8 = 2×, int4 = 4× smaller).

use tract_nnef::tract_ndarray::{Array2, ArrayView2};

/// Affine quantize→dequantize a `[rows, cols]` matrix at `bits` bits, returning the
/// reconstructed (lossy) values. `by_row = true` gives each ROW its own scale (per-token,
/// for Values); `by_row = false` gives each COLUMN its own scale (per-channel, for Keys).
/// Reconstruction error per element is ≤ scale/2 of its group.
pub fn quant_dequant(x: ArrayView2<f32>, bits: u32, by_row: bool) -> Array2<f32> {
    assert!((1..=16).contains(&bits), "bits must be 1..=16");
    let levels = ((1u32 << bits) - 1) as f32;
    let (r, c) = x.dim();
    let mut out = Array2::<f32>::zeros((r, c));
    let n_groups = if by_row { r } else { c };
    for g in 0..n_groups {
        let group = if by_row { x.row(g) } else { x.column(g) };
        let lo = group.iter().copied().fold(f32::INFINITY, f32::min);
        let hi = group.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let scale = if hi > lo { (hi - lo) / levels } else { 1.0 };
        for (k, &v) in group.iter().enumerate() {
            let q = ((v - lo) / scale).round().clamp(0.0, levels);
            let deq = lo + q * scale;
            if by_row {
                out[(g, k)] = deq;
            } else {
                out[(k, g)] = deq;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use tract_nnef::tract_ndarray::{Array2, arr2};

    fn max_abs(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
    }

    // Reconstruction error shrinks as bits grow; 16-bit is ~exact.
    #[test]
    fn error_decreases_with_bits() {
        let x = arr2(&[[0.0f32, 1.0, 2.0, 3.0], [-1.0, 0.5, 4.0, 9.0], [2.0, 2.0, 2.0, 2.1]]);
        let e4 = max_abs(&x, &quant_dequant(x.view(), 4, false));
        let e8 = max_abs(&x, &quant_dequant(x.view(), 8, false));
        let e16 = max_abs(&x, &quant_dequant(x.view(), 16, false));
        assert!(e8 < e4, "more bits => less error ({e8} !< {e4})");
        assert!(e16 < e8, "16-bit tighter than 8-bit ({e16} !< {e8})");
        assert!(e16 < 1e-3, "16-bit near-exact, got {e16}");
        // per-element error within half a quantization step of each column's range
        let levels = (1u32 << 8) - 1;
        for j in 0..x.ncols() {
            let col = x.column(j);
            let (lo, hi) = (
                col.iter().copied().fold(f32::INFINITY, f32::min),
                col.iter().copied().fold(f32::NEG_INFINITY, f32::max),
            );
            let step = if hi > lo { (hi - lo) / levels as f32 } else { 0.0 };
            let q = quant_dequant(x.view(), 8, false);
            for i in 0..x.nrows() {
                assert!((x[(i, j)] - q[(i, j)]).abs() <= step / 2.0 + 1e-6);
            }
        }
    }

    // The KIVI insight: with an outlier CHANNEL (a high-magnitude column), per-channel
    // (per-column) quantization isolates it and stays accurate, while per-token (per-row)
    // lumps it with the small dims and crushes them. So per-channel ≪ per-row for Keys.
    #[test]
    fn per_channel_beats_per_row_on_outlier_channel() {
        // 4 tokens x 4 channels; channel 0 is a big-magnitude outlier, others are small.
        let x = arr2(&[
            [100.0f32, 0.10, -0.20, 0.05],
            [-90.0, 0.02, 0.30, -0.08],
            [120.0, -0.15, 0.10, 0.20],
            [-110.0, 0.07, -0.05, 0.12],
        ]);
        // The difference shows on the SMALL channels (cols 1..4): per-token lumps them with
        // the outlier and crushes them; per-channel isolates the outlier so they stay sharp.
        let small_err = |q: &Array2<f32>| -> f32 {
            (1..4)
                .flat_map(|j| (0..4).map(move |i| (i, j)))
                .map(|(i, j)| (x[(i, j)] - q[(i, j)]).abs())
                .fold(0.0, f32::max)
        };
        let pc = small_err(&quant_dequant(x.view(), 4, false)); // per-channel (by column)
        let pt = small_err(&quant_dequant(x.view(), 4, true)); // per-token (by row)
        assert!(pc < pt * 0.2, "per-channel ≫ better on the small dims: pc={pc} pt={pt}");
    }

    // 8-bit KV is near-lossless for attention output; quality improves with bits.
    #[test]
    fn attention_near_lossless_at_8bit() {
        // single head: Q[1,d] . K[s,d] -> softmax -> . V[s,d]
        let (s, d) = (12usize, 16usize);
        let mk = |seed: u64| -> Array2<f32> {
            let mut st = seed;
            Array2::from_shape_fn((s, d), |_| {
                st = st.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((st >> 40) as f32 / (1u64 << 24) as f32) - 0.5
            })
        };
        let q = mk(1).row(0).to_owned();
        let k = mk(2);
        let v = mk(3);
        let scale = 1.0 / (d as f32).sqrt();
        let attn = |k: &Array2<f32>, v: &Array2<f32>| -> Vec<f32> {
            let mut sc: Vec<f32> = (0..s).map(|j| q.dot(&k.row(j)) * scale).collect();
            let m = sc.iter().cloned().fold(f32::MIN, f32::max);
            let mut sum = 0.0;
            sc.iter_mut().for_each(|x| {
                *x = (*x - m).exp();
                sum += *x;
            });
            (0..d).map(|e| (0..s).map(|j| sc[j] / sum * v[(j, e)]).sum()).collect()
        };
        let full = attn(&k, &v);
        let dev = |bits: u32| -> f32 {
            // Keys per-channel (by col), Values per-token (by row) — the KIVI layout.
            let kq = quant_dequant(k.view(), bits, false);
            let vq = quant_dequant(v.view(), bits, true);
            let o = attn(&kq, &vq);
            let num: f32 = o.iter().zip(&full).map(|(a, b)| (a - b).powi(2)).sum::<f32>().sqrt();
            let den: f32 = full.iter().map(|x| x * x).sum::<f32>().sqrt();
            num / den.max(1e-9)
        };
        let (d4, d8, d12) = (dev(4), dev(8), dev(12));
        assert!(d8 < d4 && d12 < d8, "deviation must shrink with bits: 4={d4} 8={d8} 12={d12}");
        assert!(d8 < 0.02, "8-bit KV near-lossless for attention, got {d8}");
    }
}
