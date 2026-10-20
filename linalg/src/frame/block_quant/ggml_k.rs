//! ggml K-quant blocks: 256-value super-blocks laid out like llama.cpp `block_q*_K`.
//!
//! The quantizers are llama.cpp's `quantize_row_q*_K_ref`, without importance matrix.
//! They are the reference the formats are validated against, not a quantization recipe.
use super::opaque::impl_opaque_block_quant;
use super::opaque::load_f16;
use super::opaque::store_f16;
use super::*;

pub(super) const QK_K: usize = 256;
pub(super) const GROUP_MAX_EPS: f32 = 1e-15;

/// ggml Q4_K block: f16 `d` and `dmin`, eight 6-bit scale and min pairs in 12 bytes,
/// then 256 4-bit levels, 144 bytes.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct Q4KF;

/// The ggml Q4_K block format.
pub const Q4_K: Q4KF = Q4KF;

/// ggml Q5_K block: the Q4_K header, the fifth bit of each level in 32 bytes, then
/// the low four bits of 256 levels, 176 bytes.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct Q5KF;

/// The ggml Q5_K block format.
pub const Q5_K: Q5KF = Q5KF;

/// ggml Q6_K block: low four bits of 256 levels, their high two bits in 64 bytes,
/// sixteen `i8` group scales, then the f16 `d`, 210 bytes.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct Q6KF;

/// The ggml Q6_K block format.
pub const Q6_K: Q6KF = Q6KF;

/// ggml `nearest_int`: round half to even through the 2^23 mantissa trick.
pub(super) fn nearest_int(fval: f32) -> i32 {
    let val = fval + 12582912.0;
    let i = val.to_bits() as i32;
    (i & 0x007fffff) - 0x00400000
}

/// ggml `make_qx_quants` with `rmse_type` 1, weighting each value by its square.
/// Writes levels in `0..2 * nmax` and returns the scale.
fn make_qx_quants(x: &[f32], l_out: &mut [i8], nmax: i32) -> f32 {
    let mut max = 0.0f32;
    let mut amax = 0.0f32;
    for &xi in x {
        if xi.abs() > amax {
            amax = xi.abs();
            max = xi;
        }
    }
    if amax < GROUP_MAX_EPS {
        l_out.fill(0);
        return 0.0;
    }
    let fit = |iscale: f32| {
        let mut sumlx = 0.0f32;
        let mut suml2 = 0.0f32;
        for &xi in x {
            let l = nearest_int(iscale * xi).clamp(-nmax, nmax - 1);
            let w = xi * xi;
            sumlx += w * xi * l as f32;
            suml2 += w * (l * l) as f32;
        }
        (sumlx, suml2)
    };
    let fill = |iscale: f32, l_out: &mut [i8]| {
        for (l, &xi) in l_out.iter_mut().zip(x) {
            *l = (nearest_int(iscale * xi).clamp(-nmax, nmax - 1) + nmax) as i8;
        }
    };
    let iscale = -nmax as f32 / max;
    fill(iscale, l_out);
    let (sumlx, suml2) = fit(iscale);
    let mut scale = if suml2 != 0.0 { sumlx / suml2 } else { 0.0 };
    let mut best = scale * sumlx;
    for is in (-9..=9).filter(|is| *is != 0) {
        let iscale = -(nmax as f32 + 0.1 * is as f32) / max;
        let (sumlx, suml2) = fit(iscale);
        if suml2 > 0.0 && sumlx * sumlx > best * suml2 {
            fill(iscale, l_out);
            scale = sumlx / suml2;
            best = scale * sumlx;
        }
    }
    scale
}

#[allow(clippy::too_many_arguments)]
fn make_qkx2_quants(
    x: &[f32],
    weights: &[f32],
    l_out: &mut [u8],
    the_min: &mut f32,
    nmax: i32,
    rmin: f32,
    rdelta: f32,
    nstep: i32,
) -> f32 {
    let n = x.len();
    let mut min = x[0];
    let mut max = x[0];
    let mut sum_w = weights[0];
    let mut sum_x = sum_w * x[0];
    for i in 1..n {
        min = min.min(x[i]);
        max = max.max(x[i]);
        sum_w += weights[i];
        sum_x += weights[i] * x[i];
    }
    if min > 0.0 {
        min = 0.0;
    }
    if max == min {
        l_out.fill(0);
        *the_min = -min;
        return 0.0;
    }
    let mut iscale = nmax as f32 / (max - min);
    let mut scale = 1.0 / iscale;
    let mut best_error = 0.0f32;
    for i in 0..n {
        let l = nearest_int(iscale * (x[i] - min)).clamp(0, nmax);
        l_out[i] = l as u8;
        let diff = scale * l as f32 + min - x[i];
        best_error += weights[i] * diff * diff;
    }
    let mut laux = [0u8; 32];
    for is in 0..=nstep {
        iscale = (rmin + rdelta * is as f32 + nmax as f32) / (max - min);
        let mut sum_l = 0.0f32;
        let mut sum_l2 = 0.0f32;
        let mut sum_xl = 0.0f32;
        for i in 0..n {
            let l = nearest_int(iscale * (x[i] - min)).clamp(0, nmax);
            laux[i] = l as u8;
            let w = weights[i];
            sum_l += w * l as f32;
            sum_l2 += w * (l * l) as f32;
            sum_xl += w * l as f32 * x[i];
        }
        let d = sum_w * sum_l2 - sum_l * sum_l;
        if d > 0.0 {
            let mut this_scale = (sum_w * sum_xl - sum_x * sum_l) / d;
            let mut this_min = (sum_l2 * sum_x - sum_l * sum_xl) / d;
            if this_min > 0.0 {
                this_min = 0.0;
                this_scale = sum_xl / sum_l2;
            }
            let mut cur_error = 0.0f32;
            for i in 0..n {
                let diff = this_scale * laux[i] as f32 + this_min - x[i];
                cur_error += weights[i] * diff * diff;
            }
            if cur_error < best_error {
                l_out[..n].copy_from_slice(&laux[..n]);
                best_error = cur_error;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    scale
}

/// Scale and min of group `j` from the 12-byte `scales` of a Q4_K or Q5_K block.
fn scale_min_k4(q: &[u8], j: usize) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        let d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

/// Writes the 16-byte header of a Q4_K or Q5_K block (`d`, `dmin`, packed scales and
/// mins) and the `0..=nmax` level of each of the 256 values of `x`.
fn quant_scales_mins(
    x: &[f32],
    header: &mut [u8],
    levels: &mut [u8; QK_K],
    nmax: i32,
    rmin: f32,
    nstep: i32,
) {
    let mut weights = [0.0f32; 32];
    let mut mins = [0.0f32; 8];
    let mut scales = [0.0f32; 8];
    for j in 0..8 {
        let x = &x[32 * j..][..32];
        let av_x = (x.iter().map(|v| v * v).sum::<f32>() / 32.0).sqrt();
        for (w, v) in weights.iter_mut().zip(x) {
            *w = av_x + v.abs();
        }
        let levels = &mut levels[32 * j..][..32];
        scales[j] = make_qkx2_quants(x, &weights, levels, &mut mins[j], nmax, rmin, 0.1, nstep);
    }
    let max_scale = scales.iter().fold(0f32, |a, &b| a.max(b));
    let max_min = mins.iter().fold(0f32, |a, &b| a.max(b));
    let inv_scale = if max_scale > 0.0 { 63.0 / max_scale } else { 0.0 };
    let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };
    let sc = &mut header[4..16];
    for j in 0..8 {
        let ls = (nearest_int(inv_scale * scales[j]) as u8).min(63);
        let lm = (nearest_int(inv_min * mins[j]) as u8).min(63);
        if j < 4 {
            sc[j] = ls;
            sc[j + 4] = lm;
        } else {
            sc[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
            sc[j - 4] |= (ls >> 4) << 6;
            sc[j] |= (lm >> 4) << 6;
        }
    }
    store_f16(&mut header[0..2], max_scale / 63.0);
    store_f16(&mut header[2..4], max_min / 63.0);
    let d_super = load_f16(&header[0..2]);
    let m_super = load_f16(&header[2..4]);
    for j in 0..8 {
        let (sc, m) = scale_min_k4(&header[4..16], j);
        let d = d_super * sc as f32;
        if d == 0.0 {
            continue;
        }
        let dm = m_super * m as f32;
        for i in 32 * j..32 * j + 32 {
            levels[i] = nearest_int((x[i] + dm) / d).clamp(0, nmax) as u8;
        }
    }
}

fn quant_q4_k(x: &[f32], quant: &mut [u8]) {
    quant.fill(0);
    let mut levels = [0u8; QK_K];
    quant_scales_mins(x, &mut quant[..16], &mut levels, 15, -1.0, 20);
    for (chunk, qs) in levels.chunks_exact(64).zip(quant[16..].chunks_exact_mut(32)) {
        for (i, q) in qs.iter_mut().enumerate() {
            *q = chunk[i] | (chunk[i + 32] << 4);
        }
    }
}

fn dequant_q4_k(quant: &[u8], block: &mut [f32]) {
    let d = load_f16(&quant[0..2]);
    let min = load_f16(&quant[2..4]);
    let scales = &quant[4..16];
    for (pair, (out, qs)) in
        block.chunks_exact_mut(64).zip(quant[16..].chunks_exact(32)).enumerate()
    {
        let (sc1, m1) = scale_min_k4(scales, 2 * pair);
        let (sc2, m2) = scale_min_k4(scales, 2 * pair + 1);
        for (l, &q) in qs.iter().enumerate() {
            out[l] = d * sc1 as f32 * (q & 0xF) as f32 - min * m1 as f32;
            out[l + 32] = d * sc2 as f32 * (q >> 4) as f32 - min * m2 as f32;
        }
    }
}

fn quant_q5_k(x: &[f32], quant: &mut [u8]) {
    quant.fill(0);
    let mut levels = [0u8; QK_K];
    quant_scales_mins(x, &mut quant[..16], &mut levels, 31, -0.5, 15);
    let (qh, ql) = quant[16..].split_at_mut(32);
    for (pair, (chunk, ql)) in levels.chunks_exact(64).zip(ql.chunks_exact_mut(32)).enumerate() {
        for i in 0..32 {
            let (l1, l2) = (chunk[i], chunk[i + 32]);
            qh[i] |= ((l1 >> 4) << (2 * pair)) | ((l2 >> 4) << (2 * pair + 1));
            ql[i] = (l1 & 0xF) | ((l2 & 0xF) << 4);
        }
    }
}

fn dequant_q5_k(quant: &[u8], block: &mut [f32]) {
    let d = load_f16(&quant[0..2]);
    let min = load_f16(&quant[2..4]);
    let scales = &quant[4..16];
    let qh = &quant[16..48];
    for (pair, (out, ql)) in
        block.chunks_exact_mut(64).zip(quant[48..].chunks_exact(32)).enumerate()
    {
        let (sc1, m1) = scale_min_k4(scales, 2 * pair);
        let (sc2, m2) = scale_min_k4(scales, 2 * pair + 1);
        for (l, &q) in ql.iter().enumerate() {
            let h1 = (qh[l] >> (2 * pair)) & 1;
            let h2 = (qh[l] >> (2 * pair + 1)) & 1;
            out[l] = d * sc1 as f32 * ((q & 0xF) | (h1 << 4)) as f32 - min * m1 as f32;
            out[l + 32] = d * sc2 as f32 * ((q >> 4) | (h2 << 4)) as f32 - min * m2 as f32;
        }
    }
}

fn quant_q6_k(x: &[f32], quant: &mut [u8]) {
    quant.fill(0);
    let mut levels = [0i8; QK_K];
    let mut scales = [0.0f32; 16];
    let mut max_scale = 0.0f32;
    let mut max_abs_scale = 0.0f32;
    for ib in 0..16 {
        let scale = make_qx_quants(&x[16 * ib..][..16], &mut levels[16 * ib..][..16], 32);
        scales[ib] = scale;
        if scale.abs() > max_abs_scale {
            max_abs_scale = scale.abs();
            max_scale = scale;
        }
    }
    if max_abs_scale < GROUP_MAX_EPS {
        return;
    }
    let iscale = -128.0 / max_scale;
    store_f16(&mut quant[208..210], 1.0 / iscale);
    for ib in 0..16 {
        quant[192 + ib] = nearest_int(iscale * scales[ib]).min(127) as i8 as u8;
    }
    let d_super = load_f16(&quant[208..210]);
    for j in 0..16 {
        let d = d_super * (quant[192 + j] as i8) as f32;
        if d == 0.0 {
            continue;
        }
        for i in 16 * j..16 * j + 16 {
            levels[i] = (nearest_int(x[i] / d).clamp(-32, 31) + 32) as i8;
        }
    }
    let (ql, rest) = quant.split_at_mut(128);
    for (half, levels) in levels.chunks_exact(128).enumerate() {
        let ql = &mut ql[64 * half..][..64];
        let qh = &mut rest[32 * half..][..32];
        for i in 0..32 {
            let l = [0, 32, 64, 96].map(|o| levels[i + o] as u8);
            ql[i] = (l[0] & 0xF) | ((l[2] & 0xF) << 4);
            ql[i + 32] = (l[1] & 0xF) | ((l[3] & 0xF) << 4);
            qh[i] = (l[0] >> 4) | ((l[1] >> 4) << 2) | ((l[2] >> 4) << 4) | ((l[3] >> 4) << 6);
        }
    }
}

fn dequant_q6_k(quant: &[u8], block: &mut [f32]) {
    let d = load_f16(&quant[208..210]);
    for (half, out) in block.chunks_exact_mut(128).enumerate() {
        let ql = &quant[64 * half..][..64];
        let qh = &quant[128 + 32 * half..][..32];
        let sc = &quant[192 + 8 * half..][..8];
        for l in 0..32 {
            let is = l / 16;
            let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i8 - 32;
            let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i8 - 32;
            let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8 - 32;
            let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i8 - 32;
            let s = |off: usize| (sc[is + off] as i8) as f32;
            out[l] = d * s(0) * q1 as f32;
            out[l + 32] = d * s(2) * q2 as f32;
            out[l + 64] = d * s(4) * q3 as f32;
            out[l + 96] = d * s(6) * q4 as f32;
        }
    }
}

impl_opaque_block_quant!(Q4KF, "Q4_K", 256, 144, quant_q4_k, dequant_q4_k);
impl_opaque_block_quant!(Q5KF, "Q5_K", 256, 176, quant_q5_k, dequant_q5_k);
impl_opaque_block_quant!(Q6KF, "Q6_K", 256, 210, quant_q6_k, dequant_q6_k);

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    /// A ramp over `[-2, 2]` with its first `zeros` values set to zero.
    pub(in super::super) fn ramp(len: usize, zeros: usize) -> Vec<f32> {
        (0..len)
            .map(|i| if i < zeros { 0.0 } else { (((i * 17 + 3) % 41) as f32 - 20.0) / 10.0 })
            .collect()
    }

    /// Checks `fmt` quantizes `data` to the hex `bytes` llama.cpp produces, and that
    /// dequantizing gives llama.cpp's f32 bits at each `(index, bits)` of `values`.
    pub(in super::super) fn assert_ggml_ref(
        fmt: &dyn BlockQuant,
        data: &[f32],
        bytes: &str,
        values: &[(usize, u32)],
    ) {
        let mut q = vec![0u8; fmt.block_bytes()];
        fmt.quant_block_f32(data, &mut q);
        let got: String = q.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(got, bytes, "{fmt} bytes");
        let mut back = vec![0f32; fmt.block_len()];
        fmt.dequant_block_f32(&q, &mut back);
        for &(i, bits) in values {
            assert_eq!(back[i].to_bits(), bits, "{fmt} value {i}");
        }
    }

    /// Quantizes two blocks of ramp and a zero block, and bounds the dequant error.
    pub(in super::super) fn assert_roundtrip(fmt: &dyn BlockQuant, max_err: f32) {
        let data = ramp(2 * fmt.block_len(), 0);
        let back = fmt.dequant_f32(&fmt.quant_f32(&data).unwrap()).unwrap();
        let back = back.try_as_plain_ram().unwrap();
        let back = back.as_slice::<f32>().unwrap();
        let err = data.iter().zip(back).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(err < max_err, "{fmt} max abs err {err}");
        let zeros = vec![0f32; fmt.block_len()];
        let back = fmt.dequant_f32(&fmt.quant_f32(&zeros).unwrap()).unwrap();
        assert!(
            back.try_as_plain_ram().unwrap().as_slice::<f32>().unwrap().iter().all(|v| *v == 0.0)
        );
    }

    #[test]
    fn q4_k_roundtrip() {
        assert_roundtrip(&Q4_K, 0.35);
    }

    /// Bytes and values from llama.cpp `quantize_row_q4_K_ref` / `dequantize_row_q4_K`.
    #[test]
    fn q4_k_matches_ggml_ref() {
        assert_ggml_ref(
            &Q4_K,
            &ramp(256, 0),
            "4e1c2028fefefffcfbfefdfeddbdfede51c73e950b72d84fa61c83ea50c72d94fb62d83fa51c73e950b72d84fa61c83ee950b72d94fa61d83ea50c73e940b62d84fa61c73ea50b72d94fb61c83fa51c762d94fa61d73ea51b72e95fb62c83fa60c73ea40b72e84fb51c83f950c73d940fb62c83f950c73d940b71d84ea51c83e950c62d94fa61d84ea51b72e95fb62d9",
            &[(0, 0xbfd2_0380), (32, 0xbf31_e300), (255, 0x3fb6_1280)],
        );
        assert_ggml_ref(
            &Q4_K,
            &ramp(256, 32),
            "4e1c2028c0fefffcc0fefdfeddbdfede50c030900070d040a01080e050c02090f060d030a01070e050b02080f060c030e950b72d94fa61d83ea50c73e940b62d84fa61c73ea50b72d94fb61c83fa51c762d94fa61d73ea51b72e95fb62c83fa60c73ea40b72e84fb51c83f950c73d940fb62c83f950c73d940b71d84ea51c83e950c62d94fa61d84ea51b72e95fb62d9",
            &[(0, 0), (32, 0xbf31_e300), (255, 0x3fb6_1280)],
        );
        assert_ggml_ref(&Q4_K, &[0f32; 256], &"00".repeat(144), &[(0, 0)]);
    }

    #[test]
    fn q5_k_roundtrip() {
        assert_roundtrip(&Q5_K, 0.2);
    }

    /// Bytes and values from llama.cpp `quantize_row_q5_K_ref` / `dequantize_row_q5_K`.
    #[test]
    fn q5_k_matches_ggml_ref() {
        assert_ggml_ref(
            &Q5_K,
            &ramp(256, 0),
            "41181d28fcfefffdfbfefefefecdfeedcc32d966994c33cd269964b3cc32d9669b4c33c9669964b3cc3699669bcc33c9b28f5d3a07d4a28f5c2af6d4a17e5c29f6c3a17e4b19f6c3906e4b18e6c2906ec3917e4b18f5c3906d4b18e5c2906d3a18e5b28f5c3a07d5b28f5c2907d4a18ed5a27f5d2af7d4a27f4c2af7c4917e4c19e6c4916e3b19e6b3906e3b08e6b380f6c3906e4b18e6c3906d3b18e5b3805d3a07d5b28f5d2a07d4a28f5c2af7d4a1",
            &[(0, 0xbfd2_c780), (32, 0xbf13_66c0), (255, 0x3fa6_a520)],
        );
        assert_ggml_ref(
            &Q5_K,
            &ramp(256, 32),
            "41181d28c0fefffdc0fefefefecdfeedcc32d866984c32cc269864b2cc32d8669a4c32c8669864b2cc3698669acc32c8b080503000d0a0805020f0d0a0705020f0c0a0704010f0c090604010e0c09060c3917e4b18f5c3906d4b18e5c2906d3a18e5b28f5c3a07d5b28f5c2907d4a18ed5a27f5d2af7d4a27f4c2af7c4917e4c19e6c4916e3b19e6b3906e3b08e6b380f6c3906e4b18e6c3906d3b18e5b3805d3a07d5b28f5d2a07d4a28f5c2af7d4a1",
            &[(0, 0), (32, 0xbf13_66c0), (255, 0x3fa6_a520)],
        );
        assert_ggml_ref(&Q5_K, &[0f32; 256], &"00".repeat(176), &[(0, 0)]);
    }

    #[test]
    fn q6_k_roundtrip() {
        assert_roundtrip(&Q6_K, 0.1);
    }

    /// Bytes and values from llama.cpp `quantize_row_q6_K_ref` / `dequantize_row_q6_K`.
    #[test]
    fn q6_k_matches_ggml_ref() {
        assert_ggml_ref(
            &Q6_K,
            &ramp(256, 0),
            "8b10d58b10d66b20d66b22b66d02b67d0159bd0158bc0468bf0367bf236ade2266d21b96d01b86c04b85f03b75f03aa520d66b22b66d02b66d01b85d01b85d03d76b20d66b22b66d02b66d01b85d01b8b30d68b11d68c11d76c22d76c22b76d0158ac0458bf0357bf036abe036abe2566e12b66d11b96d01a85cf4a85ff3a74f278e50fa51278d24fa50fb8d278e50fab1db71ec06ac1b71db46ac06b1db71ec2b812cf65cfb812b865cf6512b812cf6af1872d845af05b2d872af05afd872d87c78847b857c79847b857c78847f79802290",
            &[(0, 0xbfd8_32a0), (16, 0x3f68_7880), (255, 0x3fa5_5000)],
        );
        assert_ggml_ref(
            &Q6_K,
            &ramp(256, 16),
            "8010d08010d06020d06020b06000b0700159bd0158bc0468bf0367bf236ade2266d21b96d01b86c04b85f03b75f03aa520d66b22b66d02b66d01b85d01b85d03d76b20d66b22b66d02b66d01b85d01b8b30d68b11d68c11d76c22d76c22b76d0158ac0458bf0357bf036abe036abe2566e12b66d11b96d01a85cf4a85ff3a74f248c50f850248c24f850f88c248c50f8b1db71ec06ac1b71db46ac06b1db71ec2b812cf65cfb812b865cf6512b812cf6af1872d845af05b2d872af05afd872d80078847b857c79847b857c78847f79802290",
            &[(0, 0), (16, 0x3f68_7880), (255, 0x3fa5_5000)],
        );
        assert_ggml_ref(&Q6_K, &[0f32; 256], &"00".repeat(210), &[]);
    }
}
