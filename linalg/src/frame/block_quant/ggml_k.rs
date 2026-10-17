//! Scalar ggml K-quant and Q8_0 blocks, laid out like llama.cpp's `block_q*_K` /
//! `block_q8_0`. The quantizers follow `quantize_row_*_ref` (no importance matrix).
use super::*;

const QK_K: usize = 256;
const GROUP_MAX_EPS: f32 = 1e-15;

pub const Q8_0: Q8_0F = Q8_0F;
pub const Q3_K: Q3KF = Q3KF;
pub const Q4_K: Q4KF = Q4KF;
pub const Q5_K: Q5KF = Q5KF;
pub const Q6_K: Q6KF = Q6KF;

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct Q8_0F;
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct Q3KF;
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct Q4KF;
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct Q5KF;
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct Q6KF;

fn store_f16(dst: &mut [u8], v: f32) {
    dst[..2].copy_from_slice(&f16::from_f32(v).to_bits().to_le_bytes());
}

fn load_f16(src: &[u8]) -> f32 {
    f16::from_bits(u16::from_le_bytes([src[0], src[1]])).to_f32()
}

/// ggml `nearest_int`: round-to-nearest-even via the 2^23 mantissa trick.
fn nearest_int(fval: f32) -> i32 {
    let val = fval + 12582912.0;
    let i = val.to_bits() as i32;
    (i & 0x007fffff) - 0x00400000
}

fn make_qx_quants(x: &[f32], l_out: &mut [i8], nmax: i32, rmse_type: i32) -> f32 {
    let n = x.len();
    let mut max = 0.0f32;
    let mut amax = 0.0f32;
    for &xi in x {
        let ax = xi.abs();
        if ax > amax {
            amax = ax;
            max = xi;
        }
    }
    if amax < GROUP_MAX_EPS {
        l_out.fill(0);
        return 0.0;
    }
    let mut iscale = -nmax as f32 / max;
    if rmse_type == 0 {
        for i in 0..n {
            let l = nearest_int(iscale * x[i]);
            l_out[i] = (nmax + l.max(-nmax).min(nmax - 1)) as i8;
        }
        return 1.0 / iscale;
    }
    let mut rmse_type = rmse_type;
    let mut return_early = false;
    if rmse_type < 0 {
        rmse_type = -rmse_type;
        return_early = true;
    }
    let weight = |xi: f32| match rmse_type {
        1 => xi * xi,
        2 => 1.0,
        3 => xi.abs(),
        _ => xi.abs().sqrt(),
    };
    let mut sumlx = 0.0f32;
    let mut suml2 = 0.0f32;
    for i in 0..n {
        let l = nearest_int(iscale * x[i]).max(-nmax).min(nmax - 1);
        l_out[i] = (l + nmax) as i8;
        let w = weight(x[i]);
        sumlx += w * x[i] * l as f32;
        suml2 += w * (l * l) as f32;
    }
    let mut scale = if suml2 != 0.0 { sumlx / suml2 } else { 0.0 };
    if return_early {
        return if suml2 > 0.0 { 0.5 * (scale + 1.0 / iscale) } else { 1.0 / iscale };
    }
    let mut best = scale * sumlx;
    for is in -9..=9 {
        if is == 0 {
            continue;
        }
        iscale = -(nmax as f32 + 0.1 * is as f32) / max;
        sumlx = 0.0;
        suml2 = 0.0;
        for &xi in x.iter().take(n) {
            let l = nearest_int(iscale * xi).max(-nmax).min(nmax - 1);
            let w = weight(xi);
            sumlx += w * xi * l as f32;
            suml2 += w * (l * l) as f32;
        }
        if suml2 > 0.0 && sumlx * sumlx > best * suml2 {
            for i in 0..n {
                let l = nearest_int(iscale * x[i]).max(-nmax).min(nmax - 1);
                l_out[i] = (nmax + l) as i8;
            }
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
        if x[i] < min {
            min = x[i];
        }
        if x[i] > max {
            max = x[i];
        }
        let w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
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
        let l = nearest_int(iscale * (x[i] - min)).max(0).min(nmax);
        l_out[i] = l as u8;
        let diff = scale * l as f32 + min - x[i];
        best_error += weights[i] * diff * diff;
    }
    if nstep < 1 {
        *the_min = -min;
        return scale;
    }
    let mut laux = [0u8; 32];
    for is in 0..=nstep {
        iscale = (rmin + rdelta * is as f32 + nmax as f32) / (max - min);
        let mut sum_l = 0.0f32;
        let mut sum_l2 = 0.0f32;
        let mut sum_xl = 0.0f32;
        for i in 0..n {
            let l = nearest_int(iscale * (x[i] - min)).max(0).min(nmax);
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

fn scale_min_k4(q: &[u8], j: usize) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        let d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

fn quant_q8_0(block: &[f32], quant: &mut [u8]) {
    let mut amax = 0.0f32;
    for &v in block {
        amax = amax.max(v.abs());
    }
    let d = amax / 127.0;
    let id = if d == 0.0 { 0.0 } else { 1.0 / d };
    store_f16(&mut quant[..2], d);
    for (j, &v) in block.iter().enumerate() {
        // ggml's q8_0 ref uses roundf, not nearest_int.
        quant[2 + j] = (v * id).round() as i8 as u8;
    }
}

fn dequant_q8_0(quant: &[u8], block: &mut [f32]) {
    let d = load_f16(quant);
    for (j, dst) in block.iter_mut().enumerate() {
        *dst = (quant[2 + j] as i8) as f32 * d;
    }
}

fn quant_q4_k(block: &[f32], quant: &mut [u8]) {
    quant.fill(0);
    let mut l = [0u8; QK_K];
    let mut weights = [0.0f32; 32];
    let mut mins = [0.0f32; 8];
    let mut scales = [0.0f32; 8];
    let mut max_scale = 0.0f32;
    let mut max_min = 0.0f32;
    for j in 0..8 {
        let x = &block[32 * j..32 * j + 32];
        let mut sum_x2 = 0.0f32;
        for &v in x {
            sum_x2 += v * v;
        }
        let av_x = (sum_x2 / 32.0).sqrt();
        for i in 0..32 {
            weights[i] = av_x + x[i].abs();
        }
        scales[j] = make_qkx2_quants(
            x,
            &weights,
            &mut l[32 * j..32 * j + 32],
            &mut mins[j],
            15,
            -1.0,
            0.1,
            20,
        );
        max_scale = max_scale.max(scales[j]);
        max_min = max_min.max(mins[j]);
    }
    let inv_scale = if max_scale > 0.0 { 63.0 / max_scale } else { 0.0 };
    let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };
    let sc = &mut quant[4..16];
    for j in 0..8 {
        let mut ls = nearest_int(inv_scale * scales[j]) as u8;
        let mut lm = nearest_int(inv_min * mins[j]) as u8;
        ls = ls.min(63);
        lm = lm.min(63);
        if j < 4 {
            sc[j] = ls;
            sc[j + 4] = lm;
        } else {
            sc[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
            sc[j - 4] |= (ls >> 4) << 6;
            sc[j] |= (lm >> 4) << 6;
        }
    }
    store_f16(&mut quant[0..2], max_scale / 63.0);
    store_f16(&mut quant[2..4], max_min / 63.0);
    let d_super = load_f16(&quant[0..2]);
    let m_super = load_f16(&quant[2..4]);
    for j in 0..8 {
        let (scj, mj) = scale_min_k4(&quant[4..16], j);
        let d = d_super * scj as f32;
        if d == 0.0 {
            continue;
        }
        let dm = m_super * mj as f32;
        for ii in 0..32 {
            let mut lv = nearest_int((block[32 * j + ii] + dm) / d);
            lv = lv.clamp(0, 15);
            l[32 * j + ii] = lv as u8;
        }
    }
    let qs = &mut quant[16..144];
    let mut q = 0;
    let mut j = 0;
    while j < QK_K {
        for ii in 0..32 {
            qs[q + ii] = l[j + ii] | (l[j + ii + 32] << 4);
        }
        q += 32;
        j += 64;
    }
}

fn dequant_q4_k(quant: &[u8], block: &mut [f32]) {
    let d = load_f16(&quant[0..2]);
    let min = load_f16(&quant[2..4]);
    let scales = &quant[4..16];
    let mut q = 16;
    let mut y = 0;
    let mut is = 0;
    while y < QK_K {
        let (sc, m) = scale_min_k4(scales, is);
        let d1 = d * sc as f32;
        let m1 = min * m as f32;
        let (sc, m) = scale_min_k4(scales, is + 1);
        let d2 = d * sc as f32;
        let m2 = min * m as f32;
        for l in 0..32 {
            block[y] = d1 * (quant[q + l] & 0xF) as f32 - m1;
            y += 1;
        }
        for l in 0..32 {
            block[y] = d2 * (quant[q + l] >> 4) as f32 - m2;
            y += 1;
        }
        q += 32;
        is += 2;
    }
}

fn quant_q5_k(block: &[f32], quant: &mut [u8]) {
    quant.fill(0);
    let mut l = [0u8; QK_K];
    let mut weights = [0.0f32; 32];
    let mut mins = [0.0f32; 8];
    let mut scales = [0.0f32; 8];
    let mut max_scale = 0.0f32;
    let mut max_min = 0.0f32;
    for j in 0..8 {
        let x = &block[32 * j..32 * j + 32];
        let mut sum_x2 = 0.0f32;
        for &v in x {
            sum_x2 += v * v;
        }
        let av_x = (sum_x2 / 32.0).sqrt();
        for i in 0..32 {
            weights[i] = av_x + x[i].abs();
        }
        scales[j] = make_qkx2_quants(
            x,
            &weights,
            &mut l[32 * j..32 * j + 32],
            &mut mins[j],
            31,
            -0.5,
            0.1,
            15,
        );
        max_scale = max_scale.max(scales[j]);
        max_min = max_min.max(mins[j]);
    }
    let inv_scale = if max_scale > 0.0 { 63.0 / max_scale } else { 0.0 };
    let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };
    let sc = &mut quant[4..16];
    for j in 0..8 {
        let mut ls = nearest_int(inv_scale * scales[j]) as u8;
        let mut lm = nearest_int(inv_min * mins[j]) as u8;
        ls = ls.min(63);
        lm = lm.min(63);
        if j < 4 {
            sc[j] = ls;
            sc[j + 4] = lm;
        } else {
            sc[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
            sc[j - 4] |= (ls >> 4) << 6;
            sc[j] |= (lm >> 4) << 6;
        }
    }
    store_f16(&mut quant[0..2], max_scale / 63.0);
    store_f16(&mut quant[2..4], max_min / 63.0);
    let d_super = load_f16(&quant[0..2]);
    let m_super = load_f16(&quant[2..4]);
    for j in 0..8 {
        let (scj, mj) = scale_min_k4(&quant[4..16], j);
        let d = d_super * scj as f32;
        if d == 0.0 {
            continue;
        }
        let dm = m_super * mj as f32;
        for ii in 0..32 {
            let lv = nearest_int((block[32 * j + ii] + dm) / d).clamp(0, 31);
            l[32 * j + ii] = lv as u8;
        }
    }
    let mut m1 = 1u8;
    let mut m2 = 2u8;
    let mut ql = 0;
    let mut n = 0;
    while n < QK_K {
        for j in 0..32 {
            let mut l1 = l[n + j] as i32;
            if l1 > 15 {
                l1 -= 16;
                quant[16 + j] |= m1;
            }
            let mut l2 = l[n + j + 32] as i32;
            if l2 > 15 {
                l2 -= 16;
                quant[16 + j] |= m2;
            }
            quant[48 + ql + j] = l1 as u8 | ((l2 as u8) << 4);
        }
        m1 <<= 2;
        m2 <<= 2;
        ql += 32;
        n += 64;
    }
}

fn dequant_q5_k(quant: &[u8], block: &mut [f32]) {
    let d = load_f16(&quant[0..2]);
    let min = load_f16(&quant[2..4]);
    let scales = &quant[4..16];
    let qh = &quant[16..48];
    let mut ql = 48;
    let mut y = 0;
    let mut is = 0;
    let mut u1 = 1u8;
    let mut u2 = 2u8;
    while y < QK_K {
        let (sc, m) = scale_min_k4(scales, is);
        let d1 = d * sc as f32;
        let m1 = min * m as f32;
        let (sc, m) = scale_min_k4(scales, is + 1);
        let d2 = d * sc as f32;
        let m2 = min * m as f32;
        for l in 0..32 {
            let hi = if qh[l] & u1 != 0 { 16.0 } else { 0.0 };
            block[y] = d1 * ((quant[ql + l] & 0xF) as f32 + hi) - m1;
            y += 1;
        }
        for l in 0..32 {
            let hi = if qh[l] & u2 != 0 { 16.0 } else { 0.0 };
            block[y] = d2 * ((quant[ql + l] >> 4) as f32 + hi) - m2;
            y += 1;
        }
        ql += 32;
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
    }
}

fn quant_q6_k(block: &[f32], quant: &mut [u8]) {
    quant.fill(0);
    let mut l = [0i8; QK_K];
    let mut scales = [0.0f32; 16];
    let mut max_scale = 0.0f32;
    let mut max_abs_scale = 0.0f32;
    for ib in 0..16 {
        let scale =
            make_qx_quants(&block[16 * ib..16 * ib + 16], &mut l[16 * ib..16 * ib + 16], 32, 1);
        scales[ib] = scale;
        let abs_scale = scale.abs();
        if abs_scale > max_abs_scale {
            max_abs_scale = abs_scale;
            max_scale = scale;
        }
    }
    if max_abs_scale < GROUP_MAX_EPS {
        store_f16(&mut quant[208..210], 0.0);
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
        for ii in 0..16 {
            let lv = nearest_int(block[16 * j + ii] / d).clamp(-32, 31);
            l[16 * j + ii] = (lv + 32) as i8;
        }
    }
    let mut ql = 0;
    let mut qh = 128;
    let mut j = 0;
    while j < QK_K {
        for ii in 0..32 {
            let q1 = (l[j + ii] as u8) & 0xF;
            let q2 = (l[j + ii + 32] as u8) & 0xF;
            let q3 = (l[j + ii + 64] as u8) & 0xF;
            let q4 = (l[j + ii + 96] as u8) & 0xF;
            quant[ql + ii] = q1 | (q3 << 4);
            quant[ql + 32 + ii] = q2 | (q4 << 4);
            quant[qh + ii] = ((l[j + ii] as u8) >> 4)
                | (((l[j + ii + 32] as u8) >> 4) << 2)
                | (((l[j + ii + 64] as u8) >> 4) << 4)
                | (((l[j + ii + 96] as u8) >> 4) << 6);
        }
        ql += 64;
        qh += 32;
        j += 128;
    }
}

fn dequant_q6_k(quant: &[u8], block: &mut [f32]) {
    let d = load_f16(&quant[208..210]);
    let mut ql = 0;
    let mut qh = 128;
    let mut sc = 192;
    let mut y = 0;
    while y < QK_K {
        for l in 0..32 {
            let is = l / 16;
            let q1 = ((quant[ql + l] & 0xF) | ((quant[qh + l] & 3) << 4)) as i8 - 32;
            let q2 = ((quant[ql + 32 + l] & 0xF) | (((quant[qh + l] >> 2) & 3) << 4)) as i8 - 32;
            let q3 = ((quant[ql + l] >> 4) | (((quant[qh + l] >> 4) & 3) << 4)) as i8 - 32;
            let q4 = ((quant[ql + 32 + l] >> 4) | (((quant[qh + l] >> 6) & 3) << 4)) as i8 - 32;
            let s = |off: usize| (quant[sc + is + off] as i8) as f32;
            block[y + l] = d * s(0) * q1 as f32;
            block[y + 32 + l] = d * s(2) * q2 as f32;
            block[y + 64 + l] = d * s(4) * q3 as f32;
            block[y + 96 + l] = d * s(6) * q4 as f32;
        }
        y += 128;
        ql += 64;
        qh += 32;
        sc += 8;
    }
}

/// `make_q3_quants` with `do_rmse`: levels land in `0..=7` (`nmax` is 4).
fn make_q3_quants(x: &[f32], l_out: &mut [i8]) -> f32 {
    let n = x.len();
    let nmax = 4i32;
    let mut max = 0.0f32;
    let mut amax = 0.0f32;
    for &xi in x.iter().take(n) {
        let ax = xi.abs();
        if ax > amax {
            amax = ax;
            max = xi;
        }
    }
    if amax < GROUP_MAX_EPS {
        l_out[..n].fill(0);
        return 0.0;
    }
    let iscale = -(nmax as f32) / max;
    let mut sumlx = 0.0f32;
    let mut suml2 = 0.0f32;
    for i in 0..n {
        let l = nearest_int(iscale * x[i]).clamp(-nmax, nmax - 1);
        l_out[i] = l as i8;
        let w = x[i] * x[i];
        sumlx += w * x[i] * l as f32;
        suml2 += w * l as f32 * l as f32;
    }
    for _ in 0..5 {
        let mut n_changed = 0i32;
        for i in 0..n {
            let w = x[i] * x[i];
            let mut slx = sumlx - w * x[i] * l_out[i] as f32;
            if slx > 0.0 {
                let mut sl2 = suml2 - w * l_out[i] as f32 * l_out[i] as f32;
                let new_l = nearest_int(x[i] * sl2 / slx).clamp(-nmax, nmax - 1);
                if new_l != l_out[i] as i32 {
                    slx += w * x[i] * new_l as f32;
                    sl2 += w * new_l as f32 * new_l as f32;
                    if sl2 > 0.0 && slx * slx * suml2 > sumlx * sumlx * sl2 {
                        l_out[i] = new_l as i8;
                        sumlx = slx;
                        suml2 = sl2;
                        n_changed += 1;
                    }
                }
            }
        }
        if n_changed == 0 {
            break;
        }
    }
    for level in l_out.iter_mut().take(n) {
        *level += nmax as i8;
    }
    if suml2 > 0.0 { sumlx / suml2 } else { 0.0 }
}

/// 6-bit group scale packed the way `quantize_row_q3_K_ref` writes `scales[12]`.
fn q3_group_scale(scales: &[u8], group: usize) -> i32 {
    let lo = if group < 8 { scales[group] & 0xF } else { scales[group - 8] >> 4 };
    let hi = (scales[8 + group % 4] >> (2 * (group / 4))) & 3;
    i32::from(lo | (hi << 4)) - 32
}

fn quant_q3_k(x: &[f32], quant: &mut [u8]) {
    let mut levels = [0i8; QK_K];
    let mut scales_f = [0.0f32; 16];
    let mut max_scale = 0.0f32;
    let mut amax = 0.0f32;
    for j in 0..16 {
        scales_f[j] = make_q3_quants(&x[16 * j..16 * j + 16], &mut levels[16 * j..16 * j + 16]);
        let scale = scales_f[j].abs();
        if scale > amax {
            amax = scale;
            max_scale = scales_f[j];
        }
    }
    quant.fill(0);
    if max_scale != 0.0 {
        let iscale = -32.0 / max_scale;
        for j in 0..16 {
            let mut level = nearest_int(iscale * scales_f[j]).clamp(-32, 31) + 32;
            if j < 8 {
                quant[96 + j] = (level as u8) & 0xF;
            } else {
                quant[96 + j - 8] |= ((level as u8) & 0xF) << 4;
            }
            level >>= 4;
            let shift = (2 * (j / 4)) as u32;
            quant[96 + 8 + (j % 4)] |= (level as u8) << shift;
        }
        store_f16(&mut quant[108..110], 1.0 / iscale);
    }
    let d_all = load_f16(&quant[108..110]);
    for j in 0..16 {
        let d = d_all * q3_group_scale(&quant[96..108], j) as f32;
        if d == 0.0 {
            continue;
        }
        for ii in 0..16 {
            let level = nearest_int(x[16 * j + ii] / d).clamp(-4, 3);
            levels[16 * j + ii] = (level + 4) as i8;
        }
    }
    // High bit of element i is hmask[i % 32] bit (i / 32). qs keeps the low two bits.
    let mut m = 0usize;
    let mut hm = 1u8;
    for level in &mut levels {
        if *level > 3 {
            quant[m] |= hm;
            *level -= 4;
        }
        m += 1;
        if m == 32 {
            m = 0;
            hm <<= 1;
        }
    }
    for half in [0usize, 128] {
        for lane in 0..32 {
            quant[32 + half / 4 + lane] = levels[half + lane] as u8
                | ((levels[half + lane + 32] as u8) << 2)
                | ((levels[half + lane + 64] as u8) << 4)
                | ((levels[half + lane + 96] as u8) << 6);
        }
    }
}

fn dequant_q3_k(quant: &[u8], block: &mut [f32]) {
    let d_all = load_f16(&quant[108..110]);
    let hmask = &quant[..32];
    let qs = &quant[32..96];
    let scales = &quant[96..108];
    for (idx, out) in block.iter_mut().enumerate() {
        let half = idx / 128;
        let within = idx % 128;
        let group_in_half = within / 16;
        let pair = group_in_half / 2;
        let lane = (within % 16) + (group_in_half % 2) * 16;
        let shift = (pair * 2) as u32;
        let hbit = (pair + half * 4) as u32;
        let low = (qs[half * 32 + lane] >> shift) & 3;
        let high = (hmask[lane] >> hbit) & 1;
        let qv = i32::from(low) - if high == 1 { 0 } else { 4 };
        let scale = q3_group_scale(scales, idx / 16);
        *out = d_all * scale as f32 * qv as f32;
    }
}

fn quant_f16_via_f32(block: &[f16], quant: &mut [u8], quant_f32: impl FnOnce(&[f32], &mut [u8])) {
    let tmp: Vec<f32> = block.iter().copied().map(|v| v.to_f32()).collect();
    quant_f32(&tmp, quant);
}

fn dequant_f16_via_f32(
    quant: &[u8],
    block: &mut [f16],
    dequant_f32: impl FnOnce(&[u8], &mut [f32]),
) {
    let mut tmp = vec![0.0f32; block.len()];
    dequant_f32(quant, &mut tmp);
    for (dst, src) in block.iter_mut().zip(tmp) {
        *dst = f16::from_f32(src);
    }
}

macro_rules! impl_kquant {
    ($ty:ty, $name:literal, $len:expr, $bytes:expr, $q:path, $dq:path) => {
        impl Display for $ty {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str($name)
            }
        }

        impl BlockQuant for $ty {
            fn block_len(&self) -> usize {
                $len
            }

            fn block_bytes(&self) -> usize {
                $bytes
            }

            fn quant_block_f32(&self, block: &[f32], quant: &mut [u8]) {
                assert_eq!(block.len(), $len);
                assert_eq!(quant.len(), $bytes);
                $q(block, quant);
            }

            fn quant_block_f16(&self, block: &[f16], quant: &mut [u8]) {
                quant_f16_via_f32(block, quant, $q);
            }

            fn dequant_block_f32(&self, quant: &[u8], block: &mut [f32]) {
                assert_eq!(block.len(), $len);
                assert_eq!(quant.len(), $bytes);
                $dq(quant, block);
            }

            fn dequant_block_f16(&self, quant: &[u8], block: &mut [f16]) {
                dequant_f16_via_f32(quant, block, $dq);
            }

            fn pack(
                &self,
                input: &[u8],
                k: usize,
                r: usize,
                zip: usize,
                scales_at_end: bool,
            ) -> TractResult<EagerPackedInput> {
                super::opaque::pack_opaque(self, input, k, r, zip, scales_at_end)
            }

            unsafe fn extract_packed_panel(
                &self,
                value: &EagerPackedInput,
                target: &PackedFormat,
                panel: usize,
                scratch: *mut u8,
            ) -> TractResult<()> {
                unsafe { super::opaque::extract_packed_panel(self, value, target, panel, scratch) }
            }

            fn extract_at_mn_f16(
                &self,
                value: &EagerPackedInput,
                mn: usize,
                target: &mut [f16],
            ) -> TractResult<()> {
                super::opaque::extract_at_mn_f16(self, value, mn, target)
            }

            fn extract_at_mn_f32(
                &self,
                value: &EagerPackedInput,
                mn: usize,
                target: &mut [f32],
            ) -> TractResult<()> {
                super::opaque::extract_at_mn_f32(self, value, mn, target)
            }
        }
    };
}

impl_kquant!(Q8_0F, "Q8_0", 32, 34, quant_q8_0, dequant_q8_0);
impl_kquant!(Q3KF, "Q3_K", 256, 110, quant_q3_k, dequant_q3_k);
impl_kquant!(Q4KF, "Q4_K", 256, 144, quant_q4_k, dequant_q4_k);
impl_kquant!(Q5KF, "Q5_K", 256, 176, quant_q5_k, dequant_q5_k);
impl_kquant!(Q6KF, "Q6_K", 256, 210, quant_q6_k, dequant_q6_k);

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(fmt: &dyn BlockQuant, n: usize) {
        let data: Vec<f32> = (0..n).map(|i| (((i * 17 + 3) % 41) as f32 - 20.0) / 10.0).collect();
        let q = fmt.quant_f32(&data).unwrap();
        let back = fmt.dequant_f32(&q).unwrap();
        let got = back.try_as_plain_ram().unwrap().as_slice::<f32>().unwrap();
        let err = data.iter().zip(got).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(err < 0.35, "{fmt} max abs err {err}");
        let zeros = vec![0.0f32; fmt.block_len()];
        let qz = fmt.quant_f32(&zeros).unwrap();
        let bz = fmt.dequant_f32(&qz).unwrap();
        assert!(
            bz.try_as_plain_ram().unwrap().as_slice::<f32>().unwrap().iter().all(|v| *v == 0.0),
            "{fmt} zero block"
        );
    }

    #[test]
    fn q8_0_roundtrip() {
        roundtrip(&Q8_0, 64);
    }

    #[test]
    fn q3_k_roundtrip() {
        let fmt: &dyn BlockQuant = &Q3_K;
        let data: Vec<f32> = (0..256).map(|i| (((i * 17 + 3) % 41) as f32 - 20.0) / 10.0).collect();
        let q = fmt.quant_f32(&data).unwrap();
        let back = fmt.dequant_f32(&q).unwrap();
        let got = back.try_as_plain_ram().unwrap().as_slice::<f32>().unwrap();
        let err = data.iter().zip(got).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        // ggml's own error on this ramp is about 0.42. A dropped scale or hmask is far larger.
        assert!(err < 0.5, "{fmt} max abs err {err}");
        let zeros = vec![0.0f32; fmt.block_len()];
        let qz = fmt.quant_f32(&zeros).unwrap();
        let bz = fmt.dequant_f32(&qz).unwrap();
        assert!(
            bz.try_as_plain_ram().unwrap().as_slice::<f32>().unwrap().iter().all(|v| *v == 0.0),
            "{fmt} zero block"
        );
    }

    /// Bytes from llama.cpp `quantize_row_q3_K_ref` / `dequantize_row_q3_K` (no imatrix).
    #[test]
    fn q3_k_matches_ggml_ref() {
        let pat: Vec<f32> = (0..256).map(|i| (((i * 17 + 3) % 41) as f32 - 20.0) / 10.0).collect();
        let hole: Vec<f32> =
            pat.iter().enumerate().map(|(i, v)| if i < 16 { 0.0 } else { *v }).collect();
        let zero = vec![0.0f32; 256];
        assert_q3_bytes(
            &pat,
            "758b64df28f58a749f20df8a759b60dffc2b55ae11fe23d4ab51fe03fcab55ee5f8885f5222f5f88b5f6e21f5f88b5f622f7f4889d5e22f7fb885d6222f7f888175788bdfde2275788bdfe22175488fd5e22f5f8885f5222f5f88b5f6ee1f588df2de2ee03efde0033cffc335fa4",
        );
        assert_q3_bytes(
            &hole,
            "748a64de28f48a749e20de8a749a60defc2b55ae11fe23d4ab51fe03fcab55ee5c8884f4202c5c88b4f4e01c5c88b4f422f7f4889d5e22f7fb885d6222f7f888175788bdfde2275788bdfe22175488fd5e22f5f8885f5222f5f88b5f6ee1f588d02de2ee03efde0032cffc335fa4",
        );
        assert_q3_bytes(
            &zero,
            "0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        );
        let mut back = [0.0f32; 256];
        let mut q = [0u8; 110];
        Q3_K.quant_block_f32(&pat, &mut q);
        Q3_K.dequant_block_f32(&q, &mut back);
        assert_eq!(back[0].to_bits(), 0xbfcb_4180);
        assert_eq!(back[16].to_bits(), 0x3f7d_8600);
        assert_eq!(back[255].to_bits(), 0x3f8b_e000);
        Q3_K.quant_block_f32(&hole, &mut q);
        Q3_K.dequant_block_f32(&q, &mut back);
        assert_eq!(back[0].to_bits(), 0);
        assert_eq!(back[16].to_bits(), 0x3f7d_8600);
    }

    fn assert_q3_bytes(data: &[f32], expect: &str) {
        let mut q = [0u8; 110];
        Q3_K.quant_block_f32(data, &mut q);
        let got: String = q.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(got, expect);
    }

    #[test]
    fn q4_k_roundtrip() {
        roundtrip(&Q4_K, 256);
    }

    #[test]
    fn q5_k_roundtrip() {
        roundtrip(&Q5_K, 256);
    }

    #[test]
    fn q6_k_roundtrip() {
        roundtrip(&Q6_K, 256);
    }
}
