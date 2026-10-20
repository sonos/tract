//! ggml IQ4_XS: 256-value super-blocks of the non-linear 4-bit `iq4nl` codebook.
//!
//! The quantizer is llama.cpp `quantize_row_iq4_xs_ref`, without importance matrix.
use super::ggml_k::GROUP_MAX_EPS;
use super::ggml_k::QK_K;
use super::ggml_k::nearest_int;
use super::opaque::impl_opaque_block_quant;
use super::opaque::load_f16;
use super::opaque::store_f16;
use super::*;

/// ggml IQ4_XS block: f16 `d`, eight 6-bit group scales split into 2 high-bit and
/// 4 low-nibble bytes, then 256 4-bit codebook indices, 136 bytes.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct IQ4XSF;

/// The ggml IQ4_XS block format.
pub const IQ4_XS: IQ4XSF = IQ4XSF;

/// ggml `kvalues_iq4nl`.
const IQ4NL: [i8; 16] = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];

fn best_index_iq4nl(x: f32) -> u8 {
    if x <= IQ4NL[0] as f32 {
        return 0;
    }
    if x >= IQ4NL[15] as f32 {
        return 15;
    }
    let mut ml = 0usize;
    let mut mu = 15usize;
    while mu - ml > 1 {
        let mav = (ml + mu) / 2;
        if x < IQ4NL[mav] as f32 {
            mu = mav;
        } else {
            ml = mav;
        }
    }
    if x - (IQ4NL[mu - 1] as f32) < (IQ4NL[mu] as f32) - x { (mu - 1) as u8 } else { mu as u8 }
}

/// Scale of one 32-value group, weighting each value by its square.
fn group_scale(xb: &[f32]) -> f32 {
    let mut amax = 0.0f32;
    let mut max = 0.0f32;
    for &xi in xb {
        if xi.abs() > amax {
            amax = xi.abs();
            max = xi;
        }
    }
    if amax < GROUP_MAX_EPS {
        return 0.0;
    }
    let fit = |id: f32| {
        let mut sumqx = 0.0f32;
        let mut sumq2 = 0.0f32;
        for &xi in xb {
            let q = IQ4NL[best_index_iq4nl(id * xi) as usize] as f32;
            let w = xi * xi;
            sumqx += w * q * xi;
            sumq2 += w * q * q;
        }
        (sumqx, sumq2)
    };
    let (sumqx, sumq2) = fit(-(IQ4NL[0] as f32) / max);
    let mut d = if sumq2 > 0.0 { sumqx / sumq2 } else { 0.0 };
    let mut best = d * sumqx;
    for itry in -7..=7 {
        let (sumqx, sumq2) = fit((itry + i32::from(IQ4NL[0])) as f32 / max);
        if sumq2 > 0.0 && sumqx * sumqx > best * sumq2 {
            d = sumqx / sumq2;
            best = d * sumqx;
        }
    }
    d
}

fn quant_iq4_xs(x: &[f32], quant: &mut [u8]) {
    let mut scales = [0.0f32; 8];
    let mut max_scale = 0.0f32;
    let mut amax_scale = 0.0f32;
    for ib in 0..8 {
        scales[ib] = group_scale(&x[32 * ib..][..32]);
        if scales[ib].abs() > amax_scale {
            amax_scale = scales[ib].abs();
            max_scale = scales[ib];
        }
    }
    quant.fill(0);
    let d_super = -max_scale / 32.0;
    store_f16(&mut quant[0..2], d_super);
    let id_super = if d_super != 0.0 { 1.0 / d_super } else { 0.0 };
    let mut levels = [0u8; QK_K];
    let mut scales_h = 0u16;
    for ib in 0..8 {
        let level = nearest_int(id_super * scales[ib]).clamp(-32, 31);
        let dl = d_super * level as f32;
        let idl = if dl != 0.0 { 1.0 / dl } else { 0.0 };
        for (l, &xi) in levels[32 * ib..][..32].iter_mut().zip(&x[32 * ib..][..32]) {
            *l = best_index_iq4nl(idl * xi);
        }
        let level = (level + 32) as u8;
        quant[4 + ib / 2] |= (level & 0xF) << (4 * (ib % 2));
        scales_h |= u16::from(level >> 4) << (2 * ib);
    }
    quant[2..4].copy_from_slice(&scales_h.to_le_bytes());
    for (levels, qs) in levels.chunks_exact(32).zip(quant[8..].chunks_exact_mut(16)) {
        for j in 0..16 {
            qs[j] = levels[j] | (levels[16 + j] << 4);
        }
    }
}

fn dequant_iq4_xs(quant: &[u8], block: &mut [f32]) {
    let d = load_f16(&quant[0..2]);
    let scales_h = u16::from_le_bytes([quant[2], quant[3]]);
    for (ib, (out, qs)) in block.chunks_exact_mut(32).zip(quant[8..].chunks_exact(16)).enumerate() {
        let lo = (quant[4 + ib / 2] >> (4 * (ib % 2))) & 0xF;
        let hi = ((scales_h >> (2 * ib)) & 3) as u8;
        let dl = d * (i32::from(lo | (hi << 4)) - 32) as f32;
        for (j, &q) in qs.iter().enumerate() {
            out[j] = dl * IQ4NL[(q & 0xF) as usize] as f32;
            out[j + 16] = dl * IQ4NL[(q >> 4) as usize] as f32;
        }
    }
}

impl_opaque_block_quant!(IQ4XSF, "IQ4_XS", 256, 136, quant_iq4_xs, dequant_iq4_xs);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::block_quant::ggml_k::tests::assert_ggml_ref;
    use crate::frame::block_quant::ggml_k::tests::assert_roundtrip;
    use crate::frame::block_quant::ggml_k::tests::ramp;

    #[test]
    fn iq4_xs_roundtrip() {
        assert_roundtrip(&IQ4_XS, 0.35);
    }

    /// Bytes and values from llama.cpp `quantize_row_iq4_xs_ref` / `dequantize_row_iq4_xs`.
    #[test]
    fn iq4_xs_matches_ggml_ref() {
        assert_ggml_ref(
            &IQ4_XS,
            &ramp(256, 0),
            "5010f0f023ee20ef3fe8710cb32ed650fa921dc54ff8710c0ba32ed65ff9821dc44fe8710bb32ed63bb0178ef44cd1289ff56de23ab0167e8ff45cd129af056de23bb0178ef44cd13ed650fa922dc54fe8711cb43ed761faf9821dc44fe8710bb32ed650fa921dc54cd1289ff56de23ab0167ee34bc1289faf056de23bb0178ef44cd1289ff56de2",
            &[(0, 0xbfdc_d040), (32, 0xbf19_a200), (255, 0x3fb3_e980)],
        );
        assert_ggml_ref(
            &IQ4_XS,
            &ramp(256, 32),
            "5010f2f020ee20ef888888888888888888888888888888880ba32ed65ff9821dc44fe8710bb32ed63bb0178ef44cd1289ff56de23ab0167e8ff45cd129af056de23bb0178ef44cd13ed650fa922dc54fe8711cb43ed761faf9821dc44fe8710bb32ed650fa921dc54cd1289ff56de23ab0167ee34bc1289faf056de23bb0178ef44cd1289ff56de2",
            &[(0, 0), (32, 0xbf19_a200), (255, 0x3fb3_e980)],
        );
        assert_ggml_ref(
            &IQ4_XS,
            &[0f32; 256],
            &format!("0080aaaa00000000{}", "88".repeat(128)),
            &[],
        );
    }
}
