//! ACE "Data Format Agility" — OCP MX numeric decodes and the AVX-512 data
//! marshalling primitives (`VUNPACKB`, `VPERM`-class LUTs) that ACE relies on to
//! convert low-precision storage formats into native compute types.
//!
//! Unlike the outer-product instructions, these are *not* future-only: `VPERMB`
//! / `VPERMI2B` already exist on AVX-512, and `VUNPACKB` is a thin extension over
//! existing shift/permute capability. They are modelled here portably so the
//! conversion pipeline (e.g. FP4/FP6/FP8 → bf16/i8 dequant, sub-byte weight
//! unpacking) can be built and validated today and used to feed the emulated
//! outer-product kernels. Each models a documented ACE/AVX-512 operation.
//!
//! References: OCP Microscaling Formats (MX) Specification v1.0; ACE whitepaper
//! §"Data Format Agility".

// ---------------------------------------------------------------------------
// OCP MX scale + element numeric decodes
// ---------------------------------------------------------------------------

/// Decode an OCP MX block scale (E8M0: 8-bit, all-exponent, bias 127).
/// `value = 2^(byte - 127)`; `0xFF` encodes NaN (per OCP MX v1.0).
#[inline]
pub fn mx_scale_decode(s: u8) -> f32 {
    if s == 0xFF {
        f32::NAN
    } else {
        // 2^(s-127); representable in f32 across the whole E8M0 range
        // (subnormal for s < ~1, but powi handles it).
        2f32.powi(s as i32 - 127)
    }
}

/// Decode an OCP FP8 **E4M3** byte (1 sign, 4 exp bias 7, 3 mantissa).
/// No infinities; `S.1111.111` is NaN; max normal magnitude is 448.
#[inline]
pub fn fp8_e4m3_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = (b >> 3) & 0x0F;
    let mant = b & 0x07;
    let mag = if exp == 0 {
        // subnormal: 2^(1-7) * mant/8
        (mant as f32 / 8.0) * 2f32.powi(-6)
    } else if exp == 0x0F && mant == 0x07 {
        return f32::NAN;
    } else {
        (1.0 + mant as f32 / 8.0) * 2f32.powi(exp as i32 - 7)
    };
    sign * mag
}

/// Decode an OCP FP8 **E5M2** byte (1 sign, 5 exp bias 15, 2 mantissa).
/// IEEE-like: `exp=31,mant=0` is Inf; `exp=31,mant!=0` is NaN.
#[inline]
pub fn fp8_e5m2_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = (b >> 2) & 0x1F;
    let mant = b & 0x03;
    let mag = if exp == 0 {
        (mant as f32 / 4.0) * 2f32.powi(-14)
    } else if exp == 0x1F {
        if mant == 0 {
            return sign * f32::INFINITY;
        } else {
            return f32::NAN;
        }
    } else {
        (1.0 + mant as f32 / 4.0) * 2f32.powi(exp as i32 - 15)
    };
    sign * mag
}

/// Decode an OCP MXFP4 **E2M1** nibble (1 sign, 2 exp bias 1, 1 mantissa).
/// The 8 representable magnitudes are {0, .5, 1, 1.5, 2, 3, 4, 6}; no Inf/NaN.
#[inline]
pub fn fp4_e2m1_to_f32(nib: u8) -> f32 {
    const MAG: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let sign = if nib & 0x08 != 0 { -1.0 } else { 1.0 };
    sign * MAG[(nib & 0x07) as usize]
}

// ---------------------------------------------------------------------------
// bf16 <-> f32 (round-to-nearest-even), matching Intel VCVTNEPS2BF16
// ---------------------------------------------------------------------------

/// f32 → bf16 (top 16 bits) with round-to-nearest-even; NaN preserved as qNaN.
/// Mirrors `x86_64_fma::amx_bf16::f32_to_bf16_rne` but lives here so the portable
/// ACE model can use it on any target.
#[inline]
pub fn f32_to_bf16_rne(x: f32) -> u16 {
    let bits = x.to_bits();
    if x.is_nan() {
        ((bits >> 16) as u16) | 0x0040
    } else {
        let lsb = (bits >> 16) & 1;
        let rounding = 0x0000_7FFF + lsb;
        (bits.wrapping_add(rounding) >> 16) as u16
    }
}

/// bf16 (16-bit pattern) → f32 (exact; bf16 is the high half of f32).
#[inline]
pub fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

// ---------------------------------------------------------------------------
// VUNPACKB: unpack sub-byte packed elements into i8 lanes
// ---------------------------------------------------------------------------

/// Model of ACE's `VUNPACKB`: unpack `count` little-endian bit-packed elements of
/// `width` bits each (valid widths 2..=7) starting at `bit_offset` in `src`,
/// emitting one i8 lane per element. With `signed`, each element's top bit is
/// treated as a sign bit and sign-extended; otherwise zero-extended.
///
/// This is the marshalling step that turns packed 2–7-bit weight storage into the
/// 8-bit lanes the outer-product unit consumes (whitepaper §Data Format Agility).
pub fn vunpackb(
    src: &[u8],
    width: usize,
    count: usize,
    bit_offset: usize,
    signed: bool,
) -> Vec<i8> {
    assert!((2..=7).contains(&width), "VUNPACKB width must be 2..=7");
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let base = bit_offset + i * width;
        let mut v = 0u32;
        for b in 0..width {
            let bp = base + b;
            let bit = (src[bp / 8] >> (bp % 8)) & 1;
            v |= (bit as u32) << b;
        }
        let val = if signed && (v & (1 << (width - 1))) != 0 {
            (v as i32 - (1i32 << width)) as i8
        } else {
            v as i8
        };
        out.push(val);
    }
    out
}

// ---------------------------------------------------------------------------
// VPERMB / VPERMI2B: byte lookup tables for software-defined format conversion
// ---------------------------------------------------------------------------

/// Model of `VPERMB` with a 64-entry byte LUT (one 512-bit table register): for
/// each index byte, select `lut[idx & 0x3F]`. This converts any format up to
/// 6 bits to an arbitrary byte value in one pass (whitepaper notes the byte
/// VPERM forms enable LUT-based conversion of formats up to 7 bits).
pub fn vpermb(lut: &[u8; 64], idx: &[u8]) -> Vec<u8> {
    idx.iter().map(|&i| lut[(i & 0x3F) as usize]).collect()
}

/// Model of `VPERMI2B` with a 128-entry LUT spanning two table registers: select
/// `lut[idx & 0x7F]`. Enables 7-bit format conversion / codebook lookup.
pub fn vpermi2b(lut: &[u8; 128], idx: &[u8]) -> Vec<u8> {
    idx.iter().map(|&i| lut[(i & 0x7F) as usize]).collect()
}

/// Build a 16-entry FP4(E2M1) → bf16 dequant LUT (as bf16 bit patterns split into
/// the low/high byte tables a real `VPERMI2B`-based dequant would use). Returned
/// as `f32` values here for direct verification; the production path would store
/// bf16 bytes. Demonstrates the LUT-dequant pattern ACE/AVX-512 use for FP4.
pub fn fp4_e2m1_dequant_lut_f32() -> [f32; 16] {
    let mut lut = [0f32; 16];
    for (n, slot) in lut.iter_mut().enumerate() {
        *slot = fp4_e2m1_to_f32(n as u8);
    }
    lut
}

// ---------------------------------------------------------------------------
// f32 -> low-precision ENCODERS + the canonical MX block quantizer
// ---------------------------------------------------------------------------

/// Which element type an MX block stores.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum AceMxElem {
    /// OCP MXFP8 with E4M3 elements.
    MxFp8,
    /// OCP MXINT8 with signed-8-bit elements.
    MxInt8,
}

/// f32 -> OCP FP8 E4M3 byte, round-to-nearest over the representable magnitudes.
/// Never emits a NaN encoding for finite input (scans only finite bytes 0x00..=0x7E
/// and clamps to ±448). This is the inverse of [`fp8_e4m3_to_f32`]; written as an
/// exhaustive nearest-value search so it is obviously correct (it is a pack-time /
/// reference-model encoder, not a hot path).
pub fn f32_to_fp8_e4m3(x: f32) -> u8 {
    if x.is_nan() {
        return 0x7F;
    }
    let sign: u8 = if x.is_sign_negative() { 0x80 } else { 0 };
    if x.is_infinite() {
        return sign | 0x7E; // saturate ±Inf to ±448 (max finite), matching the >448 clamp
    }
    let mag = x.abs();
    let mut best = 0u8;
    let mut best_err = f32::INFINITY;
    for b in 0u8..=0x7E {
        let err = (fp8_e4m3_to_f32(b) - mag).abs();
        if err < best_err {
            best_err = err;
            best = b;
        }
    }
    sign | best
}

/// f32 -> signed 8-bit, round-to-nearest, saturating to [-127, 127] (symmetric).
pub fn f32_to_int8_sat(x: f32) -> i8 {
    let r = x.round();
    if r >= 127.0 {
        127
    } else if r <= -127.0 {
        -127
    } else {
        r as i8
    }
}

/// Canonical OCP-MX block quantizer — the **single source of truth** shared by the
/// MX packer and the differential-test reference. Given up to 32 f32 values of one
/// row's MX block, pick a shared E8M0 power-of-two scale `s` such that
/// `max|v| / s <= FMAX` (FMAX = 448 for E4M3, 127 for int8), then encode each
/// element as `v / s`. Writes the element bytes into `out` (E4M3 bytes for MXFP8;
/// `i8 as u8` for MXINT8) and returns the E8M0 scale byte (never 0xFF for finite
/// input). Decoding `s * decode(out[i])` reconstructs `v[i]` to within the element
/// format's resolution; the kernel's `scale_a*scale_b*Σ(a*b)` then recovers the
/// original products.
pub fn quantize_mx_block(vals: &[f32], elem: AceMxElem, out: &mut [u8]) -> u8 {
    debug_assert_eq!(vals.len(), out.len());
    let fmax = match elem {
        AceMxElem::MxFp8 => 448.0f32,
        AceMxElem::MxInt8 => 127.0f32,
    };
    let max_abs = vals.iter().fold(0f32, |a, &b| a.max(b.abs()));
    let scale_byte = if max_abs > 0.0 && max_abs.is_finite() {
        // smallest e with 2^e >= max_abs/FMAX  =>  max_abs / 2^e <= FMAX.
        let e = (max_abs / fmax).log2().ceil() as i32;
        (e.clamp(-127, 127) + 127) as u8
    } else {
        127 // 2^0
    };
    let inv = 1.0 / mx_scale_decode(scale_byte);
    for (o, &v) in out.iter_mut().zip(vals) {
        *o = match elem {
            AceMxElem::MxFp8 => f32_to_fp8_e4m3(v * inv),
            AceMxElem::MxInt8 => f32_to_int8_sat(v * inv) as u8,
        };
    }
    scale_byte
}

#[cfg(test)]
mod tests {
    use super::*;

    // Independently generated (python3, OCP e4m3fn / e5m2 spec, self-checked vs
    // external anchors) 256-entry decode ground truth. Used by the exhaustive
    // decode tests below to pin the numerics without reusing the impl under test.
    #[rustfmt::skip]
    const E4M3_GROUND_TRUTH: [f32; 256] = [
        0.0f32, 0.001953125f32, 0.00390625f32, 0.005859375f32, 0.0078125f32, 0.009765625f32, 0.01171875f32, 0.013671875f32,
        0.015625f32, 0.017578125f32, 0.01953125f32, 0.021484375f32, 0.0234375f32, 0.025390625f32, 0.02734375f32, 0.029296875f32,
        0.03125f32, 0.03515625f32, 0.0390625f32, 0.04296875f32, 0.046875f32, 0.05078125f32, 0.0546875f32, 0.05859375f32,
        0.0625f32, 0.0703125f32, 0.078125f32, 0.0859375f32, 0.09375f32, 0.1015625f32, 0.109375f32, 0.1171875f32,
        0.125f32, 0.140625f32, 0.15625f32, 0.171875f32, 0.1875f32, 0.203125f32, 0.21875f32, 0.234375f32,
        0.25f32, 0.28125f32, 0.3125f32, 0.34375f32, 0.375f32, 0.40625f32, 0.4375f32, 0.46875f32,
        0.5f32, 0.5625f32, 0.625f32, 0.6875f32, 0.75f32, 0.8125f32, 0.875f32, 0.9375f32,
        1.0f32, 1.125f32, 1.25f32, 1.375f32, 1.5f32, 1.625f32, 1.75f32, 1.875f32,
        2.0f32, 2.25f32, 2.5f32, 2.75f32, 3.0f32, 3.25f32, 3.5f32, 3.75f32,
        4.0f32, 4.5f32, 5.0f32, 5.5f32, 6.0f32, 6.5f32, 7.0f32, 7.5f32,
        8.0f32, 9.0f32, 10.0f32, 11.0f32, 12.0f32, 13.0f32, 14.0f32, 15.0f32,
        16.0f32, 18.0f32, 20.0f32, 22.0f32, 24.0f32, 26.0f32, 28.0f32, 30.0f32,
        32.0f32, 36.0f32, 40.0f32, 44.0f32, 48.0f32, 52.0f32, 56.0f32, 60.0f32,
        64.0f32, 72.0f32, 80.0f32, 88.0f32, 96.0f32, 104.0f32, 112.0f32, 120.0f32,
        128.0f32, 144.0f32, 160.0f32, 176.0f32, 192.0f32, 208.0f32, 224.0f32, 240.0f32,
        256.0f32, 288.0f32, 320.0f32, 352.0f32, 384.0f32, 416.0f32, 448.0f32, f32::NAN,
        -0.0f32, -0.001953125f32, -0.00390625f32, -0.005859375f32, -0.0078125f32, -0.009765625f32, -0.01171875f32, -0.013671875f32,
        -0.015625f32, -0.017578125f32, -0.01953125f32, -0.021484375f32, -0.0234375f32, -0.025390625f32, -0.02734375f32, -0.029296875f32,
        -0.03125f32, -0.03515625f32, -0.0390625f32, -0.04296875f32, -0.046875f32, -0.05078125f32, -0.0546875f32, -0.05859375f32,
        -0.0625f32, -0.0703125f32, -0.078125f32, -0.0859375f32, -0.09375f32, -0.1015625f32, -0.109375f32, -0.1171875f32,
        -0.125f32, -0.140625f32, -0.15625f32, -0.171875f32, -0.1875f32, -0.203125f32, -0.21875f32, -0.234375f32,
        -0.25f32, -0.28125f32, -0.3125f32, -0.34375f32, -0.375f32, -0.40625f32, -0.4375f32, -0.46875f32,
        -0.5f32, -0.5625f32, -0.625f32, -0.6875f32, -0.75f32, -0.8125f32, -0.875f32, -0.9375f32,
        -1.0f32, -1.125f32, -1.25f32, -1.375f32, -1.5f32, -1.625f32, -1.75f32, -1.875f32,
        -2.0f32, -2.25f32, -2.5f32, -2.75f32, -3.0f32, -3.25f32, -3.5f32, -3.75f32,
        -4.0f32, -4.5f32, -5.0f32, -5.5f32, -6.0f32, -6.5f32, -7.0f32, -7.5f32,
        -8.0f32, -9.0f32, -10.0f32, -11.0f32, -12.0f32, -13.0f32, -14.0f32, -15.0f32,
        -16.0f32, -18.0f32, -20.0f32, -22.0f32, -24.0f32, -26.0f32, -28.0f32, -30.0f32,
        -32.0f32, -36.0f32, -40.0f32, -44.0f32, -48.0f32, -52.0f32, -56.0f32, -60.0f32,
        -64.0f32, -72.0f32, -80.0f32, -88.0f32, -96.0f32, -104.0f32, -112.0f32, -120.0f32,
        -128.0f32, -144.0f32, -160.0f32, -176.0f32, -192.0f32, -208.0f32, -224.0f32, -240.0f32,
        -256.0f32, -288.0f32, -320.0f32, -352.0f32, -384.0f32, -416.0f32, -448.0f32, f32::NAN,
    ];
    #[rustfmt::skip]
    const E5M2_GROUND_TRUTH: [f32; 256] = [
        0.0f32, 1.52587890625e-05f32, 3.0517578125e-05f32, 4.57763671875e-05f32, 6.103515625e-05f32, 7.62939453125e-05f32, 9.1552734375e-05f32, 0.0001068115234375f32,
        0.0001220703125f32, 0.000152587890625f32, 0.00018310546875f32, 0.000213623046875f32, 0.000244140625f32, 0.00030517578125f32, 0.0003662109375f32, 0.00042724609375f32,
        0.00048828125f32, 0.0006103515625f32, 0.000732421875f32, 0.0008544921875f32, 0.0009765625f32, 0.001220703125f32, 0.00146484375f32, 0.001708984375f32,
        0.001953125f32, 0.00244140625f32, 0.0029296875f32, 0.00341796875f32, 0.00390625f32, 0.0048828125f32, 0.005859375f32, 0.0068359375f32,
        0.0078125f32, 0.009765625f32, 0.01171875f32, 0.013671875f32, 0.015625f32, 0.01953125f32, 0.0234375f32, 0.02734375f32,
        0.03125f32, 0.0390625f32, 0.046875f32, 0.0546875f32, 0.0625f32, 0.078125f32, 0.09375f32, 0.109375f32,
        0.125f32, 0.15625f32, 0.1875f32, 0.21875f32, 0.25f32, 0.3125f32, 0.375f32, 0.4375f32,
        0.5f32, 0.625f32, 0.75f32, 0.875f32, 1.0f32, 1.25f32, 1.5f32, 1.75f32,
        2.0f32, 2.5f32, 3.0f32, 3.5f32, 4.0f32, 5.0f32, 6.0f32, 7.0f32,
        8.0f32, 10.0f32, 12.0f32, 14.0f32, 16.0f32, 20.0f32, 24.0f32, 28.0f32,
        32.0f32, 40.0f32, 48.0f32, 56.0f32, 64.0f32, 80.0f32, 96.0f32, 112.0f32,
        128.0f32, 160.0f32, 192.0f32, 224.0f32, 256.0f32, 320.0f32, 384.0f32, 448.0f32,
        512.0f32, 640.0f32, 768.0f32, 896.0f32, 1024.0f32, 1280.0f32, 1536.0f32, 1792.0f32,
        2048.0f32, 2560.0f32, 3072.0f32, 3584.0f32, 4096.0f32, 5120.0f32, 6144.0f32, 7168.0f32,
        8192.0f32, 10240.0f32, 12288.0f32, 14336.0f32, 16384.0f32, 20480.0f32, 24576.0f32, 28672.0f32,
        32768.0f32, 40960.0f32, 49152.0f32, 57344.0f32, f32::INFINITY, f32::NAN, f32::NAN, f32::NAN,
        -0.0f32, -1.52587890625e-05f32, -3.0517578125e-05f32, -4.57763671875e-05f32, -6.103515625e-05f32, -7.62939453125e-05f32, -9.1552734375e-05f32, -0.0001068115234375f32,
        -0.0001220703125f32, -0.000152587890625f32, -0.00018310546875f32, -0.000213623046875f32, -0.000244140625f32, -0.00030517578125f32, -0.0003662109375f32, -0.00042724609375f32,
        -0.00048828125f32, -0.0006103515625f32, -0.000732421875f32, -0.0008544921875f32, -0.0009765625f32, -0.001220703125f32, -0.00146484375f32, -0.001708984375f32,
        -0.001953125f32, -0.00244140625f32, -0.0029296875f32, -0.00341796875f32, -0.00390625f32, -0.0048828125f32, -0.005859375f32, -0.0068359375f32,
        -0.0078125f32, -0.009765625f32, -0.01171875f32, -0.013671875f32, -0.015625f32, -0.01953125f32, -0.0234375f32, -0.02734375f32,
        -0.03125f32, -0.0390625f32, -0.046875f32, -0.0546875f32, -0.0625f32, -0.078125f32, -0.09375f32, -0.109375f32,
        -0.125f32, -0.15625f32, -0.1875f32, -0.21875f32, -0.25f32, -0.3125f32, -0.375f32, -0.4375f32,
        -0.5f32, -0.625f32, -0.75f32, -0.875f32, -1.0f32, -1.25f32, -1.5f32, -1.75f32,
        -2.0f32, -2.5f32, -3.0f32, -3.5f32, -4.0f32, -5.0f32, -6.0f32, -7.0f32,
        -8.0f32, -10.0f32, -12.0f32, -14.0f32, -16.0f32, -20.0f32, -24.0f32, -28.0f32,
        -32.0f32, -40.0f32, -48.0f32, -56.0f32, -64.0f32, -80.0f32, -96.0f32, -112.0f32,
        -128.0f32, -160.0f32, -192.0f32, -224.0f32, -256.0f32, -320.0f32, -384.0f32, -448.0f32,
        -512.0f32, -640.0f32, -768.0f32, -896.0f32, -1024.0f32, -1280.0f32, -1536.0f32, -1792.0f32,
        -2048.0f32, -2560.0f32, -3072.0f32, -3584.0f32, -4096.0f32, -5120.0f32, -6144.0f32, -7168.0f32,
        -8192.0f32, -10240.0f32, -12288.0f32, -14336.0f32, -16384.0f32, -20480.0f32, -24576.0f32, -28672.0f32,
        -32768.0f32, -40960.0f32, -49152.0f32, -57344.0f32, f32::NEG_INFINITY, f32::NAN, f32::NAN, f32::NAN,
    ];

    #[test]
    fn mx_scale_powers_of_two() {
        assert_eq!(mx_scale_decode(127), 1.0); // 2^0
        assert_eq!(mx_scale_decode(128), 2.0); // 2^1
        assert_eq!(mx_scale_decode(126), 0.5); // 2^-1
        assert!(mx_scale_decode(0xFF).is_nan());
    }

    #[test]
    fn fp8_e4m3_known_values() {
        assert_eq!(fp8_e4m3_to_f32(0x00), 0.0);
        // 0 01110 000? exp bits: value 1.0 = exp 7 (bias 7) -> 0_0111_000 = 0x38
        assert_eq!(fp8_e4m3_to_f32(0x38), 1.0);
        assert_eq!(fp8_e4m3_to_f32(0xB8), -1.0); // sign bit set
        // max normal: 0_1111_110 = 448
        assert_eq!(fp8_e4m3_to_f32(0x7E), 448.0);
        assert!(fp8_e4m3_to_f32(0x7F).is_nan()); // S.1111.111
    }

    #[test]
    fn fp8_e5m2_known_values() {
        assert_eq!(fp8_e5m2_to_f32(0x00), 0.0);
        // 1.0 = exp 15 (bias 15), mant 0 -> 0_01111_00 = 0x3C
        assert_eq!(fp8_e5m2_to_f32(0x3C), 1.0);
        assert!(fp8_e5m2_to_f32(0x7C) == f32::INFINITY); // 0_11111_00
        assert!(fp8_e5m2_to_f32(0x7D).is_nan()); // 0_11111_01
    }

    #[test]
    fn fp4_e2m1_full_table() {
        let want = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        for (i, &w) in want.iter().enumerate() {
            assert_eq!(fp4_e2m1_to_f32(i as u8), w);
            assert_eq!(fp4_e2m1_to_f32(i as u8 | 0x08), -w);
        }
        assert_eq!(fp4_e2m1_dequant_lut_f32()[5], 3.0);
    }

    #[test]
    fn bf16_roundtrip_and_rounding() {
        // exact bf16 values round-trip
        for &v in &[0.0f32, 1.0, -2.0, 0.5, 256.0] {
            assert_eq!(bf16_to_f32(f32_to_bf16_rne(v)), v);
        }
        // a value needing rounding is within bf16 resolution of the original
        let x = 1.0001f32;
        let r = bf16_to_f32(f32_to_bf16_rne(x));
        assert!((r - x).abs() < 0.01);
        assert!(f32::from_bits((f32_to_bf16_rne(f32::NAN) as u32) << 16).is_nan());
    }

    #[test]
    fn vunpackb_unsigned_and_signed() {
        // pack 3-bit values 0..8 little-endian: 0,1,2,3,4,5,6,7
        let vals: [u8; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
        let mut packed = vec![0u8; 4]; // 8*3 = 24 bits = 3 bytes (+slack)
        for (i, &v) in vals.iter().enumerate() {
            for b in 0..3 {
                if (v >> b) & 1 != 0 {
                    let bp = i * 3 + b;
                    packed[bp / 8] |= 1 << (bp % 8);
                }
            }
        }
        let unsigned = vunpackb(&packed, 3, 8, 0, false);
        assert_eq!(unsigned, vec![0, 1, 2, 3, 4, 5, 6, 7]);
        // same bits, signed 3-bit: 4,5,6,7 -> -4,-3,-2,-1
        let signed = vunpackb(&packed, 3, 8, 0, true);
        assert_eq!(signed, vec![0, 1, 2, 3, -4, -3, -2, -1]);
    }

    #[test]
    fn vpermb_lut_gather() {
        let mut lut = [0u8; 64];
        for (i, slot) in lut.iter_mut().enumerate() {
            *slot = (i as u8).wrapping_mul(3);
        }
        let idx = [0u8, 1, 5, 63, 64 /* wraps to 0 */];
        assert_eq!(vpermb(&lut, &idx), vec![0, 3, 15, 189, 0]);
    }

    #[test]
    fn vpermi2b_lut_gather() {
        let mut lut = [0u8; 128];
        for (i, slot) in lut.iter_mut().enumerate() {
            *slot = (i as u8).wrapping_mul(3);
        }
        // idx & 0x7F = [0,1,64,127,0,127]
        let idx = [0u8, 1, 64, 127, 128 /* ->0 */, 0xFF /* ->127 */];
        assert_eq!(vpermi2b(&lut, &idx), vec![lut[0], lut[1], lut[64], lut[127], lut[0], lut[127]]);
        // The 7-bit mask keeps bit 6: index 64 selects lut[64], not lut[0] (which
        // VPERMB's 6-bit 0x3F mask would give). This is what distinguishes the two.
        assert_eq!(vpermi2b(&lut, &[64])[0], lut[64]);
        assert_ne!(lut[64], lut[0]);
    }

    #[test]
    fn vunpackb_width7_signed_unsigned() {
        let codes: [u8; 8] = [0, 1, 63, 64, 100, 127, 65, 90];
        let mut packed = vec![0u8; 8]; // 8*7 = 56 bits = 7 bytes (+slack)
        for (i, &v) in codes.iter().enumerate() {
            for b in 0..7 {
                if (v >> b) & 1 != 0 {
                    let bp = i * 7 + b;
                    packed[bp / 8] |= 1 << (bp % 8);
                }
            }
        }
        assert_eq!(vunpackb(&packed, 7, 8, 0, false), vec![0, 1, 63, 64, 100, 127, 65, 90]);
        // 7-bit signed: codes with bit6 set (64,100,127,65,90) -> value - 128
        assert_eq!(vunpackb(&packed, 7, 8, 0, true), vec![0, 1, 63, -64, -28, -1, -63, -38]);
    }

    #[test]
    fn vunpackb_width5_nonzero_offset() {
        let codes: [u8; 4] = [5, 31, 16, 10];
        let mut packed = vec![0u8; 4]; // bit_offset 3 + 4*5 = 23 bits -> 3 bytes (+slack)
        for (i, &v) in codes.iter().enumerate() {
            for b in 0..5 {
                if (v >> b) & 1 != 0 {
                    let bp = 3 + i * 5 + b;
                    packed[bp / 8] |= 1 << (bp % 8);
                }
            }
        }
        assert_eq!(vunpackb(&packed, 5, 4, 3, false), vec![5, 31, 16, 10]);
        // 5-bit signed: 31->-1, 16->-16 (bit4 set); 5,10 unchanged
        assert_eq!(vunpackb(&packed, 5, 4, 3, true), vec![5, -1, -16, 10]);
    }

    // E8M0 range endpoints (the test above only covered the unit region). E8M0 has
    // no mantissa, so every byte 0..=254 is an exact power of two 2^(s-127); only
    // s=0 (2^-127) lands in the f32 subnormal range, and 0xFF is NaN.
    #[test]
    fn mx_scale_range_endpoints() {
        assert_eq!(mx_scale_decode(0), 2f32.powi(-127)); // smallest; an f32 subnormal
        assert_eq!(mx_scale_decode(1), 2f32.powi(-126)); // smallest normal f32
        assert_eq!(mx_scale_decode(120), 2f32.powi(-7));
        assert_eq!(mx_scale_decode(129), 2f32.powi(2));
        assert_eq!(mx_scale_decode(254), 2f32.powi(127)); // largest finite
        assert!(mx_scale_decode(0) > 0.0 && mx_scale_decode(0).is_finite());
    }

    // Compare exact equality, treating NaN/Inf specially. All FP8 values are exactly
    // representable in f32, so finite comparison is bit-exact.
    fn fp8_eq(got: f32, want: f32) -> bool {
        if want.is_nan() {
            got.is_nan()
        } else {
            got == want && got.is_infinite() == want.is_infinite()
        }
    }

    // Exhaustive E4M3 decode against an independently generated (python3, from the
    // OCP e4m3fn spec, self-checked vs external anchors) 256-entry ground-truth
    // table. This is what makes the MXFP8 block tests non-tautological: the decode
    // numerics are pinned here, independent of the kernel's own helpers.
    #[test]
    fn fp8_e4m3_exhaustive_vs_ground_truth() {
        for b in 0..=255u8 {
            let got = fp8_e4m3_to_f32(b);
            let want = E4M3_GROUND_TRUTH[b as usize];
            assert!(fp8_eq(got, want), "E4M3 {b:#04x}: got {got}, want {want}");
        }
        // structural: exactly two NaN encodings (0x7F,0xFF), no infinities, sign-symmetric
        assert_eq!((0..=255u8).filter(|&b| fp8_e4m3_to_f32(b).is_nan()).count(), 2);
        assert!(!(0..=255u8).any(|b| fp8_e4m3_to_f32(b).is_infinite()));
        assert_eq!(fp8_e4m3_to_f32(0x80).to_bits(), (-0.0f32).to_bits()); // negative zero
        for b in 0..0x80u8 {
            if !fp8_e4m3_to_f32(b).is_nan() {
                assert_eq!(fp8_e4m3_to_f32(b | 0x80), -fp8_e4m3_to_f32(b));
            }
        }
    }

    #[test]
    fn fp8_e5m2_exhaustive_vs_ground_truth() {
        for b in 0..=255u8 {
            let got = fp8_e5m2_to_f32(b);
            let want = E5M2_GROUND_TRUTH[b as usize];
            assert!(fp8_eq(got, want), "E5M2 {b:#04x}: got {got}, want {want}");
        }
        // structural: 2 infinities (0x7C,0xFC), 6 NaNs (0x7D-0x7F, 0xFD-0xFF)
        assert_eq!((0..=255u8).filter(|&b| fp8_e5m2_to_f32(b).is_infinite()).count(), 2);
        assert_eq!((0..=255u8).filter(|&b| fp8_e5m2_to_f32(b).is_nan()).count(), 6);
    }

    // Encoding any exactly-representable E4M3 value must round-trip to the same value.
    #[test]
    fn f32_to_fp8_e4m3_roundtrips() {
        for b in 0u8..=255u8 {
            let v = fp8_e4m3_to_f32(b);
            if v.is_nan() {
                continue;
            }
            let re = fp8_e4m3_to_f32(f32_to_fp8_e4m3(v));
            assert_eq!(re, v, "byte {b:#04x} value {v} did not round-trip (got {re})");
        }
        // never emits a NaN encoding for finite input
        assert!(!fp8_e4m3_to_f32(f32_to_fp8_e4m3(1e9)).is_nan());
        assert_eq!(fp8_e4m3_to_f32(f32_to_fp8_e4m3(1e9)), 448.0); // saturates to max
    }

    #[test]
    fn f32_to_int8_saturates() {
        assert_eq!(f32_to_int8_sat(3.4), 3);
        assert_eq!(f32_to_int8_sat(-3.6), -4);
        assert_eq!(f32_to_int8_sat(1000.0), 127);
        assert_eq!(f32_to_int8_sat(-1000.0), -127);
    }

    // quantize_mx_block: decode(scale)*decode(elem) reconstructs the input within
    // the element format's resolution; all-zero block yields a unit scale.
    #[test]
    fn quantize_mx_block_reconstructs() {
        let vals: Vec<f32> =
            (0..32).map(|i| (i as f32 - 16.0) * 0.37 + (i as f32).sin() * 5.0).collect();

        let mut out = [0u8; 32];
        let s = quantize_mx_block(&vals, AceMxElem::MxFp8, &mut out);
        assert_ne!(s, 0xFF);
        let scale = mx_scale_decode(s);
        for (i, &v) in vals.iter().enumerate() {
            let re = scale * fp8_e4m3_to_f32(out[i]);
            assert!((re - v).abs() <= 0.2 * v.abs() + 0.5, "fp8 [{i}] v={v} re={re}");
        }

        let mut out8 = [0u8; 32];
        let s8 = quantize_mx_block(&vals, AceMxElem::MxInt8, &mut out8);
        let scale8 = mx_scale_decode(s8);
        for (i, &v) in vals.iter().enumerate() {
            let re = scale8 * (out8[i] as i8 as f32);
            assert!((re - v).abs() <= 0.05 * v.abs() + scale8, "int8 [{i}] v={v} re={re}");
        }

        // all-zero block -> unit scale (2^0), all-zero elements
        let mut z = [0xAAu8; 32];
        assert_eq!(quantize_mx_block(&[0.0; 32], AceMxElem::MxFp8, &mut z), 127);
        assert!(z.iter().all(|&b| b == 0));
    }

    #[test]
    fn f32_to_fp8_e4m3_saturates_inf() {
        assert_eq!(fp8_e4m3_to_f32(f32_to_fp8_e4m3(f32::INFINITY)), 448.0);
        assert_eq!(fp8_e4m3_to_f32(f32_to_fp8_e4m3(f32::NEG_INFINITY)), -448.0);
        assert!(fp8_e4m3_to_f32(f32_to_fp8_e4m3(f32::NAN)).is_nan());
    }

    // Independent oracle for the E8M0 scale byte: clamp(ceil(log2(max_abs/FMAX)))+127.
    // Does NOT reuse the element encoder, so it pins the scale selection on its own.
    #[test]
    fn quantize_mx_block_scale_byte_oracle() {
        let mut o = [0u8; 32];
        let mk = |v: f32| {
            let mut a = [0f32; 32];
            a[0] = v;
            a
        };
        // MXFP8, FMAX=448
        assert_eq!(quantize_mx_block(&mk(1.0), AceMxElem::MxFp8, &mut o), 119); // 2^-8
        assert_eq!(quantize_mx_block(&mk(448.0), AceMxElem::MxFp8, &mut o), 127); // 2^0
        assert_eq!(quantize_mx_block(&mk(896.0), AceMxElem::MxFp8, &mut o), 128); // 2^1
        // MXINT8, FMAX=127
        assert_eq!(quantize_mx_block(&mk(1.0), AceMxElem::MxInt8, &mut o), 121); // 2^-6
        assert_eq!(quantize_mx_block(&mk(127.0), AceMxElem::MxInt8, &mut o), 127);
        assert_eq!(quantize_mx_block(&mk(254.0), AceMxElem::MxInt8, &mut o), 128);
    }

    // Pins the rounding convention: f32::round is half-AWAY-from-zero (2.5->3),
    // NOT IEEE half-to-even. Documented so a future change is a deliberate choice.
    #[test]
    fn f32_to_int8_rounding_is_half_away() {
        assert_eq!(f32_to_int8_sat(2.5), 3);
        assert_eq!(f32_to_int8_sat(-2.5), -3);
        assert_eq!(f32_to_int8_sat(0.5), 1);
    }
}
