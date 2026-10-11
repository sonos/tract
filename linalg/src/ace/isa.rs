//! Portable reference model of the ACE (AI Compute Extensions) v1 tile ISA.
//!
//! Each function models one instruction of the x86 Ecosystem Advisory Group "AI
//! Compute Extensions (ACE) Specification" v1.15, bit-exact to its pseudocode
//! (§14, helpers §16.5). The model is the fallback compute path of the ACE kernels
//! on hosts without ACE, and the oracle the hardware path is checked against in
//! [`super::hw`] (run under Intel SDE `-future-ag` until silicon ships).
//!
//! ## ACE state (§10.2)
//!   * **8 tile registers**, each `512b × 16 rows` = a **16×16** i32/f32 accumulator.
//!   * **1 Block Scale Register** (BSR), 1024 bits: the high half holds the A
//!     scales, the low half the B scales, each as 4 groups of 16 E8M0 bytes
//!     interleaved at stride 4 (`A: byte[64 + s*4 + g]`, `B: byte[s*4 + g]`).
//!
//! ## Operands (§10.4)
//! An outer product consumes two ZMM registers: `src1` (ModRM.r/m) supplies one
//! 32-bit lane per tile **row** (the A side), `src2` (VEX.vvvv) one lane per tile
//! **column** (the B side). Each lane packs K sub-elements with `k0` in the low
//! bits, so a ZMM holds a `16×K` sub-matrix in lane order `row*K + k` — the
//! layout of tract's [`crate::pack::PackedI8K4`] for K=4.

use super::format::bf16_to_f32;

/// Tile registers are square, 16×16.
pub const ACE_TILE_DIM: usize = 16;
/// K elements consumed per 8-bit ZMM operand (a `16×4` sub-matrix).
pub const ACE_K_INT8: usize = 4;
/// K elements consumed per bf16 ZMM operand (a `16×2` sub-matrix).
pub const ACE_K_BF16: usize = 2;
/// Bytes in one ZMM operand.
pub const ACE_ZMM_BYTES: usize = 64;
/// Bytes in one 8-bit ZMM operand / one K-block of a PackedI8K4 panel (r=16).
pub const ACE_I8_BLOCK_BYTES: usize = ACE_TILE_DIM * ACE_K_INT8;
/// bf16 lanes in one ZMM operand (16 rows × 2 K).
pub const ACE_BF16_LANES: usize = ACE_TILE_DIM * ACE_K_BF16;
/// MX block length along K: one E8M0 scale per 32 consecutive K elements (OCP MX).
pub const ACE_MX_BLOCK_K: usize = 32;
/// Elements in one 16-wide × 32-K MX block: 8 consecutive K=4 ZMM operands.
pub const ACE_MX_BLOCK_ELEMS: usize = ACE_TILE_DIM * ACE_MX_BLOCK_K;

/// `QNaN_Indefinite`: the result of an invalid operation (`Inf - Inf`, `0 * Inf`).
pub const ACE_QNAN_INDEFINITE: u32 = 0xFFC0_0000;
/// The result of an MX outer product whose E8M0 scale is NaN (`0xFF`).
pub const ACE_MX_SCALE_NAN: u32 = 0x7FC0_0000;

/// A tile register viewed as a 16×16 **i32** accumulator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AceTileI32 {
    pub e: [[i32; ACE_TILE_DIM]; ACE_TILE_DIM],
}

impl AceTileI32 {
    #[inline]
    pub fn zero() -> Self {
        AceTileI32 { e: [[0; ACE_TILE_DIM]; ACE_TILE_DIM] }
    }
}

/// A tile register viewed as a 16×16 **f32** accumulator.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AceTileF32 {
    pub e: [[f32; ACE_TILE_DIM]; ACE_TILE_DIM],
}

impl AceTileF32 {
    #[inline]
    pub fn zero() -> Self {
        AceTileF32 { e: [[0.0; ACE_TILE_DIM]; ACE_TILE_DIM] }
    }
}

/// The 1024-bit Block Scale Register, as 128 E8M0 bytes (§10.2.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AceBsr {
    pub bytes: [u8; 128],
}

impl AceBsr {
    /// `BSRINIT`: every scale is `0x7F` (2^0). Also the state after `LDTILECFG`.
    pub fn init() -> Self {
        AceBsr { bytes: [0x7F; 128] }
    }

    /// `BSRMOVF bsr0, a, b`: `a` fills the A half (bits 1023:512), `b` the B half.
    pub fn movf(&mut self, a: &[u8; ACE_ZMM_BYTES], b: &[u8; ACE_ZMM_BYTES]) {
        self.bytes[64..].copy_from_slice(a);
        self.bytes[..64].copy_from_slice(b);
    }

    /// Loads one 16-wide scale strip per side into group 0 (imm8 = 0), the form
    /// `vpmovzxbd` + `BSRMOVF` produces: lane `s` = `scale[s]` zero-extended.
    pub fn load_group0(&mut self, a: &[u8; ACE_TILE_DIM], b: &[u8; ACE_TILE_DIM]) {
        let widen = |s: &[u8; ACE_TILE_DIM]| {
            let mut z = [0u8; ACE_ZMM_BYTES];
            for (i, &v) in s.iter().enumerate() {
                z[i * 4] = v;
            }
            z
        };
        self.movf(&widen(a), &widen(b));
    }

    #[inline]
    fn a_scale(&self, imm8: u8, s: usize) -> u8 {
        self.bytes[64 + s * 4 + (imm8 & 3) as usize]
    }

    #[inline]
    fn b_scale(&self, imm8: u8, s: usize) -> u8 {
        self.bytes[s * 4 + ((imm8 >> 3) & 3) as usize]
    }
}

/// FP8 element encodings accepted by the MX FP8 outer products.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AceFp8 {
    /// E5M2 ("BF8").
    Bf8,
    /// E4M3 ("HF8").
    Hf8,
}

#[inline]
fn lane(z: &[u8; ACE_ZMM_BYTES], i: usize) -> [u8; 4] {
    [z[i * 4], z[i * 4 + 1], z[i * 4 + 2], z[i * 4 + 3]]
}

fn top4b(
    dst: &mut AceTileI32,
    a: &[u8; ACE_ZMM_BYTES],
    a_signed: bool,
    b: &[u8; ACE_ZMM_BYTES],
    b_signed: bool,
) {
    let widen = |x: u8, signed: bool| if signed { x as i8 as i32 } else { x as i32 };
    for m in 0..ACE_TILE_DIM {
        let la = lane(a, m);
        for n in 0..ACE_TILE_DIM {
            let lb = lane(b, n);
            let mut acc = 0i32;
            for k in 0..ACE_K_INT8 {
                acc += widen(la[k], a_signed) * widen(lb[k], b_signed);
            }
            dst.e[m][n] = dst.e[m][n].wrapping_add(acc);
        }
    }
}

/// `TOP4BSSD tmm, a, b` (§14.4): signed×signed rank-4 byte outer product, i32
/// accumulate (wrapping).
pub fn ace_top4bssd(dst: &mut AceTileI32, a: &[i8; ACE_ZMM_BYTES], b: &[i8; ACE_ZMM_BYTES]) {
    let as_u8 = |z: &[i8; ACE_ZMM_BYTES]| z.map(|x| x as u8);
    top4b(dst, &as_u8(a), true, &as_u8(b), true)
}

/// `TOP4BUUD tmm, a, b` (§14.4): unsigned×unsigned rank-4 byte outer product.
pub fn ace_top4buud(dst: &mut AceTileI32, a: &[u8; ACE_ZMM_BYTES], b: &[u8; ACE_ZMM_BYTES]) {
    top4b(dst, a, false, b, false)
}

#[inline]
fn flush_subnormal(x: f32) -> f32 {
    if x.is_subnormal() { f32::from_bits(x.to_bits() & 0x8000_0000) } else { x }
}

/// x86 SSE NaN rules for a binary op: a NaN operand propagates quieted (the first
/// one wins), an invalid operation yields `QNaN_Indefinite`.
#[inline]
fn x86_op(a: f32, b: f32, op: impl Fn(f32, f32) -> f32) -> f32 {
    let quiet = |x: f32| f32::from_bits(x.to_bits() | 0x0040_0000);
    if a.is_nan() {
        quiet(a)
    } else if b.is_nan() {
        quiet(b)
    } else {
        let r = op(a, b);
        if r.is_nan() { f32::from_bits(ACE_QNAN_INDEFINITE) } else { r }
    }
}

/// `float32_add(acc, x, daz_a=1, daz_b=0, ftz=1)`.
#[inline]
fn add_daz_ftz(acc: f32, x: f32) -> f32 {
    flush_subnormal(x86_op(flush_subnormal(acc), x, |a, b| a + b))
}

/// `TOP2BF16PS tmm, a, b` (§14.3): rank-2 BF16 outer product, f32 accumulate.
///
/// Per tile element: `p0 = a0*b0`, `sop = fma(a1, b1, p0)`, `acc + sop`, each
/// rounded RNE with FTZ, no DAZ, and x86 NaN propagation (`a1`, `b1`, then `p0`).
/// This follows Intel SDE 10.13.1 (as does Bochs' ACE model), which departs from
/// the v1.15 pseudocode (two rounded products, DAZ on inputs and accumulator).
pub fn ace_top2bf16ps(dst: &mut AceTileF32, a: &[u16; ACE_BF16_LANES], b: &[u16; ACE_BF16_LANES]) {
    let quiet = |x: f32| f32::from_bits(x.to_bits() | 0x0040_0000);
    for m in 0..ACE_TILE_DIM {
        let (a0, a1) = (bf16_to_f32(a[m * 2]), bf16_to_f32(a[m * 2 + 1]));
        for n in 0..ACE_TILE_DIM {
            let (b0, b1) = (bf16_to_f32(b[n * 2]), bf16_to_f32(b[n * 2 + 1]));
            let p0 = flush_subnormal(x86_op(a0, b0, |a, b| a * b));
            let sop = if let Some(nan) = [a1, b1, p0].into_iter().find(|x| x.is_nan()) {
                quiet(nan)
            } else {
                let r = a1.mul_add(b1, p0);
                if r.is_nan() { f32::from_bits(ACE_QNAN_INDEFINITE) } else { flush_subnormal(r) }
            };
            dst.e[m][n] = flush_subnormal(x86_op(dst.e[m][n], sop, |a, b| a + b));
        }
    }
}

/// `convert_fp8_to_fixpoint64` (§16.5): the FP8 value as an integer multiple of
/// its smallest subnormal (2^-16 for BF8, 2^-9 for HF8).
fn fp8_to_fixpoint(x: u8, fmt: AceFp8) -> i64 {
    let (exp, frac, mant_bits) = match fmt {
        AceFp8::Bf8 => ((x & 0x7C) >> 2, x & 0x03, 2),
        AceFp8::Hf8 => ((x & 0x78) >> 3, x & 0x07, 3),
    };
    let mant = if exp == 0 { frac as i64 } else { (frac | (1 << mant_bits)) as i64 };
    let mag = mant << if exp == 0 { 0 } else { exp - 1 };
    if x & 0x80 != 0 { -mag } else { mag }
}

/// `convert_fixpoint128_scaled_to_fp32_ftz_rne` (§16.5): `x * 2^adjust` rounded to
/// f32 (RNE), flushing tiny results to signed zero and saturating to ±Inf.
fn fixpoint_to_f32(x: i128, adjust: i32) -> f32 {
    if x == 0 {
        return 0.0;
    }
    let sign = if x < 0 { 1u32 << 31 } else { 0 };
    let mag = x.unsigned_abs();
    let lz = mag.leading_zeros() as i32;
    let jbit = 127 - lz;
    let norm = mag << (lz - 1);
    let sticky = norm & ((1u128 << 102) - 1) != 0;
    let g = (norm >> 102) & 1 == 1;
    let l = (norm >> 103) & 1 == 1;
    let rounded = (norm >> 103) as u32 + (g && (l || sticky)) as u32;
    let ovf = rounded >> 24;
    let biased = 127 + jbit + adjust + ovf as i32;
    if biased > 254 {
        f32::from_bits(sign | 0x7F80_0000)
    } else if biased <= 0 {
        f32::from_bits(sign)
    } else {
        f32::from_bits(sign | ((biased as u32) << 23) | (rounded & 0x7F_FFFF))
    }
}

/// One MX outer-product step: scale both E8M0 strips, convert the exact
/// sum-of-products `sop * 2^(adjust + sa + sb - 254)`, accumulate.
#[inline]
fn mx_accumulate(acc: f32, sop: i128, adjust: i32, sa: u8, sb: u8) -> f32 {
    if sa == 0xFF || sb == 0xFF {
        return f32::from_bits(ACE_MX_SCALE_NAN);
    }
    let x = fixpoint_to_f32(sop, adjust + sa as i32 + sb as i32 - 254);
    add_daz_ftz(acc, x)
}

/// `TOP4MX{B,H}{B,H}F8PS tmm, a, b, imm8` (§14.1): rank-4 MX FP8 outer product.
///
/// `fa`/`fb` select the element format of each side (`TOP4MXHF8PS` is HF8×HF8);
/// `imm8[1:0]` / `imm8[4:3]` select the A / B scale group in `bsr`.
pub fn ace_top4mxf8ps(
    dst: &mut AceTileF32,
    a: &[u8; ACE_ZMM_BYTES],
    fa: AceFp8,
    b: &[u8; ACE_ZMM_BYTES],
    fb: AceFp8,
    bsr: &AceBsr,
    imm8: u8,
) {
    let factor = match (fa, fb) {
        (AceFp8::Bf8, AceFp8::Bf8) => 32,
        (AceFp8::Hf8, AceFp8::Hf8) => 18,
        _ => 25,
    };
    for m in 0..ACE_TILE_DIM {
        let la = lane(a, m);
        let sa = bsr.a_scale(imm8, m);
        for n in 0..ACE_TILE_DIM {
            let lb = lane(b, n);
            let sop: i128 = (0..ACE_K_INT8)
                .map(|k| fp8_to_fixpoint(la[k], fa) as i128 * fp8_to_fixpoint(lb[k], fb) as i128)
                .sum();
            dst.e[m][n] = mx_accumulate(dst.e[m][n], sop, -factor, sa, bsr.b_scale(imm8, n));
        }
    }
}

/// `TOP4MXBSSPS tmm, a, b, imm8` (§14.2): rank-4 MX INT8 outer product. MX INT8
/// elements carry an implicit 2^-6, so each product carries 2^-12.
pub fn ace_top4mxbssps(
    dst: &mut AceTileF32,
    a: &[i8; ACE_ZMM_BYTES],
    b: &[i8; ACE_ZMM_BYTES],
    bsr: &AceBsr,
    imm8: u8,
) {
    for m in 0..ACE_TILE_DIM {
        let sa = bsr.a_scale(imm8, m);
        for n in 0..ACE_TILE_DIM {
            let sop: i32 = (0..ACE_K_INT8).map(|k| a[m * 4 + k] as i32 * b[n * 4 + k] as i32).sum();
            dst.e[m][n] = mx_accumulate(dst.e[m][n], sop as i128, -12, sa, bsr.b_scale(imm8, n));
        }
    }
}

/// One 16×32 MX FP8 (HF8) block: load the scale strips into BSR group 0, then
/// eight `TOP4MXHF8PS` over consecutive K=4 sub-blocks.
pub fn ace_top_mxfp8_block(
    dst: &mut AceTileF32,
    a: &[u8; ACE_MX_BLOCK_ELEMS],
    a_scale: &[u8; ACE_TILE_DIM],
    b: &[u8; ACE_MX_BLOCK_ELEMS],
    b_scale: &[u8; ACE_TILE_DIM],
) {
    let mut bsr = AceBsr::init();
    bsr.load_group0(a_scale, b_scale);
    for (za, zb) in a.chunks_exact(ACE_ZMM_BYTES).zip(b.chunks_exact(ACE_ZMM_BYTES)) {
        let (za, zb) = (za.try_into().unwrap(), zb.try_into().unwrap());
        ace_top4mxf8ps(dst, za, AceFp8::Hf8, zb, AceFp8::Hf8, &bsr, 0);
    }
}

/// One 16×32 MX INT8 block, as [`ace_top_mxfp8_block`] with `TOP4MXBSSPS`.
pub fn ace_top_mxint8_block(
    dst: &mut AceTileF32,
    a: &[i8; ACE_MX_BLOCK_ELEMS],
    a_scale: &[u8; ACE_TILE_DIM],
    b: &[i8; ACE_MX_BLOCK_ELEMS],
    b_scale: &[u8; ACE_TILE_DIM],
) {
    let mut bsr = AceBsr::init();
    bsr.load_group0(a_scale, b_scale);
    for (za, zb) in a.chunks_exact(ACE_ZMM_BYTES).zip(b.chunks_exact(ACE_ZMM_BYTES)) {
        ace_top4mxbssps(dst, za.try_into().unwrap(), zb.try_into().unwrap(), &bsr, 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn splat<T: Copy, const N: usize>(v: T) -> [T; N] {
        [v; N]
    }

    #[test]
    fn top4bssd_matches_reference() {
        let a: [i8; 64] = std::array::from_fn(|i| ((i as i32 * 37 - 61) % 127) as i8);
        let b: [i8; 64] = std::array::from_fn(|i| ((i as i32 * 13 + 5) % 127 - 40) as i8);
        let mut tile = AceTileI32::zero();
        ace_top4bssd(&mut tile, &a, &b);
        for m in 0..16 {
            for n in 0..16 {
                let want: i32 = (0..4).map(|k| a[m * 4 + k] as i32 * b[n * 4 + k] as i32).sum();
                assert_eq!(tile.e[m][n], want, "({m},{n})");
            }
        }
    }

    #[test]
    fn top4bssd_accumulates_and_wraps() {
        let mut tile = AceTileI32::zero();
        ace_top4bssd(&mut tile, &splat(1), &splat(2));
        ace_top4bssd(&mut tile, &splat(1), &splat(2));
        assert!(tile.e.iter().flatten().all(|&v| v == 16));
        tile.e[0][0] = i32::MAX;
        ace_top4bssd(&mut tile, &splat(1), &splat(1));
        assert_eq!(tile.e[0][0], i32::MIN + 3);
    }

    #[test]
    fn top4buud_zero_extends() {
        let mut t = AceTileI32::zero();
        ace_top4buud(&mut t, &splat(200), &splat(255));
        assert!(t.e.iter().flatten().all(|&v| v == 4 * 200 * 255));
    }

    #[test]
    fn top2bf16ps_rounds_the_pair_once() {
        // acc=1 plus products 2^-24 and 2^-24: adding them one at a time ties back
        // to 1.0 twice, summing the pair first gives 1 + 2^-23.
        let a = [0x3380u16; 32]; // 2^-24
        let b = [0x3F80u16; 32]; // 1.0
        let mut t = AceTileF32 { e: [[1.0; 16]; 16] };
        ace_top2bf16ps(&mut t, &a, &b);
        assert!(t.e.iter().flatten().all(|&v| v == 1.0 + 2f32.powi(-23)));
    }

    #[test]
    fn top2bf16ps_specials() {
        let mut a = [0x3F80u16; 32];
        let mut b = [0x3F80u16; 32];
        // Subnormal bf16 input kept: -0.785 * -1.78 + 5*2^-133 * -2.66e38 (from SDE).
        (a[0], a[1], b[0], b[1]) = (0xBF49, 0x0005, 0xBFE4, 0xFF48);
        a[2] = 0x7F80; // +Inf
        (a[4], a[5]) = (0xFFD7, 0x7FD3); // two NaN products: a1*b1 wins
        let mut t = AceTileF32::zero();
        t.e[3][1] = f32::from_bits(1); // subnormal accumulator kept, tiny vs 2.0
        ace_top2bf16ps(&mut t, &a, &b);
        assert_eq!(t.e[0][0].to_bits(), 0x3FA3_6400);
        assert_eq!(t.e[1][1], f32::INFINITY);
        assert_eq!(t.e[2][1].to_bits(), 0x7FD3_0000);
        assert_eq!(t.e[3][1], 2.0);
        a[3] = 0xFF80; // +Inf + -Inf
        let mut t = AceTileF32::zero();
        ace_top2bf16ps(&mut t, &a, &b);
        assert_eq!(t.e[1][1].to_bits(), ACE_QNAN_INDEFINITE);
    }

    #[test]
    fn fixpoint_conversion_rounds_and_saturates() {
        assert_eq!(fixpoint_to_f32(3, 0), 3.0);
        assert_eq!(fixpoint_to_f32(-5, -2), -1.25);
        assert_eq!(fixpoint_to_f32((1 << 24) + 1, 0), 16777216.0); // tie to even
        assert_eq!(fixpoint_to_f32((1 << 24) + 3, 0), 16777220.0);
        assert_eq!(fixpoint_to_f32(1, 128), f32::INFINITY);
        assert_eq!(fixpoint_to_f32(-1, -127).to_bits(), 0x8000_0000);
        assert_eq!(fixpoint_to_f32(1, -126), f32::MIN_POSITIVE);
    }

    #[test]
    fn fp8_fixpoint_decodes() {
        assert_eq!(fp8_to_fixpoint(0x38, AceFp8::Hf8), 1 << 9); // 1.0
        assert_eq!(fp8_to_fixpoint(0x7E, AceFp8::Hf8), 448 << 9);
        assert_eq!(fp8_to_fixpoint(0x01, AceFp8::Hf8), 1); // 2^-9
        assert_eq!(fp8_to_fixpoint(0x3C, AceFp8::Bf8), 1 << 16); // 1.0
        assert_eq!(fp8_to_fixpoint(0xFB, AceFp8::Bf8), -(57344 << 16));
        assert_eq!(fp8_to_fixpoint(0x01, AceFp8::Bf8), 1); // 2^-16
    }

    #[test]
    fn top_mxfp8_block_hand_computed() {
        // HF8 1.0 x 2.0 over K=32 => 64.0 per cell, at unit and at (2.0 x 0.5) scale.
        let (a, b) = ([0x38u8; 512], [0x40u8; 512]);
        for (sa, sb) in [(127u8, 127u8), (128, 126)] {
            let mut t = AceTileF32::zero();
            ace_top_mxfp8_block(&mut t, &a, &[sa; 16], &b, &[sb; 16]);
            assert!(t.e.iter().flatten().all(|&v| v == 64.0));
        }
    }

    #[test]
    fn top_mxint8_block_carries_implicit_2_pow_minus_12() {
        let a = [64i8; 512]; // 1.0 in MX INT8
        let b = [-32i8; 512]; // -0.5
        let mut t = AceTileF32::zero();
        ace_top_mxint8_block(&mut t, &a, &[127; 16], &b, &[129; 16]);
        assert!(t.e.iter().flatten().all(|&v| v == -64.0)); // 32 * 1.0 * -0.5 * 4
    }

    #[test]
    fn mx_nan_scale_poisons_its_row() {
        let mut sa = [127u8; 16];
        sa[3] = 0xFF;
        let mut t = AceTileF32::zero();
        ace_top_mxint8_block(&mut t, &[64; 512], &sa, &[64; 512], &[127; 16]);
        for m in 0..16 {
            for n in 0..16 {
                if m == 3 {
                    assert_eq!(t.e[m][n].to_bits(), ACE_MX_SCALE_NAN);
                } else {
                    assert_eq!(t.e[m][n], 32.0);
                }
            }
        }
    }

    #[test]
    fn bsr_group_selection() {
        let mut bsr = AceBsr::init();
        let a: [u8; 64] = std::array::from_fn(|i| 100 + (i % 4) as u8);
        let b: [u8; 64] = std::array::from_fn(|i| 120 + (i % 4) as u8);
        bsr.movf(&a, &b);
        assert_eq!(bsr.a_scale((2 << 3) | 2, 5), 102);
        assert_eq!(bsr.b_scale((2 << 3) | 2, 5), 122);
        assert_eq!(bsr.a_scale(3, 0), 103);
        assert_eq!(bsr.b_scale(3 << 3, 15), 123);
    }
}
