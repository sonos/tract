//! Emulated ACE (AI Compute Extensions) ISA primitives.
//!
//! A portable, bit-exact *software model* of the AMD/Intel ACE outer-product
//! matrix unit, as specified in "The AI Compute Extensions (ACE) for x86",
//! x86 Ecosystem Advisory Group whitepaper v1.0 (2026-04-15).
//!
//! No ACE silicon and no assembler/intrinsic support exist yet (hardware is not
//! expected before ~2028). This module models the *documented semantics* so that
//! tract's packing layout, kernel structure, dispatch and numerics can be built
//! and validated on today's hardware, with the inner compute swapped for the real
//! instructions once compilers/assemblers support them. Each function below is
//! annotated with the real ACE intrinsic it stands in for.
//!
//! ## ACE state (whitepaper Table 3)
//!   * **8 tile registers**, each `512b × 16 rows` = a **16×16** i32/f32
//!     accumulator (intrinsic type `__tile1024i`, 1024 bytes).
//!   * **1 Block Scale Register**, `1024b` = **8 groups × 16 × 8-bit** OCP MX
//!     scales (one group per tile register, for blocked-register MX kernels).
//!
//! ## Programming model
//! Unlike Intel AMX (which loads tiles from memory), ACE consumes its operands
//! from **two AVX10/AVX-512 ZMM registers** and accumulates into a tile. For the
//! 8-bit form, each ZMM holds a `16×4` sub-matrix of 8-bit data laid out exactly
//! like tract's [`crate::pack::PackedI8K4`] (lane index = `row*4 + kr`), and one
//! `top4bssd` performs the `16×16×4 = 1024`-MAC outer product. This is why ACE
//! reuses tract's existing K=4-inner int8 packing unchanged — no new packer.

use super::format::{bf16_to_f32, fp8_e4m3_to_f32, mx_scale_decode};

/// Tile registers are square, 16×16.
pub const ACE_TILE_DIM: usize = 16;
/// K elements consumed per 8-bit ZMM operand (a `16×4` sub-matrix).
pub const ACE_K_INT8: usize = 4;
/// K elements consumed per bf16 ZMM operand (a `16×2` sub-matrix).
pub const ACE_K_BF16: usize = 2;
/// Bytes in one 8-bit ZMM operand / one K-block of a PackedI8K4 panel (r=16): 16*4.
pub const ACE_I8_BLOCK_BYTES: usize = ACE_TILE_DIM * ACE_K_INT8; // 64 = one ZMM

/// ACE tile register holding a 16×16 **i32** accumulator — the INT8 / MXINT8 form.
/// Models `__tile1024i`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AceTileI32 {
    pub e: [[i32; ACE_TILE_DIM]; ACE_TILE_DIM],
}

impl AceTileI32 {
    /// `__tile_zero(&dst)` — clear a tile register.
    #[inline]
    pub fn zero() -> Self {
        AceTileI32 { e: [[0; ACE_TILE_DIM]; ACE_TILE_DIM] }
    }
    /// `__tile_movrow(tile, row)` — read one 16-wide i32 row of the tile into a ZMM.
    #[inline]
    pub fn movrow(&self, row: usize) -> [i32; ACE_TILE_DIM] {
        self.e[row]
    }
}

/// ACE tile register holding a 16×16 **f32** accumulator — the BF16 / MXFP8 form.
/// Models `__tile1024f` (same 1024-byte tile state, f32 interpretation).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AceTileF32 {
    pub e: [[f32; ACE_TILE_DIM]; ACE_TILE_DIM],
}

impl AceTileF32 {
    #[inline]
    pub fn zero() -> Self {
        AceTileF32 { e: [[0.0; ACE_TILE_DIM]; ACE_TILE_DIM] }
    }
    #[inline]
    pub fn movrow(&self, row: usize) -> [f32; ACE_TILE_DIM] {
        self.e[row]
    }
}

/// `void __tile_top4bssd(__tile1024i *dst, __m512i a, __m512i b)` — ACE INT8
/// outer-product accumulate, signed×signed → i32 (whitepaper Listing 1).
///
/// `a` and `b` each model one 512-bit ZMM register holding a `16×4` i8 sub-matrix
/// in PackedI8K4 lane order (`lane = row*4 + kr`). Accumulates the full
/// `16×16×4` outer product into `dst`. Bit-exact; on real ACE this is a single
/// instruction whose latency is "not dissimilar to its AVX10 cousins".
#[inline]
pub fn ace_top4bssd(
    dst: &mut AceTileI32,
    a: &[i8; ACE_I8_BLOCK_BYTES],
    b: &[i8; ACE_I8_BLOCK_BYTES],
) {
    for m in 0..ACE_TILE_DIM {
        for n in 0..ACE_TILE_DIM {
            let mut acc = dst.e[m][n];
            for kr in 0..ACE_K_INT8 {
                acc += a[m * ACE_K_INT8 + kr] as i32 * b[n * ACE_K_INT8 + kr] as i32;
            }
            dst.e[m][n] = acc;
        }
    }
}

/// `void __tile_top4buud(...)` style unsigned×unsigned INT8 outer product
/// (u8×u8 → i32). NOTE: speculative — the ACE v1 whitepaper names only the signed
/// `top4bssd`; this unsigned variant is extrapolated from the AVX10 `VPDPBUUD`
/// family (every commercial matrix ISA pairs signed/unsigned dot ops) and is not
/// yet part of the published ISA. The dispatch wires only the signed form today.
#[inline]
pub fn ace_top4buud(
    dst: &mut AceTileI32,
    a: &[u8; ACE_I8_BLOCK_BYTES],
    b: &[u8; ACE_I8_BLOCK_BYTES],
) {
    for m in 0..ACE_TILE_DIM {
        for n in 0..ACE_TILE_DIM {
            let mut acc = dst.e[m][n];
            for kr in 0..ACE_K_INT8 {
                acc += a[m * ACE_K_INT8 + kr] as i32 * b[n * ACE_K_INT8 + kr] as i32;
            }
            dst.e[m][n] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// BF16 outer product
// ---------------------------------------------------------------------------

/// Bytes in one bf16 ZMM operand / one K-block of a bf16 K-inner-2 panel (r=16):
/// 16 rows × 2 K × 2 bytes = 64 bytes = one ZMM (= 32 bf16 lanes).
pub const ACE_BF16_LANES: usize = ACE_TILE_DIM * ACE_K_BF16; // 32

/// `void __tile_top2bf16ps(__tile1024f *dst, __m512i a, __m512i b)` — ACE BF16
/// outer-product accumulate, bf16×bf16 → f32 (the tile analog of `TDPBF16PS`).
///
/// `a`,`b` each model one 512-bit ZMM holding a `16×2` bf16 sub-matrix as 32
/// bf16 bit-patterns in lane order `lane = row*2 + kr`. Products are formed in
/// f32 and accumulated into the f32 tile — bit-exact to a "scalar bf16 multiply +
/// f32 accumulate" reference (this is the *only* difference from a pure-f32 GEMM,
/// the same precision the AMX `TDPBF16PS` path exhibits).
#[inline]
pub fn ace_top2bf16ps(dst: &mut AceTileF32, a: &[u16; ACE_BF16_LANES], b: &[u16; ACE_BF16_LANES]) {
    for m in 0..ACE_TILE_DIM {
        for n in 0..ACE_TILE_DIM {
            let mut acc = dst.e[m][n];
            for kr in 0..ACE_K_BF16 {
                acc += bf16_to_f32(a[m * ACE_K_BF16 + kr]) * bf16_to_f32(b[n * ACE_K_BF16 + kr]);
            }
            dst.e[m][n] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// OCP MX block-scaled outer products (the novel ACE capability)
// ---------------------------------------------------------------------------

/// MX block length along K — one E8M0 scale is shared by 32 consecutive K
/// elements (OCP MX v1.0).
pub const ACE_MX_BLOCK_K: usize = 32;
/// Elements in one 16-wide × 32-K MX block, laid out K=4-inner over 8 sub-blocks
/// (i.e. fed to the unit as 8 successive ZMM operands).
pub const ACE_MX_BLOCK_ELEMS: usize = ACE_TILE_DIM * ACE_MX_BLOCK_K; // 512

/// ACE Block Scale Register state (whitepaper Table 3): 8 groups × 16 × 8-bit
/// E8M0 scales — one 16-wide scale group per tile register, enough for an MX
/// blocked-register kernel using all 8 tiles. The scale group for each input is
/// "encoded in the instruction and separately addressable".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AceBlockScaleReg {
    pub group: [[u8; ACE_TILE_DIM]; 8],
}

impl AceBlockScaleReg {
    #[inline]
    pub fn zero() -> Self {
        AceBlockScaleReg { group: [[0; ACE_TILE_DIM]; 8] }
    }
}

/// Index into a K=4-inner-packed 16×32 MX block: element `(row, k)`.
#[inline]
fn mx_k4_index(row: usize, k: usize) -> usize {
    (k / ACE_K_INT8) * (ACE_TILE_DIM * ACE_K_INT8) + row * ACE_K_INT8 + (k % ACE_K_INT8)
}

/// ACE **MXFP8** (E4M3 element) block-scaled outer product over one K=32 MX block.
///
/// `a`/`b` are FP8 bytes for a 16×32 block in K=4-inner order; `a_scale`/`b_scale`
/// are the per-row / per-col E8M0 block scales (one 16-wide scale group each, as
/// held in the Block Scale Register). For each `(m,n)`:
///   `dst[m][n] += scale(a_scale[m]) * scale(b_scale[n]) * Σ_{k<32} a[m,k]·b[n,k]`
/// (whitepaper Figure 3: inline block scale applied at each row/col intersection).
pub fn ace_top_mxfp8_block(
    dst: &mut AceTileF32,
    a: &[u8; ACE_MX_BLOCK_ELEMS],
    a_scale: &[u8; ACE_TILE_DIM],
    b: &[u8; ACE_MX_BLOCK_ELEMS],
    b_scale: &[u8; ACE_TILE_DIM],
) {
    for m in 0..ACE_TILE_DIM {
        let sa = mx_scale_decode(a_scale[m]);
        for n in 0..ACE_TILE_DIM {
            let sb = mx_scale_decode(b_scale[n]);
            let mut acc = 0f32;
            for k in 0..ACE_MX_BLOCK_K {
                acc +=
                    fp8_e4m3_to_f32(a[mx_k4_index(m, k)]) * fp8_e4m3_to_f32(b[mx_k4_index(n, k)]);
            }
            dst.e[m][n] += sa * sb * acc;
        }
    }
}

/// ACE **MXINT8** block-scaled outer product over one K=32 MX block. Same shape
/// as [`ace_top_mxfp8_block`] but the elements are signed 8-bit integers; the
/// integer dot product is formed first, then scaled by the product of the two
/// E8M0 block scales.
pub fn ace_top_mxint8_block(
    dst: &mut AceTileF32,
    a: &[i8; ACE_MX_BLOCK_ELEMS],
    a_scale: &[u8; ACE_TILE_DIM],
    b: &[i8; ACE_MX_BLOCK_ELEMS],
    b_scale: &[u8; ACE_TILE_DIM],
) {
    for m in 0..ACE_TILE_DIM {
        let sa = mx_scale_decode(a_scale[m]);
        for n in 0..ACE_TILE_DIM {
            let sb = mx_scale_decode(b_scale[n]);
            let mut acc = 0i32;
            for k in 0..ACE_MX_BLOCK_K {
                acc += a[mx_k4_index(m, k)] as i32 * b[mx_k4_index(n, k)] as i32;
            }
            dst.e[m][n] += sa * sb * acc as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::format::{f32_to_bf16_rne, fp8_e4m3_to_f32, mx_scale_decode};
    use super::*;

    // A 16×16 outer-product step over K=4 must match a hand-rolled reference.
    #[test]
    fn top4bssd_matches_reference() {
        // Deterministic pseudo-random fill.
        let mut a = [0i8; ACE_I8_BLOCK_BYTES];
        let mut b = [0i8; ACE_I8_BLOCK_BYTES];
        for i in 0..ACE_I8_BLOCK_BYTES {
            a[i] = ((i as i32 * 37 - 61) % 127) as i8;
            b[i] = ((i as i32 * 13 + 5) % 127 - 40) as i8;
        }
        let mut tile = AceTileI32::zero();
        ace_top4bssd(&mut tile, &a, &b);

        for m in 0..ACE_TILE_DIM {
            for n in 0..ACE_TILE_DIM {
                let mut want = 0i32;
                for kr in 0..ACE_K_INT8 {
                    want += a[m * 4 + kr] as i32 * b[n * 4 + kr] as i32;
                }
                assert_eq!(tile.e[m][n], want, "mismatch at ({m},{n})");
            }
        }
    }

    // Accumulation across multiple top4bssd calls is additive (multi-block K).
    #[test]
    fn top4bssd_accumulates() {
        let a = [1i8; ACE_I8_BLOCK_BYTES];
        let b = [2i8; ACE_I8_BLOCK_BYTES];
        let mut tile = AceTileI32::zero();
        ace_top4bssd(&mut tile, &a, &b);
        ace_top4bssd(&mut tile, &a, &b);
        // each call adds sum_{kr} 1*2 = 8 per (m,n); two calls => 16.
        for m in 0..ACE_TILE_DIM {
            for n in 0..ACE_TILE_DIM {
                assert_eq!(tile.e[m][n], 16);
            }
        }
    }

    #[test]
    fn movrow_reads_back() {
        let mut tile = AceTileI32::zero();
        for m in 0..ACE_TILE_DIM {
            for n in 0..ACE_TILE_DIM {
                tile.e[m][n] = (m * 100 + n) as i32;
            }
        }
        assert_eq!(tile.movrow(3)[7], 307);
    }

    // A multi-block bf16 16×16 GEMM via ace_top2bf16ps must be bit-exact to a
    // scalar "round f32->bf16, multiply, accumulate in f32" reference.
    #[test]
    fn top2bf16ps_matches_bf16_reference() {
        const KB: usize = 5; // 5 K-blocks of 2 => K=10
        let k = KB * ACE_K_BF16;
        // generate f32 inputs, store as bf16 patterns in K=2-inner blocks
        let af = |m: usize, kk: usize| ((m as f32) * 0.5 - 3.0 + kk as f32 * 0.25).sin();
        let bf = |n: usize, kk: usize| ((n as f32) * 0.3 + 1.0 - kk as f32 * 0.1).cos();

        let mut tile = AceTileF32::zero();
        for kb in 0..KB {
            let mut a = [0u16; ACE_BF16_LANES];
            let mut b = [0u16; ACE_BF16_LANES];
            for row in 0..ACE_TILE_DIM {
                for kr in 0..ACE_K_BF16 {
                    a[row * ACE_K_BF16 + kr] = f32_to_bf16_rne(af(row, kb * ACE_K_BF16 + kr));
                    b[row * ACE_K_BF16 + kr] = f32_to_bf16_rne(bf(row, kb * ACE_K_BF16 + kr));
                }
            }
            ace_top2bf16ps(&mut tile, &a, &b);
        }

        for m in 0..ACE_TILE_DIM {
            for n in 0..ACE_TILE_DIM {
                let mut want = 0f32;
                for kk in 0..k {
                    let av = bf16_to_f32(f32_to_bf16_rne(af(m, kk)));
                    let bv = bf16_to_f32(f32_to_bf16_rne(bf(n, kk)));
                    want += av * bv;
                }
                assert_eq!(tile.e[m][n], want, "bf16 mismatch at ({m},{n})");
            }
        }
    }

    // MXINT8 block-scaled outer product: feeding K=4-inner-packed data with
    // per-row/col E8M0 scales must equal a logical (row,k) reference.
    #[test]
    fn top_mxint8_block_matches_reference() {
        let a_raw = |m: usize, kk: usize| (((m * 7 + kk * 3) % 23) as i32 - 11) as i8;
        let b_raw = |n: usize, kk: usize| (((n * 5 + kk * 2) % 19) as i32 - 9) as i8;
        let a_scale: [u8; ACE_TILE_DIM] = std::array::from_fn(|m| 120 + m as u8 % 8);
        let b_scale: [u8; ACE_TILE_DIM] = std::array::from_fn(|n| 125 + n as u8 % 5);

        let mut a = [0i8; ACE_MX_BLOCK_ELEMS];
        let mut b = [0i8; ACE_MX_BLOCK_ELEMS];
        for row in 0..ACE_TILE_DIM {
            for kk in 0..ACE_MX_BLOCK_K {
                a[mx_k4_index(row, kk)] = a_raw(row, kk);
                b[mx_k4_index(row, kk)] = b_raw(row, kk);
            }
        }
        let mut tile = AceTileF32::zero();
        ace_top_mxint8_block(&mut tile, &a, &a_scale, &b, &b_scale);

        for m in 0..ACE_TILE_DIM {
            for n in 0..ACE_TILE_DIM {
                let mut acc = 0i32;
                for kk in 0..ACE_MX_BLOCK_K {
                    acc += a_raw(m, kk) as i32 * b_raw(n, kk) as i32;
                }
                let want = mx_scale_decode(a_scale[m]) * mx_scale_decode(b_scale[n]) * acc as f32;
                assert_eq!(tile.e[m][n], want, "mxint8 mismatch at ({m},{n})");
            }
        }
    }

    // MXFP8 (E4M3) block-scaled outer product: K=4-inner-packed fp8 bytes with
    // E8M0 scales must equal a logical reference using the same decode.
    #[test]
    fn top_mxfp8_block_matches_reference() {
        let a_byte = |m: usize, kk: usize| ((m * 11 + kk * 5) % 120) as u8; // avoid 0x7F NaN region
        let b_byte = |n: usize, kk: usize| ((n * 9 + kk * 7) % 120) as u8;
        let a_scale: [u8; ACE_TILE_DIM] = std::array::from_fn(|m| 127 - (m as u8 % 4));
        let b_scale: [u8; ACE_TILE_DIM] = std::array::from_fn(|n| 127 + (n as u8 % 3));

        let mut a = [0u8; ACE_MX_BLOCK_ELEMS];
        let mut b = [0u8; ACE_MX_BLOCK_ELEMS];
        for row in 0..ACE_TILE_DIM {
            for kk in 0..ACE_MX_BLOCK_K {
                a[mx_k4_index(row, kk)] = a_byte(row, kk);
                b[mx_k4_index(row, kk)] = b_byte(row, kk);
            }
        }
        let mut tile = AceTileF32::zero();
        ace_top_mxfp8_block(&mut tile, &a, &a_scale, &b, &b_scale);

        for m in 0..ACE_TILE_DIM {
            for n in 0..ACE_TILE_DIM {
                let mut acc = 0f32;
                for kk in 0..ACE_MX_BLOCK_K {
                    acc += fp8_e4m3_to_f32(a_byte(m, kk)) * fp8_e4m3_to_f32(b_byte(n, kk));
                }
                let want = mx_scale_decode(a_scale[m]) * mx_scale_decode(b_scale[n]) * acc;
                assert_eq!(tile.e[m][n], want, "mxfp8 mismatch at ({m},{n})");
            }
        }
    }

    // ---- Hand-computed tests: the RHS uses literal constants (NOT the decode
    // ---- helpers), so they pin the decode+scale numerics end-to-end through the
    // ---- kernels, not just the K=4-inner layout. (decode helpers themselves are
    // ---- exhaustively ground-truthed in format.rs.)

    #[test]
    fn top_mxfp8_block_hand_computed() {
        // A = 0x38 (E4M3 1.0), B = 0x40 (E4M3 2.0), K=32 => each cell = 32*1.0*2.0 = 64.0.
        let a = [0x38u8; ACE_MX_BLOCK_ELEMS];
        let b = [0x40u8; ACE_MX_BLOCK_ELEMS];
        let s127 = [127u8; ACE_TILE_DIM]; // E8M0 1.0
        let mut t = AceTileF32::zero();
        ace_top_mxfp8_block(&mut t, &a, &s127, &b, &s127);
        for m in 0..ACE_TILE_DIM {
            for n in 0..ACE_TILE_DIM {
                assert_eq!(t.e[m][n], 64.0, "unit scale ({m},{n})");
            }
        }
        // scales 128 (=2.0) x 126 (=0.5): product 1.0, still 64.0 -- pins decode + product.
        let sa = [128u8; ACE_TILE_DIM];
        let sb = [126u8; ACE_TILE_DIM];
        let mut t2 = AceTileF32::zero();
        ace_top_mxfp8_block(&mut t2, &a, &sa, &b, &sb);
        for row in &t2.e {
            for &v in row {
                assert_eq!(v, 64.0);
            }
        }
    }

    #[test]
    fn top_mxint8_block_hand_computed() {
        let a_val = |m: usize, k: usize| ((m + k) % 5) as i8 - 2;
        let b_val = |n: usize, k: usize| ((n * 2 + k) % 7) as i8 - 3;
        let mut a = [0i8; ACE_MX_BLOCK_ELEMS];
        let mut b = [0i8; ACE_MX_BLOCK_ELEMS];
        for row in 0..ACE_TILE_DIM {
            for k in 0..ACE_MX_BLOCK_K {
                a[mx_k4_index(row, k)] = a_val(row, k);
                b[mx_k4_index(row, k)] = b_val(row, k);
            }
        }
        let dot = |m: usize, n: usize| {
            (0..ACE_MX_BLOCK_K).map(|k| a_val(m, k) as i32 * b_val(n, k) as i32).sum::<i32>()
        };
        // a_scale=128 (2.0) x b_scale=126 (0.5): product scale 1.0 => result == integer dot.
        let mut t = AceTileF32::zero();
        ace_top_mxint8_block(&mut t, &a, &[128; ACE_TILE_DIM], &b, &[126; ACE_TILE_DIM]);
        for m in 0..ACE_TILE_DIM {
            for n in 0..ACE_TILE_DIM {
                assert_eq!(t.e[m][n], dot(m, n) as f32, "unit-scale ({m},{n})");
            }
        }
        // a_scale=129 (4.0) x b_scale=123 (2^-4 = 0.0625): product scale 0.25.
        let mut t2 = AceTileF32::zero();
        ace_top_mxint8_block(&mut t2, &a, &[129; ACE_TILE_DIM], &b, &[123; ACE_TILE_DIM]);
        for m in 0..ACE_TILE_DIM {
            for n in 0..ACE_TILE_DIM {
                assert_eq!(t2.e[m][n], dot(m, n) as f32 * 0.25, "0.25-scale ({m},{n})");
            }
        }
    }

    #[test]
    fn top2bf16ps_hand_computed() {
        // bf16(1.5)=0x3FC0, bf16(2.0)=0x4000; over K=2 => 2*(1.5*2.0) = 6.0.
        let a = [0x3FC0u16; ACE_BF16_LANES];
        let b = [0x4000u16; ACE_BF16_LANES];
        let mut t = AceTileF32::zero();
        ace_top2bf16ps(&mut t, &a, &b);
        for row in &t.e {
            for &v in row {
                assert_eq!(v, 6.0);
            }
        }
    }

    // ---- Special-value propagation through the kernels (is_nan/is_infinite, not ==).

    #[test]
    fn mxfp8_nan_scale_poisons_its_row() {
        let a = [0x38u8; ACE_MX_BLOCK_ELEMS]; // 1.0
        let b = [0x40u8; ACE_MX_BLOCK_ELEMS]; // 2.0
        let mut a_scale = [127u8; ACE_TILE_DIM];
        a_scale[3] = 0xFF; // E8M0 NaN scale
        let mut t = AceTileF32::zero();
        ace_top_mxfp8_block(&mut t, &a, &a_scale, &b, &[127; ACE_TILE_DIM]);
        for n in 0..ACE_TILE_DIM {
            assert!(t.e[3][n].is_nan(), "row 3 col {n} should be NaN");
        }
        for m in 0..ACE_TILE_DIM {
            if m != 3 {
                assert_eq!(t.e[m][0], 64.0);
            }
        }
    }

    #[test]
    fn mxfp8_nan_element_poisons_its_row() {
        let mut a = [0x38u8; ACE_MX_BLOCK_ELEMS];
        a[mx_k4_index(5, 0)] = 0x7F; // E4M3 NaN element in row 5
        let b = [0x40u8; ACE_MX_BLOCK_ELEMS];
        let s = [127u8; ACE_TILE_DIM];
        let mut t = AceTileF32::zero();
        ace_top_mxfp8_block(&mut t, &a, &s, &b, &s);
        for n in 0..ACE_TILE_DIM {
            assert!(t.e[5][n].is_nan());
        }
        assert_eq!(t.e[0][0], 64.0);
    }

    #[test]
    fn bf16_inf_and_nan_propagate() {
        let b = [0x3F80u16; ACE_BF16_LANES]; // 1.0
        let mut a = [0x4000u16; ACE_BF16_LANES]; // 2.0
        a[2 * ACE_K_BF16] = 0x7F80; // +Inf lane in row 2
        let mut t = AceTileF32::zero();
        ace_top2bf16ps(&mut t, &a, &b);
        for n in 0..ACE_TILE_DIM {
            assert!(t.e[2][n].is_infinite() && t.e[2][n] > 0.0);
        }
        assert_eq!(t.e[0][0], 4.0); // 2*(2.0*1.0)
        let mut a2 = [0x4000u16; ACE_BF16_LANES];
        a2[4 * ACE_K_BF16] = 0x7FC0; // qNaN lane in row 4
        let mut t2 = AceTileF32::zero();
        ace_top2bf16ps(&mut t2, &a2, &b);
        for n in 0..ACE_TILE_DIM {
            assert!(t2.e[4][n].is_nan());
        }
    }

    #[test]
    fn mxint8_nan_scale_poisons_its_row() {
        let a = [1i8; ACE_MX_BLOCK_ELEMS];
        let b = [1i8; ACE_MX_BLOCK_ELEMS];
        let mut a_scale = [128u8; ACE_TILE_DIM]; // 2.0
        a_scale[4] = 0xFF; // NaN
        let mut t = AceTileF32::zero();
        ace_top_mxint8_block(&mut t, &a, &a_scale, &b, &[126; ACE_TILE_DIM]);
        for n in 0..ACE_TILE_DIM {
            assert!(t.e[4][n].is_nan());
        }
        assert_eq!(t.e[0][0], 32.0); // product scale 1.0, integer dot of 32 ones
    }

    #[test]
    fn top4buud_unsigned_dot() {
        // u8 bytes > 127 must zero-extend, not sign-extend.
        let a = [200u8; ACE_I8_BLOCK_BYTES];
        let b = [255u8; ACE_I8_BLOCK_BYTES];
        let mut t = AceTileI32::zero();
        ace_top4buud(&mut t, &a, &b);
        for row in &t.e {
            for &v in row {
                assert_eq!(v, 4 * 200 * 255); // 204000
            }
        }
        // a signed misinterpretation would give (-56)*(-1)*4 = 224 -- very different.
        assert_ne!(4 * 200 * 255, 224);
    }
}
