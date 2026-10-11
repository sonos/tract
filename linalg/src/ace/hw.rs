//! ACE v1 tile kernels on real instructions.
//!
//! Each routine runs a whole 16×16 tile computation inside one `asm!` block:
//! `LDTILECFG` palette 2 (zeroes the tiles, resets the BSR to 2^0), a loop of
//! ZMM loads feeding one outer product into `tmm0`, then `TILEMOVROW` ×16 to spill
//! the accumulator and `TILERELEASE`. The ACE instructions are emitted as `.byte`
//! sequences (encodings from Intel XED, `xed64 -64 -e`) as no released assembler
//! accepts the mnemonics yet.
//!
//! Callers must check [`super::has_ace`] first. Until silicon ships, these are
//! exercised under Intel SDE: `sde64 -future-ag -- <test binary> ace::`.

use super::isa::{ACE_MX_BLOCK_ELEMS, ACE_TILE_DIM};

#[repr(C, align(64))]
struct TileCfg([u8; 64]);

static PALETTE_ACE: TileCfg = {
    let mut cfg = [0u8; 64];
    cfg[0] = 2;
    TileCfg(cfg)
};

/// `TILEMOVROW zmm0, tmm0, imm8` then store `zmm0` to row `imm8` of `{c}`.
macro_rules! spill_tmm0 {
    ($($row:literal)*) => {
        concat!($(
            ".byte 0x62,0xf3,0x7d,0x48,0x07,0xc0,", $row, "\n",
            "vmovdqu64 [{c} + 64*", $row, "], zmm0\n",
        )*)
    };
}

/// Generates a kernel accumulating `steps` outer products of consecutive 64-byte
/// ZMM operands from `pa` (rows) and `pb` (columns) into a zeroed tile, written
/// row-major to `out`.
macro_rules! outer_product_kernel {
    ($(#[$doc:meta])* $name:ident, $acc:ty, $op:literal) => {
        $(#[$doc])*
        #[target_feature(enable = "avx512f")]
        pub unsafe fn $name(
            pa: *const u8,
            pb: *const u8,
            steps: usize,
            out: &mut [[$acc; ACE_TILE_DIM]; ACE_TILE_DIM],
        ) {
            unsafe {
                std::arch::asm!(
                    ".byte 0xc4,0xe2,0x78,0x49,0x07", // ldtilecfg [rdi]
                    "test {n}, {n}",
                    "jz 3f",
                    "2:",
                    "vmovdqu64 zmm0, [{a}]",
                    "vmovdqu64 zmm1, [{b}]",
                    $op,
                    "add {a}, 64",
                    "add {b}, 64",
                    "dec {n}",
                    "jnz 2b",
                    "3:",
                    spill_tmm0!(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15),
                    ".byte 0xc4,0xe2,0x78,0x49,0xc0", // tilerelease
                    in("rdi") PALETTE_ACE.0.as_ptr(),
                    a = inout(reg) pa => _,
                    b = inout(reg) pb => _,
                    n = inout(reg) steps => _,
                    c = in(reg) out.as_mut_ptr(),
                    out("zmm0") _,
                    out("zmm1") _,
                    options(nostack),
                );
            }
        }
    };
}

outer_product_kernel!(
    /// `steps` × `TOP4BSSD tmm0, zmm0, zmm1` over PackedI8K4 panels.
    top4bssd_16x16,
    i32,
    ".byte 0x62,0xf2,0x77,0x48,0x5e,0xc0"
);

outer_product_kernel!(
    /// `steps` × `TOP2BF16PS tmm0, zmm0, zmm1` over K=2-inner bf16 panels.
    top2bf16ps_16x16,
    f32,
    ".byte 0x62,0xf2,0x76,0x48,0x5c,0xc0"
);

/// Generates an MX kernel over `blocks` 16×32 MX blocks of `PackedAceScaledBlock`
/// panels: per block, the two 16-byte E8M0 strips are widened to dwords
/// (`VPMOVZXBD`, putting each scale in BSR group 0), loaded with `BSRMOVF`, and
/// eight outer products with `imm8 = 0` consume the block's K=4 sub-blocks.
macro_rules! mx_kernel {
    ($(#[$doc:meta])* $name:ident, $op:literal) => {
        $(#[$doc])*
        #[target_feature(enable = "avx512f")]
        pub unsafe fn $name(
            pa: *const u8,
            pb: *const u8,
            blocks: usize,
            out: &mut [[f32; ACE_TILE_DIM]; ACE_TILE_DIM],
        ) {
            unsafe {
                let scales = blocks * ACE_MX_BLOCK_ELEMS;
                std::arch::asm!(
                    ".byte 0xc4,0xe2,0x78,0x49,0x07", // ldtilecfg [rdi]
                    "test {n}, {n}",
                    "jz 3f",
                    "2:",
                    "vpmovzxbd zmm2, xmmword ptr [{sa}]",
                    "vpmovzxbd zmm3, xmmword ptr [{sb}]",
                    ".byte 0x62,0xf6,0xec,0x48,0x95,0xc3", // bsrmovf bsr0, zmm2, zmm3
                    "add {sa}, 16",
                    "add {sb}, 16",
                    "mov {i:e}, 8",
                    "4:",
                    "vmovdqu64 zmm0, [{a}]",
                    "vmovdqu64 zmm1, [{b}]",
                    $op,
                    "add {a}, 64",
                    "add {b}, 64",
                    "dec {i:e}",
                    "jnz 4b",
                    "dec {n}",
                    "jnz 2b",
                    "3:",
                    spill_tmm0!(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15),
                    ".byte 0xc4,0xe2,0x78,0x49,0xc0", // tilerelease
                    in("rdi") PALETTE_ACE.0.as_ptr(),
                    a = inout(reg) pa => _,
                    b = inout(reg) pb => _,
                    sa = inout(reg) pa.add(scales) => _,
                    sb = inout(reg) pb.add(scales) => _,
                    n = inout(reg) blocks => _,
                    i = out(reg) _,
                    c = in(reg) out.as_mut_ptr(),
                    out("zmm0") _,
                    out("zmm1") _,
                    out("zmm2") _,
                    out("zmm3") _,
                    options(nostack),
                );
            }
        }
    };
}

mx_kernel!(
    /// MX FP8 (E4M3 × E4M3): `TOP4MXHF8PS tmm0, zmm0, zmm1, 0`.
    top4mxhf8ps_16x16,
    ".byte 0x62,0xf3,0x75,0x48,0x8d,0xc0,0x00"
);

mx_kernel!(
    /// MX INT8: `TOP4MXBSSPS tmm0, zmm0, zmm1, 0`.
    top4mxbssps_16x16,
    ".byte 0x62,0xf3,0x77,0x48,0x8f,0xc0,0x00"
);

#[cfg(test)]
mod tests {
    //! Hardware vs. [`super::super::isa`] model, bit for bit. These only do work
    //! where `has_ace()` holds — today, under `sde64 -future-ag`.
    use super::super::has_ace;
    use super::super::isa::*;
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn byte(&mut self) -> u8 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 56) as u8
        }
        fn bytes(&mut self, n: usize) -> Vec<u8> {
            (0..n).map(|_| self.byte()).collect()
        }
    }

    fn zmm(v: &[u8], i: usize) -> &[u8; 64] {
        v[i * 64..][..64].try_into().unwrap()
    }

    fn bits(t: &[[f32; 16]; 16]) -> Vec<u32> {
        t.iter().flatten().map(|x| x.to_bits()).collect()
    }

    #[test]
    fn top4bssd_matches_model() {
        if !has_ace() {
            return;
        }
        let mut rng = Rng(1);
        for steps in [0, 1, 2, 7, 33] {
            let (a, b) = (rng.bytes(steps * 64), rng.bytes(steps * 64));
            let mut want = AceTileI32::zero();
            for s in 0..steps {
                let i8s = |z: &[u8; 64]| z.map(|x| x as i8);
                ace_top4bssd(&mut want, &i8s(zmm(&a, s)), &i8s(zmm(&b, s)));
            }
            let mut got = [[0i32; 16]; 16];
            unsafe { top4bssd_16x16(a.as_ptr(), b.as_ptr(), steps, &mut got) };
            assert_eq!(got, want.e, "steps={steps}");
        }
    }

    #[test]
    fn top2bf16ps_matches_model() {
        if !has_ace() {
            return;
        }
        let mut rng = Rng(2);
        // Biased exponents so products, sums, overflow, underflow, subnormals,
        // infinities and NaNs all occur.
        let bf16 = |rng: &mut Rng| -> [u8; 2] {
            let hi = match rng.byte() % 8 {
                0 => rng.byte() & 0x80,
                1 => 0x7F | (rng.byte() & 0x80),
                2 => 0x3F,
                3 => 0xBF,
                _ => rng.byte(),
            };
            [rng.byte(), hi]
        };
        for steps in [1, 2, 5, 40, 200] {
            let mut operand = || (0..steps * 32).flat_map(|_| bf16(&mut rng)).collect::<Vec<u8>>();
            let (a, b) = (operand(), operand());
            let mut want = AceTileF32::zero();
            for s in 0..steps {
                let lanes = |z: &[u8; 64]| -> [u16; 32] {
                    std::array::from_fn(|i| u16::from_le_bytes([z[2 * i], z[2 * i + 1]]))
                };
                ace_top2bf16ps(&mut want, &lanes(zmm(&a, s)), &lanes(zmm(&b, s)));
            }
            let mut got = [[0f32; 16]; 16];
            unsafe { top2bf16ps_16x16(a.as_ptr(), b.as_ptr(), steps, &mut got) };
            assert_eq!(bits(&got), bits(&want.e), "steps={steps}");
        }
    }

    /// Lays out `blocks` MX blocks the way `PackedAceScaledBlock` does: elements,
    /// then one 16-byte scale strip per block.
    fn mx_panel(rng: &mut Rng, blocks: usize, scale: impl Fn(&mut Rng) -> u8) -> Vec<u8> {
        let mut v = rng.bytes(blocks * ACE_MX_BLOCK_ELEMS);
        v.extend((0..blocks * 16).map(|_| scale(rng)));
        v
    }

    fn check_mx(
        seed: u64,
        hw: unsafe fn(*const u8, *const u8, usize, &mut [[f32; 16]; 16]),
        model: impl Fn(&mut AceTileF32, &[u8; 64], &[u8; 64], &AceBsr),
        scale: impl Fn(&mut Rng) -> u8 + Copy,
    ) {
        let mut rng = Rng(seed);
        for blocks in [1, 2, 5] {
            let a = mx_panel(&mut rng, blocks, scale);
            let b = mx_panel(&mut rng, blocks, scale);
            let elems = blocks * ACE_MX_BLOCK_ELEMS;
            let mut want = AceTileF32::zero();
            for blk in 0..blocks {
                let strip =
                    |p: &[u8]| -> [u8; 16] { p[elems + blk * 16..][..16].try_into().unwrap() };
                let mut bsr = AceBsr::init();
                bsr.load_group0(&strip(&a), &strip(&b));
                for s in 0..8 {
                    model(&mut want, zmm(&a, blk * 8 + s), zmm(&b, blk * 8 + s), &bsr);
                }
            }
            let mut got = [[0f32; 16]; 16];
            unsafe { hw(a.as_ptr(), b.as_ptr(), blocks, &mut got) };
            assert_eq!(bits(&got), bits(&want.e), "blocks={blocks}");
        }
    }

    fn realistic_scale(rng: &mut Rng) -> u8 {
        110 + rng.byte() % 30
    }

    fn extreme_scale(rng: &mut Rng) -> u8 {
        match rng.byte() % 6 {
            0 => 0xFF,
            1 => rng.byte() % 4,
            2 => 250 + rng.byte() % 5,
            _ => rng.byte(),
        }
    }

    fn hf8_model(t: &mut AceTileF32, a: &[u8; 64], b: &[u8; 64], bsr: &AceBsr) {
        ace_top4mxf8ps(t, a, AceFp8::Hf8, b, AceFp8::Hf8, bsr, 0)
    }

    fn mxint8_model(t: &mut AceTileF32, a: &[u8; 64], b: &[u8; 64], bsr: &AceBsr) {
        ace_top4mxbssps(t, &a.map(|x| x as i8), &b.map(|x| x as i8), bsr, 0)
    }

    #[test]
    fn top4mxhf8ps_matches_model() {
        if !has_ace() {
            return;
        }
        check_mx(3, top4mxhf8ps_16x16, hf8_model, realistic_scale);
        check_mx(4, top4mxhf8ps_16x16, hf8_model, extreme_scale);
    }

    #[test]
    fn top4mxbssps_matches_model() {
        if !has_ace() {
            return;
        }
        check_mx(5, top4mxbssps_16x16, mxint8_model, realistic_scale);
        check_mx(6, top4mxbssps_16x16, mxint8_model, extreme_scale);
    }
}
