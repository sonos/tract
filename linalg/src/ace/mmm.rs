//! Emulated ACE matrix-multiply microkernels.
//!
//! These are pure-Rust kernels registered through tract's normal `MMMRustKernel!`
//! path, so they go through the standard packing, fusion, store and (auto-generated)
//! correctness-test machinery. The *only* part that differs from a real ACE kernel
//! is the inner accumulation, which here calls the bit-exact ISA model in
//! [`crate::ace::isa`] instead of emitting `top4bssd`. The packing
//! ([`PackedI8K4`]), tile geometry (16×16), dispatch and numerics are exactly what
//! a hardware ACE kernel will use.
//!
//! Drop-in path: when an assembler can encode ACE, the `AddMatMul` arm's
//! `ace_top4bssd(...)` call is replaced by the real `__tile_top4bssd` intrinsic
//! (or an external `.S.j2`), and nothing else in tract changes.

use super::isa::{
    ACE_BF16_LANES, ACE_I8_BLOCK_BYTES, ACE_MX_BLOCK_ELEMS, ACE_MX_BLOCK_K, ACE_TILE_DIM,
    AceTileF32, AceTileI32, ace_top_mxfp8_block, ace_top_mxint8_block, ace_top2bf16ps,
    ace_top4bssd,
};
use super::packers::{PackedAceBf16K2, PackedAceScaledBlock};
use crate::frame::mmm::*;
use crate::mmm::DynKernel;
use crate::pack::{PackedI8K4, Packing};
use crate::{Ops, ScaleShiftAndRound, Scaler};
use tract_data::prelude::f16;

const MR: usize = ACE_TILE_DIM; // 16
const NR: usize = ACE_TILE_DIM; // 16

/// ACE INT8 LPGEMM inner loop over one tile register (cf. whitepaper Listing 2).
///
/// `pa`/`pb` point at `PackedI8K4` panels (r=16): K is stepped 4 at a time and
/// each K-block is a contiguous 64-byte `16×4` i8 sub-matrix — exactly one ZMM.
/// PackedI8K4 zero-pads the K tail, so iterating `ceil(k/4)` whole blocks is
/// correct for any `k`.
#[inline]
unsafe fn ace_otop_i8_16x16(pa: *const u8, pb: *const u8, k: usize, ab: &mut [[i32; NR]; MR]) {
    unsafe {
        let mut acc = AceTileI32::zero(); // __tile_zero(&acc)
        let blocks = k.div_ceil(super::isa::ACE_K_INT8);
        for kb in 0..blocks {
            // _mm512_load_si512(&A[kb][..]) / &B[kb][..]: 16×4 i8 = 64 bytes each.
            let a = &*(pa.add(kb * ACE_I8_BLOCK_BYTES) as *const [i8; ACE_I8_BLOCK_BYTES]);
            let b = &*(pb.add(kb * ACE_I8_BLOCK_BYTES) as *const [i8; ACE_I8_BLOCK_BYTES]);
            ace_top4bssd(&mut acc, a, b); // __tile_top4bssd(&acc, a, b)  <-- real ACE op here
        }
        for m in 0..MR {
            for n in 0..NR {
                ab[m][n] += acc.e[m][n];
            }
        }
    }
}

/// Framework default packing[0] = i32×i32 K-major. ACE has no i32×i32 compute
/// mode (its int8 path is packing[1]); this exists only so the standard kernel
/// test-suite's `i32i32:0` case is satisfied with a plain reference accumulation.
#[inline]
unsafe fn add_mat_mul_i32_kmajor(pa: *const u8, pb: *const u8, k: usize, ab: &mut [[i32; NR]; MR]) {
    unsafe {
        let a = pa as *const i32;
        let b = pb as *const i32;
        for ik in 0..k {
            let arow = std::slice::from_raw_parts(a.add(MR * ik), MR);
            let brow = std::slice::from_raw_parts(b.add(NR * ik), NR);
            for i in 0..MR {
                for j in 0..NR {
                    ab[i][j] += arow[i] * brow[j];
                }
            }
        }
    }
}

#[inline]
unsafe fn store_i32_16x16<TC: Copy>(tile: &OutputStoreKer, ab: &[[i32; NR]; MR]) {
    unsafe {
        for i in 0..MR {
            for j in 0..NR {
                let loc: *mut TC = tile
                    .ptr
                    .offset(tile.row_byte_stride * i as isize + tile.col_byte_stride * j as isize)
                    as _;
                let val: *const TC = (&ab[i][j]) as *const i32 as _;
                *loc = *val;
            }
        }
    }
}

#[inline]
unsafe fn add_unicast_16x16<T, TO>(ab: &mut [[T; NR]; MR], other: &OutputStoreKer)
where
    T: Copy + 'static + std::ops::AddAssign,
    TO: Copy + num_traits::AsPrimitive<T>,
{
    unsafe {
        for i in 0..MR {
            for j in 0..NR {
                let value: *const TO = other
                    .ptr
                    .offset(other.row_byte_stride * i as isize + other.col_byte_stride * j as isize)
                    as _;
                ab[i][j] += (*value).as_();
            }
        }
    }
}

/// Store an f32 accumulator tile, casting to the requested element width.
#[inline]
unsafe fn store_f32_16x16<TC: Copy + 'static>(tile: &OutputStoreKer, ab: &[[f32; NR]; MR])
where
    f32: num_traits::AsPrimitive<TC>,
{
    use num_traits::AsPrimitive;
    unsafe {
        for i in 0..MR {
            for j in 0..NR {
                let loc: *mut TC = tile
                    .ptr
                    .offset(tile.row_byte_stride * i as isize + tile.col_byte_stride * j as isize)
                    as _;
                *loc = ab[i][j].as_();
            }
        }
    }
}

/// Framework default packing[0] = f32×f32 K-major (the plain-f32 reference path for
/// the bf16/MX kernels). Not an ACE compute mode; mirrors `add_mat_mul_i32_kmajor`.
#[inline]
unsafe fn add_mat_mul_f32_kmajor(pa: *const u8, pb: *const u8, k: usize, ab: &mut [[f32; NR]; MR]) {
    unsafe {
        let a = pa as *const f32;
        let b = pb as *const f32;
        for ik in 0..k {
            let arow = std::slice::from_raw_parts(a.add(MR * ik), MR);
            let brow = std::slice::from_raw_parts(b.add(NR * ik), NR);
            for i in 0..MR {
                for j in 0..NR {
                    ab[i][j] += arow[i] * brow[j];
                }
            }
        }
    }
}

// ---- ACE exotic outer-product matmuls (the isolated SWAP POINTS). Each reads the
// ---- corresponding packer's layout and accumulates one tile; on real ACE the
// ---- single `ace_top*` call is replaced by the instruction (intrinsic or an
// ---- external .S behind cfg(tract_ace) — see ace::detect::has_ace + build.rs).

/// bf16 outer product over a `PackedAceBf16K2` panel pair (K-inner 2).
#[inline]
unsafe fn ace_matmul_bf16(pa: *const u8, pb: *const u8, k: usize, ab: &mut [[f32; NR]; MR]) {
    unsafe {
        let mut tile = AceTileF32::zero();
        for kb in 0..k.div_ceil(2) {
            // one ZMM = 16×2 bf16 = 64 bytes
            let a = &*(pa.add(kb * ACE_BF16_LANES * 2) as *const [u16; ACE_BF16_LANES]);
            let b = &*(pb.add(kb * ACE_BF16_LANES * 2) as *const [u16; ACE_BF16_LANES]);
            ace_top2bf16ps(&mut tile, a, b); // ACE SWAP POINT -> __tile_ttop2bf16ps
        }
        for m in 0..MR {
            for n in 0..NR {
                ab[m][n] += tile.e[m][n];
            }
        }
    }
}

/// MXFP8 (E4M3) block-scaled outer product over a `PackedAceScaledBlock` panel pair.
/// Scales live in the panel tail at `pa.add(nb*512)`; see `PackedAceScaledBlock`.
#[inline]
unsafe fn ace_matmul_mxfp8(pa: *const u8, pb: *const u8, k: usize, ab: &mut [[f32; NR]; MR]) {
    unsafe {
        let nb = k.div_ceil(ACE_MX_BLOCK_K);
        let elements_bytes = nb * ACE_MX_BLOCK_ELEMS; // nb * 512
        let a_scales = pa.add(elements_bytes);
        let b_scales = pb.add(elements_bytes);
        let mut tile = AceTileF32::zero();
        for blk in 0..nb {
            let a = &*(pa.add(blk * ACE_MX_BLOCK_ELEMS) as *const [u8; ACE_MX_BLOCK_ELEMS]);
            let b = &*(pb.add(blk * ACE_MX_BLOCK_ELEMS) as *const [u8; ACE_MX_BLOCK_ELEMS]);
            let asc = &*(a_scales.add(blk * ACE_TILE_DIM) as *const [u8; ACE_TILE_DIM]);
            let bsc = &*(b_scales.add(blk * ACE_TILE_DIM) as *const [u8; ACE_TILE_DIM]);
            ace_top_mxfp8_block(&mut tile, a, asc, b, bsc); // ACE SWAP POINT -> __tile_ttopmxfp8
        }
        for m in 0..MR {
            for n in 0..NR {
                ab[m][n] += tile.e[m][n];
            }
        }
    }
}

/// MXINT8 block-scaled outer product over a `PackedAceScaledBlock` panel pair.
#[inline]
unsafe fn ace_matmul_mxint8(pa: *const u8, pb: *const u8, k: usize, ab: &mut [[f32; NR]; MR]) {
    unsafe {
        let nb = k.div_ceil(ACE_MX_BLOCK_K);
        let elements_bytes = nb * ACE_MX_BLOCK_ELEMS;
        let a_scales = pa.add(elements_bytes);
        let b_scales = pb.add(elements_bytes);
        let mut tile = AceTileF32::zero();
        for blk in 0..nb {
            let a = &*(pa.add(blk * ACE_MX_BLOCK_ELEMS) as *const [i8; ACE_MX_BLOCK_ELEMS]);
            let b = &*(pb.add(blk * ACE_MX_BLOCK_ELEMS) as *const [i8; ACE_MX_BLOCK_ELEMS]);
            let asc = &*(a_scales.add(blk * ACE_TILE_DIM) as *const [u8; ACE_TILE_DIM]);
            let bsc = &*(b_scales.add(blk * ACE_TILE_DIM) as *const [u8; ACE_TILE_DIM]);
            ace_top_mxint8_block(&mut tile, a, asc, b, bsc); // ACE SWAP POINT -> __tile_ttopmxint8
        }
        for m in 0..MR {
            for n in 0..NR {
                ab[m][n] += tile.e[m][n];
            }
        }
    }
}

/// f32-accumulator kernel shared by the bf16/MXFP8/MXINT8 ACE kernels. The const
/// `V` selects the exotic `packing==1` path (0=bf16, 1=mxfp8, 2=mxint8) — it
/// monomorphizes to three distinct fn pointers, so all the fused-op/store epilogue
/// code is written once. `packing==0` is the plain-f32 K-major reference path.
#[inline(never)]
unsafe fn ace_kernel_f32_16x16<const V: usize>(mut pnl: *const FusedKerSpec<f32>) -> isize {
    unsafe {
        let mut ab = [[0f32; NR]; MR];
        loop {
            if pnl.is_null() {
                break;
            }
            match *pnl {
                FusedKerSpec::Done => break,
                FusedKerSpec::Clear => ab = std::mem::zeroed(),
                FusedKerSpec::LoadTile(col_major, _row_major) => {
                    for row in 0..MR {
                        for col in 0..NR {
                            ab[row][col] = *col_major.add(col * MR + row);
                        }
                    }
                }
                FusedKerSpec::ScalarAdd(a) => apply(&mut ab, |x| x + a),
                FusedKerSpec::ScalarMul(a) => apply(&mut ab, |x| x * a),
                FusedKerSpec::ScalarMin(m) => apply(&mut ab, |x| x.min(m)),
                FusedKerSpec::ScalarMax(m) => apply(&mut ab, |x| x.max(m)),
                FusedKerSpec::ScalarSub(m) => apply(&mut ab, |x| m - x),
                FusedKerSpec::ScalarSubF(m) => apply(&mut ab, |x| x - m),
                FusedKerSpec::LeakyRelu(m) => apply(&mut ab, |x| if x > 0.0 { x } else { x * m }),
                FusedKerSpec::PerRowMin(m) => per_row(&mut ab, m, |a, b| a.min(b)),
                FusedKerSpec::PerRowMax(m) => per_row(&mut ab, m, |a, b| a.max(b)),
                FusedKerSpec::PerRowAdd(m) => per_row(&mut ab, m, |a, b| a + b),
                FusedKerSpec::PerRowMul(m) => per_row(&mut ab, m, |a, b| a * b),
                FusedKerSpec::PerRowSub(m) => per_row(&mut ab, m, |a, b| a - b),
                FusedKerSpec::PerRowSubF(m) => per_row(&mut ab, m, |a, b| b - a),
                FusedKerSpec::PerColMin(m) => per_col(&mut ab, m, |a, b| a.min(b)),
                FusedKerSpec::PerColMax(m) => per_col(&mut ab, m, |a, b| a.max(b)),
                FusedKerSpec::PerColAdd(m) => per_col(&mut ab, m, |a, b| a + b),
                FusedKerSpec::PerColMul(m) => per_col(&mut ab, m, |a, b| a * b),
                FusedKerSpec::PerColSub(m) => per_col(&mut ab, m, |a, b| a - b),
                FusedKerSpec::PerColSubF(m) => per_col(&mut ab, m, |a, b| b - a),
                FusedKerSpec::AddRowColProducts(rows, cols) => {
                    for i in 0..MR {
                        for j in 0..NR {
                            ab[i][j] += *rows.add(i) * *cols.add(j);
                        }
                    }
                }
                FusedKerSpec::AddUnicast(other) => match other.item_size {
                    2 => add_unicast_16x16::<f32, f16>(&mut ab, &other),
                    4 => add_unicast_16x16::<f32, f32>(&mut ab, &other),
                    _ => unimplemented!("ACE f32 AddUnicast item_size {}", other.item_size),
                },
                FusedKerSpec::AddMatMul { k, pa, pb, packing } => match (packing, V) {
                    (0, _) => add_mat_mul_f32_kmajor(pa, pb, k, &mut ab),
                    (1, 0) => ace_matmul_bf16(pa, pb, k, &mut ab),
                    (1, 1) => ace_matmul_mxfp8(pa, pb, k, &mut ab),
                    (1, 2) => ace_matmul_mxint8(pa, pb, k, &mut ab),
                    _ => return 1,
                },
                FusedKerSpec::Store(tile) => match tile.item_size {
                    2 => store_f32_16x16::<f16>(&tile, &ab),
                    4 => store_f32_16x16::<f32>(&tile, &ab),
                    _ => unimplemented!("ACE f32 Store item_size {}", tile.item_size),
                },
                _ => return 1,
            };
            pnl = pnl.add(1);
        }
    }
    0
}

/// The kernel entry point: walks the fused-op list for one 16×16 i32 tile. Mirrors
/// `crate::generic::mmm::kernel` for the fusion/store ops (so the standard kernel
/// tests pass), differing only in `AddMatMul`, which routes through the ACE
/// outer-product ISA model over PackedI8K4.
#[inline(never)]
unsafe fn ace_kernel_i32_16x16(mut pnl: *const FusedKerSpec<i32>) -> isize {
    unsafe {
        let mut ab = [[0i32; NR]; MR];
        loop {
            if pnl.is_null() {
                break;
            }
            match *pnl {
                FusedKerSpec::Done => break,
                FusedKerSpec::Clear => ab = std::mem::zeroed(),
                FusedKerSpec::LoadTile(col_major, _row_major) => {
                    for row in 0..MR {
                        for col in 0..NR {
                            ab[row][col] = *col_major.add(col * MR + row);
                        }
                    }
                }
                FusedKerSpec::ScalarAdd(a) => apply(&mut ab, |x| x + a),
                FusedKerSpec::ScalarMul(a) => apply(&mut ab, |x| x * a),
                FusedKerSpec::ScalarMin(m) => apply(&mut ab, |x| if x < m { x } else { m }),
                FusedKerSpec::ScalarMax(m) => apply(&mut ab, |x| if x > m { x } else { m }),
                FusedKerSpec::ScalarSub(m) => apply(&mut ab, |x| m - x),
                FusedKerSpec::ScalarSubF(m) => apply(&mut ab, |x| x - m),
                FusedKerSpec::LeakyRelu(m) => apply(&mut ab, |x| if x > 0 { x } else { x * m }),
                FusedKerSpec::PerRowMin(m) => per_row(&mut ab, m, |a, b| if a < b { a } else { b }),
                FusedKerSpec::PerRowMax(m) => per_row(&mut ab, m, |a, b| if a > b { a } else { b }),
                FusedKerSpec::PerRowAdd(m) => per_row(&mut ab, m, |a, b| a + b),
                FusedKerSpec::PerRowMul(m) => per_row(&mut ab, m, |a, b| a * b),
                FusedKerSpec::PerRowSub(m) => per_row(&mut ab, m, |a, b| a - b),
                FusedKerSpec::PerRowSubF(m) => per_row(&mut ab, m, |a, b| b - a),
                FusedKerSpec::PerColMin(m) => per_col(&mut ab, m, |a, b| if a < b { a } else { b }),
                FusedKerSpec::PerColMax(m) => per_col(&mut ab, m, |a, b| if a > b { a } else { b }),
                FusedKerSpec::PerColAdd(m) => per_col(&mut ab, m, |a, b| a + b),
                FusedKerSpec::PerColMul(m) => per_col(&mut ab, m, |a, b| a * b),
                FusedKerSpec::PerColSub(m) => per_col(&mut ab, m, |a, b| a - b),
                FusedKerSpec::PerColSubF(m) => per_col(&mut ab, m, |a, b| b - a),
                FusedKerSpec::AddRowColProducts(rows, cols) => {
                    for i in 0..MR {
                        for j in 0..NR {
                            ab[i][j] += *rows.add(i) * *cols.add(j);
                        }
                    }
                }
                FusedKerSpec::AddUnicast(other) => match other.item_size {
                    1 => add_unicast_16x16::<i32, i8>(&mut ab, &other),
                    4 => add_unicast_16x16::<i32, i32>(&mut ab, &other),
                    _ => unimplemented!("ACE emu AddUnicast item_size {}", other.item_size),
                },
                FusedKerSpec::ShiftLeft(shift) => apply(&mut ab, |x| x.q_shl(shift)),
                FusedKerSpec::RoundingShiftRight(shift, rp) => {
                    apply(&mut ab, |x| x.q_shr(shift, rp))
                }
                FusedKerSpec::QScale(shift, rp, mult) => {
                    apply(&mut ab, |x| x.q_scale(Scaler::from_fuse_params(shift, rp, mult)))
                }
                FusedKerSpec::AddMatMul { k, pa, pb, packing } => match packing {
                    0 => add_mat_mul_i32_kmajor(pa, pb, k, &mut ab),
                    1 => ace_otop_i8_16x16(pa, pb, k, &mut ab),
                    _ => return 1,
                },
                FusedKerSpec::Store(tile) => match tile.item_size {
                    1 => store_i32_16x16::<u8>(&tile, &ab),
                    2 => store_i32_16x16::<u16>(&tile, &ab),
                    4 => store_i32_16x16::<u32>(&tile, &ab),
                    8 => store_i32_16x16::<u64>(&tile, &ab),
                    _ => unimplemented!(),
                },
            };
            pnl = pnl.add(1);
        }
    }
    0
}

#[inline]
fn apply<T: Copy>(ab: &mut [[T; NR]; MR], f: impl Fn(T) -> T) {
    for i in 0..MR {
        for j in 0..NR {
            ab[i][j] = f(ab[i][j]);
        }
    }
}

#[inline]
unsafe fn per_row<T: Copy>(ab: &mut [[T; NR]; MR], m: *const T, f: impl Fn(T, T) -> T) {
    unsafe {
        for i in 0..MR {
            for j in 0..NR {
                ab[i][j] = f(*m.add(i), ab[i][j]);
            }
        }
    }
}

#[inline]
unsafe fn per_col<T: Copy>(ab: &mut [[T; NR]; MR], m: *const T, f: impl Fn(T, T) -> T) {
    unsafe {
        for i in 0..MR {
            for j in 0..NR {
                ab[i][j] = f(*m.add(j), ab[i][j]);
            }
        }
    }
}

MMMRustKernel! { ace_kernel_i32_16x16 => ace_emu_mmm_i32_16x16<i32>(16, 16)
    packing[1] = i8i8 => |k| k.with_packing(PackedI8K4::new(16), PackedI8K4::new(16));
    quality(ImplementationQuality::Generic)
    store(i8)
}

// The bf16 / MXFP8 / MXINT8 kernels are built as manual `DynKernel`s rather than via
// `MMMRustKernel!`: that macro force-emits an auto proptest whose pure-f32 reference
// with `Approximate` tolerance (rtol 5e-4) would fail for bf16 (~2^-8) and MX. They
// are instead validated by the precision-matched differential tests below (exact ==).
// `supported_predicate` stays the default `|| true` (NOT gated on `has_ace`) so the
// differential tests run on this host; `has_ace` is reserved for the future real-asm
// kernel (the cfg(tract_ace) sibling).
/// Adapter from the `*const`-taking kernel body to the slice-taking `Kernel<f32>`
/// the framework expects. `MMMRustKernel!` generates this wrapper for the i32
/// kernel; the manually-built f32 DynKernels need it explicitly.
unsafe fn ace_rusty_f32<const V: usize>(op: &[FusedKerSpec<f32>]) -> isize {
    unsafe { ace_kernel_f32_16x16::<V>(op.as_ptr()) }
}

/// The f32 ACE kernels implement the float epilogues but not the integer-quant
/// fusions (QScale / RoundingShiftRight / ShiftLeft), which are i32-only and route
/// to the i32 kernel. Declare that so the planner never fuses them onto an f32 ACE
/// kernel (they would otherwise hit the kernel's `_ => return 1`).
fn ace_f32_can_fuse(spec: &FusedSpec) -> bool {
    !matches!(
        spec,
        FusedSpec::QScale(..) | FusedSpec::RoundingShiftRight(..) | FusedSpec::ShiftLeft(..)
    )
}

lazy_static::lazy_static! {
    pub static ref ace_emu_mmm_f32_bf16_16x16: DynKernel<16, 16, f32> = DynKernel::new(
        "ace_emu_mmm_f32_bf16_16x16",
        ace_rusty_f32::<0>,
        f32::packing(16),
        f32::packing(16),
        ImplementationQuality::Generic,
    )
    .with_packing(PackedAceBf16K2::new(16), PackedAceBf16K2::new(16))
    .with_store::<f16>()
    .with_can_fuse(ace_f32_can_fuse);

    pub static ref ace_emu_mmm_mxfp8_16x16: DynKernel<16, 16, f32> = DynKernel::new(
        "ace_emu_mmm_mxfp8_16x16",
        ace_rusty_f32::<1>,
        f32::packing(16),
        f32::packing(16),
        ImplementationQuality::Generic,
    )
    .with_packing(PackedAceScaledBlock::mxfp8(16), PackedAceScaledBlock::mxfp8(16))
    .with_store::<f16>()
    .with_can_fuse(ace_f32_can_fuse);

    pub static ref ace_emu_mmm_mxint8_16x16: DynKernel<16, 16, f32> = DynKernel::new(
        "ace_emu_mmm_mxint8_16x16",
        ace_rusty_f32::<2>,
        f32::packing(16),
        f32::packing(16),
        ImplementationQuality::Generic,
    )
    .with_packing(PackedAceScaledBlock::mxint8(16), PackedAceScaledBlock::mxint8(16))
    .with_store::<f16>()
    .with_can_fuse(ace_f32_can_fuse);
}

/// Opt-in registration of the emulated ACE kernels into an `Ops` registry. Not
/// called from production dispatch — the model would only ever be a slow fallback
/// on real silicon — but available for benchmarking / experimentation and to keep
/// the registration path warm. The auto-generated correctness tests run regardless.
pub fn plug(ops: &mut Ops) {
    ops.mmm_impls.push(ace_emu_mmm_i32_16x16.mmm());
    ops.mmm_impls.push(ace_emu_mmm_f32_bf16_16x16.mmm());
    ops.mmm_impls.push(ace_emu_mmm_mxfp8_16x16.mmm());
    ops.mmm_impls.push(ace_emu_mmm_mxint8_16x16.mmm());
}

#[cfg(test)]
mod f32_kernel_tests {
    use super::super::format::{
        AceMxElem, bf16_to_f32, f32_to_bf16_rne, fp8_e4m3_to_f32, mx_scale_decode,
        quantize_mx_block,
    };
    use super::*;
    use tract_data::prelude::Tensor;

    // Deterministic pseudo-random f32 in [-1, 1).
    fn rnd(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    // Run one 16x16 tile through pack -> kernel -> store, returning the 16x16 output.
    fn run_tile<P: MMMInputFormat>(
        ker: &DynKernel<16, 16, f32>,
        packer: &P,
        a: &Tensor,
        b: &Tensor,
        k: usize,
    ) -> Vec<f32> {
        let ka = k.next_multiple_of(packer.k_alignment());
        let pa = packer.prepare_one(a, 1, 0).unwrap();
        let pb = packer.prepare_one(b, 0, 1).unwrap();
        let mut out = vec![0f32; 256];
        let c = OutputStoreKer {
            ptr: out.as_mut_ptr() as *mut u8,
            row_byte_stride: 64,
            col_byte_stride: 4,
            item_size: 4,
        };
        let ops = [
            FusedKerSpec::Clear,
            FusedKerSpec::AddMatMul {
                k: ka,
                pa: pa.panel_bytes(0, None).unwrap(),
                pb: pb.panel_bytes(0, None).unwrap(),
                packing: 1,
            },
            FusedKerSpec::Store(c),
            FusedKerSpec::Done,
        ];
        assert_eq!(unsafe { (ker.kernel)(&ops) }, 0);
        out
    }

    // Precision-matched references: a (16xk row-major), b (kx16 row-major).
    fn ref_bf16(a: &[f32], b: &[f32], k: usize) -> Vec<f32> {
        let mut out = vec![0f32; 256];
        for m in 0..16 {
            for n in 0..16 {
                let mut acc = 0f32;
                for kk in 0..k {
                    acc += bf16_to_f32(f32_to_bf16_rne(a[m * k + kk]))
                        * bf16_to_f32(f32_to_bf16_rne(b[kk * 16 + n]));
                }
                out[m * 16 + n] = acc;
            }
        }
        out
    }

    fn ref_mx(a: &[f32], b: &[f32], k: usize, elem: AceMxElem) -> Vec<f32> {
        let nb = k.div_ceil(32);
        let mut out = vec![0f32; 256];
        for m in 0..16 {
            for n in 0..16 {
                let mut acc = 0f32;
                for blk in 0..nb {
                    let (mut av, mut bv) = ([0f32; 32], [0f32; 32]);
                    for i in 0..32 {
                        let kk = blk * 32 + i;
                        if kk < k {
                            av[i] = a[m * k + kk];
                            bv[i] = b[kk * 16 + n];
                        }
                    }
                    let (mut ab, mut bb) = ([0u8; 32], [0u8; 32]);
                    let sad = mx_scale_decode(quantize_mx_block(&av, elem, &mut ab));
                    let sbd = mx_scale_decode(quantize_mx_block(&bv, elem, &mut bb));
                    acc += match elem {
                        AceMxElem::MxFp8 => {
                            let mut inner = 0f32;
                            for i in 0..32 {
                                inner += fp8_e4m3_to_f32(ab[i]) * fp8_e4m3_to_f32(bb[i]);
                            }
                            sad * sbd * inner
                        }
                        AceMxElem::MxInt8 => {
                            let mut inner = 0i32;
                            for i in 0..32 {
                                inner += ab[i] as i8 as i32 * (bb[i] as i8 as i32);
                            }
                            sad * sbd * inner as f32
                        }
                    };
                }
                out[m * 16 + n] = acc;
            }
        }
        out
    }

    // K sweep including non-block-multiples and K-tails.
    const KS: &[usize] = &[1, 2, 3, 4, 8, 16, 17, 31, 32, 33, 40, 64];

    #[test]
    fn bf16_kernel_differential() {
        for &k in KS {
            let a = rnd(16 * k, 0x1111 + k as u64);
            let b = rnd(k * 16, 0x9999 + k as u64);
            let at = Tensor::from_shape(&[16, k], &a).unwrap();
            let bt = Tensor::from_shape(&[k, 16], &b).unwrap();
            let p = PackedAceBf16K2::new(16);
            let got = run_tile(&ace_emu_mmm_f32_bf16_16x16, &p, &at, &bt, k);
            assert_eq!(got, ref_bf16(&a, &b, k), "bf16 k={k}");
        }
    }

    #[test]
    fn mxfp8_kernel_differential() {
        for &k in KS {
            let a = rnd(16 * k, 0x2222 + k as u64);
            let b = rnd(k * 16, 0x8888 + k as u64);
            let at = Tensor::from_shape(&[16, k], &a).unwrap();
            let bt = Tensor::from_shape(&[k, 16], &b).unwrap();
            let p = PackedAceScaledBlock::mxfp8(16);
            let got = run_tile(&ace_emu_mmm_mxfp8_16x16, &p, &at, &bt, k);
            assert_eq!(got, ref_mx(&a, &b, k, AceMxElem::MxFp8), "mxfp8 k={k}");
        }
    }

    #[test]
    fn mxint8_kernel_differential() {
        for &k in KS {
            let a = rnd(16 * k, 0x3333 + k as u64);
            let b = rnd(k * 16, 0x7777 + k as u64);
            let at = Tensor::from_shape(&[16, k], &a).unwrap();
            let bt = Tensor::from_shape(&[k, 16], &b).unwrap();
            let p = PackedAceScaledBlock::mxint8(16);
            let got = run_tile(&ace_emu_mmm_mxint8_16x16, &p, &at, &bt, k);
            assert_eq!(got, ref_mx(&a, &b, k, AceMxElem::MxInt8), "mxint8 k={k}");
        }
    }

    // Fused epilogue (Clear -> AddMatMul -> PerRowAdd bias -> ScalarMax relu -> Store).
    #[test]
    fn f32_fused_epilogue() {
        let k = 20;
        let a = rnd(16 * k, 5);
        let b = rnd(k * 16, 6);
        let at = Tensor::from_shape(&[16, k], &a).unwrap();
        let bt = Tensor::from_shape(&[k, 16], &b).unwrap();
        let p = PackedAceBf16K2::new(16);
        let bias: Vec<f32> = (0..16).map(|i| i as f32 * 0.1 - 0.5).collect();
        let pa = p.prepare_one(&at, 1, 0).unwrap();
        let pb = p.prepare_one(&bt, 0, 1).unwrap();
        let mut out = vec![0f32; 256];
        let c = OutputStoreKer {
            ptr: out.as_mut_ptr() as *mut u8,
            row_byte_stride: 64,
            col_byte_stride: 4,
            item_size: 4,
        };
        let ops = [
            FusedKerSpec::Clear,
            FusedKerSpec::AddMatMul {
                k: k.next_multiple_of(2),
                pa: pa.panel_bytes(0, None).unwrap(),
                pb: pb.panel_bytes(0, None).unwrap(),
                packing: 1,
            },
            FusedKerSpec::PerRowAdd(bias.as_ptr()),
            FusedKerSpec::ScalarMax(0.0),
            FusedKerSpec::Store(c),
            FusedKerSpec::Done,
        ];
        assert_eq!(unsafe { (ace_emu_mmm_f32_bf16_16x16.kernel)(&ops) }, 0);
        let base = ref_bf16(&a, &b, k);
        for m in 0..16 {
            for n in 0..16 {
                assert_eq!(out[m * 16 + n], (base[m * 16 + n] + bias[m]).max(0.0), "({m},{n})");
            }
        }
    }

    // ---- Multi-panel coverage: M/N > 16 exercises the packer's panel loop, panel>0
    // ---- stride, and partial-panel (pw<16) zero-padding — none of which the
    // ---- single-tile tests above reach. a is [.,k] row-major, b is [k, n_total].
    fn bf16_cell(a: &[f32], b: &[f32], k: usize, nt: usize, gm: usize, gn: usize) -> f32 {
        let mut acc = 0f32;
        for kk in 0..k {
            acc += bf16_to_f32(f32_to_bf16_rne(a[gm * k + kk]))
                * bf16_to_f32(f32_to_bf16_rne(b[kk * nt + gn]));
        }
        acc
    }
    fn mx_cell(
        a: &[f32],
        b: &[f32],
        k: usize,
        nt: usize,
        gm: usize,
        gn: usize,
        e: AceMxElem,
    ) -> f32 {
        let mut acc = 0f32;
        for blk in 0..k.div_ceil(32) {
            let (mut av, mut bv) = ([0f32; 32], [0f32; 32]);
            for i in 0..32 {
                let kk = blk * 32 + i;
                if kk < k {
                    av[i] = a[gm * k + kk];
                    bv[i] = b[kk * nt + gn];
                }
            }
            let (mut ab, mut bb) = ([0u8; 32], [0u8; 32]);
            let sad = mx_scale_decode(quantize_mx_block(&av, e, &mut ab));
            let sbd = mx_scale_decode(quantize_mx_block(&bv, e, &mut bb));
            acc += match e {
                AceMxElem::MxFp8 => {
                    let mut inner = 0f32;
                    for i in 0..32 {
                        inner += fp8_e4m3_to_f32(ab[i]) * fp8_e4m3_to_f32(bb[i]);
                    }
                    sad * sbd * inner
                }
                AceMxElem::MxInt8 => {
                    let mut inner = 0i32;
                    for i in 0..32 {
                        inner += ab[i] as i8 as i32 * (bb[i] as i8 as i32);
                    }
                    sad * sbd * inner as f32
                }
            };
        }
        acc
    }

    fn run_tile_at(
        ker: &DynKernel<16, 16, f32>,
        pa: &dyn MMMInputValue,
        pb: &dyn MMMInputValue,
        pm: usize,
        pn: usize,
        k_aligned: usize,
    ) -> Vec<f32> {
        let mut out = vec![0f32; 256];
        let c = OutputStoreKer {
            ptr: out.as_mut_ptr() as *mut u8,
            row_byte_stride: 64,
            col_byte_stride: 4,
            item_size: 4,
        };
        let ops = [
            FusedKerSpec::Clear,
            FusedKerSpec::AddMatMul {
                k: k_aligned,
                pa: pa.panel_bytes(pm, None).unwrap(),
                pb: pb.panel_bytes(pn, None).unwrap(),
                packing: 1,
            },
            FusedKerSpec::Store(c),
            FusedKerSpec::Done,
        ];
        assert_eq!(unsafe { (ker.kernel)(&ops) }, 0);
        out
    }

    fn check_multipanel<P: MMMInputFormat>(
        ker: &DynKernel<16, 16, f32>,
        packer: &P,
        k: usize,
        cell: impl Fn(&[f32], &[f32], usize, usize, usize, usize) -> f32,
    ) {
        for &(mt, nt) in &[(32usize, 32usize), (17, 16), (16, 23), (33, 17)] {
            let a = rnd(mt * k, mt as u64 * 7 + 1);
            let b = rnd(k * nt, nt as u64 * 13 + 3);
            let at = Tensor::from_shape(&[mt, k], &a).unwrap();
            let bt = Tensor::from_shape(&[k, nt], &b).unwrap();
            let pa = packer.prepare_one(&at, 1, 0).unwrap();
            let pb = packer.prepare_one(&bt, 0, 1).unwrap();
            let ka = k.next_multiple_of(packer.k_alignment());
            for pm in 0..mt.div_ceil(16) {
                for pn in 0..nt.div_ceil(16) {
                    let got = run_tile_at(ker, &*pa, &*pb, pm, pn, ka);
                    for i in 0..16 {
                        for j in 0..16 {
                            let (gm, gn) = (pm * 16 + i, pn * 16 + j);
                            // padded (mn-tail) lanes must be exactly 0
                            let want =
                                if gm < mt && gn < nt { cell(&a, &b, k, nt, gm, gn) } else { 0.0 };
                            assert_eq!(
                                got[i * 16 + j],
                                want,
                                "mt={mt} nt={nt} panel=({pm},{pn}) cell=({i},{j})"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn bf16_multipanel() {
        check_multipanel(
            &ace_emu_mmm_f32_bf16_16x16,
            &PackedAceBf16K2::new(16),
            40,
            |a, b, k, nt, m, n| bf16_cell(a, b, k, nt, m, n),
        );
    }
    #[test]
    fn mxfp8_multipanel() {
        check_multipanel(
            &ace_emu_mmm_mxfp8_16x16,
            &PackedAceScaledBlock::mxfp8(16),
            40,
            |a, b, k, nt, m, n| mx_cell(a, b, k, nt, m, n, AceMxElem::MxFp8),
        );
    }
    #[test]
    fn mxint8_multipanel() {
        check_multipanel(
            &ace_emu_mmm_mxint8_16x16,
            &PackedAceScaledBlock::mxint8(16),
            40,
            |a, b, k, nt, m, n| mx_cell(a, b, k, nt, m, n, AceMxElem::MxInt8),
        );
    }

    // f16 store + AddUnicast + LoadTile epilogue arms (bypassed by the manual-DynKernel
    // path, so the macro's store/fuse auto-tests don't cover them).
    #[test]
    fn f32_store_f16_unicast_loadtile() {
        let k = 18;
        let a = rnd(16 * k, 21);
        let b = rnd(k * 16, 22);
        let at = Tensor::from_shape(&[16, k], &a).unwrap();
        let bt = Tensor::from_shape(&[k, 16], &b).unwrap();
        let p = PackedAceBf16K2::new(16);
        let pa = p.prepare_one(&at, 1, 0).unwrap();
        let pb = p.prepare_one(&bt, 0, 1).unwrap();
        let ka = k.next_multiple_of(2);
        let base = ref_bf16(&a, &b, k);
        let mm = |c: OutputStoreKer, extra: Option<FusedKerSpec<f32>>| {
            let mut ops = vec![
                FusedKerSpec::Clear,
                FusedKerSpec::AddMatMul {
                    k: ka,
                    pa: pa.panel_bytes(0, None).unwrap(),
                    pb: pb.panel_bytes(0, None).unwrap(),
                    packing: 1,
                },
            ];
            if let Some(e) = extra {
                ops.push(e);
            }
            ops.push(FusedKerSpec::Store(c));
            ops.push(FusedKerSpec::Done);
            assert_eq!(unsafe { (ace_emu_mmm_f32_bf16_16x16.kernel)(&ops) }, 0);
        };

        // (1) f16 store
        let mut o16 = vec![0u16; 256];
        mm(
            OutputStoreKer {
                ptr: o16.as_mut_ptr() as *mut u8,
                row_byte_stride: 32,
                col_byte_stride: 2,
                item_size: 2,
            },
            None,
        );
        for idx in 0..256 {
            assert_eq!(o16[idx], f16::from_f32(base[idx]).to_bits(), "f16 store [{idx}]");
        }

        // (2) AddUnicast (f32) then store
        let add: Vec<f32> = (0..256).map(|i| i as f32 * 0.013 - 1.0).collect();
        let mut o = vec![0f32; 256];
        mm(
            OutputStoreKer {
                ptr: o.as_mut_ptr() as *mut u8,
                row_byte_stride: 64,
                col_byte_stride: 4,
                item_size: 4,
            },
            Some(FusedKerSpec::AddUnicast(OutputStoreKer {
                ptr: add.as_ptr() as *mut u8,
                row_byte_stride: 64,
                col_byte_stride: 4,
                item_size: 4,
            })),
        );
        for idx in 0..256 {
            assert_eq!(o[idx], base[idx] + add[idx], "unicast [{idx}]");
        }

        // (3) LoadTile (col-major) then store -> transpose check
        let tile: Vec<f32> = (0..256).map(|i| i as f32 * 0.5).collect();
        let mut o2 = vec![0f32; 256];
        let c2 = OutputStoreKer {
            ptr: o2.as_mut_ptr() as *mut u8,
            row_byte_stride: 64,
            col_byte_stride: 4,
            item_size: 4,
        };
        let ops = [
            FusedKerSpec::LoadTile(tile.as_ptr(), std::ptr::null()),
            FusedKerSpec::Store(c2),
            FusedKerSpec::Done,
        ];
        assert_eq!(unsafe { (ace_emu_mmm_f32_bf16_16x16.kernel)(&ops) }, 0);
        for r in 0..16 {
            for col in 0..16 {
                assert_eq!(o2[r * 16 + col], tile[col * 16 + r], "loadtile ({r},{col})");
            }
        }
    }

    // The framework default packing[0] = f32 K-major path (full precision, V-independent).
    #[test]
    fn f32_packing0_kmajor() {
        let k = 12;
        let a = rnd(16 * k, 31);
        let b = rnd(k * 16, 32);
        let at = Tensor::from_shape(&[16, k], &a).unwrap();
        let bt = Tensor::from_shape(&[k, 16], &b).unwrap();
        let pf = f32::packing(16);
        let pa = pf.prepare_one(&at, 1, 0).unwrap();
        let pb = pf.prepare_one(&bt, 0, 1).unwrap();
        let mut out = vec![0f32; 256];
        let c = OutputStoreKer {
            ptr: out.as_mut_ptr() as *mut u8,
            row_byte_stride: 64,
            col_byte_stride: 4,
            item_size: 4,
        };
        let ops = [
            FusedKerSpec::Clear,
            FusedKerSpec::AddMatMul {
                k,
                pa: pa.panel_bytes(0, None).unwrap(),
                pb: pb.panel_bytes(0, None).unwrap(),
                packing: 0,
            },
            FusedKerSpec::Store(c),
            FusedKerSpec::Done,
        ];
        assert_eq!(unsafe { (ace_emu_mmm_f32_bf16_16x16.kernel)(&ops) }, 0);
        for m in 0..16 {
            for n in 0..16 {
                let mut w = 0f32;
                for kk in 0..k {
                    w += a[m * k + kk] * b[kk * 16 + n];
                }
                assert_eq!(out[m * 16 + n], w, "({m},{n})");
            }
        }
    }
}
