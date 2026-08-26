use crate::DatumType;
use crate::block_quant::*;
use crate::isa::Arch;
use crate::isa::Isa;
use crate::isa::IsaSet;
use crate::mmm::{Query, Suitable};
use crate::mmm_tiers::MmmTier;
use crate::pack::PackedFormat;
#[cfg(any(tract_avx512vnni, tract_avxvnni, tract_amx_int8))]
use crate::pack::PackedI8K4;

#[cfg(tract_amx_int8)]
use super::amx::PackedAmxA;
#[cfg(tract_amx_bf16)]
use super::amx_bf16::{PackedAmxBf16A, PackedBf16K2};
#[cfg(tract_avx512vnni)]
use super::fma_width::has_dual_avx512_fma;

/// The zmm 16x16 VNNI tile only out-throughputs the ymm 8x8 on cores with two 512-bit FMA
/// ports; elsewhere it is pure tile-padding overhead, so there it must lose every tie to the
/// 8x8. It stays runnable on every VNNI core, so that its tests run there too.
#[cfg(tract_avx512vnni)]
const AVX512VNNI_WIDE_TILE: fn() -> isize = || if has_dual_avx512_fma() { 50 } else { -1 };

/// The AMX bf16 kernel truncates its f32 operands to bf16 (~1/2^8 relative error per multiply),
/// so it is opt-in through TRACT_AMX_BF16 and off by default even where the hardware has it.
/// Being off is a preference, not a capability: the kernel stays runnable wherever the ISA is,
/// so that its tests run there, and instead loses every tie no tier can win back.
#[cfg(tract_amx_bf16)]
const AMX_BF16_OPT_IN: fn() -> isize =
    || if crate::knobs::TRACT_AMX_BF16.get() { 100 } else { crate::isa::NEVER_PREFERRED };
use super::*;

/// One kernel in a dispatcher’s preference table, with its tile geometry
/// and a relative-throughput scale (1.0 = baseline, used to break
/// near-ties between kernels with similar tile waste).
#[derive(Clone, Copy)]
struct KernelChoice {
    mr: usize,
    nr: usize,
    scale: f32,
    name: fn() -> &'static str,
}

/// Fraction of the M-or-N axis covered by useful work after rounding up
/// to the kernel's tile size. 1.0 = exact fit; smaller is worse.
/// Empty axis (d == 0) is treated as "no waste" — no work to misallocate.
fn tile_util(d: usize, tile: usize) -> f32 {
    if d == 0 {
        return 1.0;
    }
    let batches = d.div_ceil(tile);
    d as f32 / (batches * tile) as f32
}

/// Pick the kernel that maximises `scale * m_util * n_util`. Ties are
/// broken first in favour of fewer total tile passes (less loop
/// overhead), then in favour of larger `nr` (more K-loop amortisation
/// per inner iteration). An unknown M or N is treated as
/// "large enough" — its utilisation contribution is 1.0.
fn pick_mmm(choices: &[KernelChoice], m: Option<usize>, n: Option<usize>) -> &'static str {
    let key = |c: &KernelChoice| -> (f32, i32, i32) {
        let m_u = m.map(|m| tile_util(m, c.mr)).unwrap_or(1.0);
        let n_u = n.map(|n| tile_util(n, c.nr)).unwrap_or(1.0);
        let m_b = m.map(|m| m.div_ceil(c.mr)).unwrap_or(1) as i32;
        let n_b = n.map(|n| n.div_ceil(c.nr)).unwrap_or(1) as i32;
        (c.scale * m_u * n_u, -(m_b * n_b), c.nr as i32)
    };
    let best = choices
        .iter()
        .max_by(|a, b| key(a).partial_cmp(&key(b)).unwrap())
        .expect("non-empty kernel pool");
    (best.name)()
}

// AVX-without-FMA f32 tier for pre-Haswell CPUs (Sandy Bridge / Ivy Bridge):
// same tile geometries as their fma_ siblings but the inner loops use
// vmulps+vaddps, and add_unicast avoids the avx2-only vgatherdps.
MMMExternKernel!(x86_64; avx_mmm_f32_8x8 <f32>(8, 8)@(256,4) isa(X86_64Avx));
MMMExternKernel!(x86_64; avx_mmm_f32_16x5<f32>(16,5)@(256,4) isa(X86_64Avx));
MMMExternKernel!(x86_64; avx_mmm_f32_16x6<f32>(16,6)@(256,4) isa(X86_64Avx));
MMMExternKernel!(x86_64; avx_mmm_f32_24x4<f32>(24,4)@(256,4) isa(X86_64Avx));
MMMExternKernel!(x86_64; avx_mmm_f32_32x3<f32>(32,3)@(256,4) isa(X86_64Avx));
MMMExternKernel!(x86_64; avx_mmm_f32_40x2<f32>(40,2)@(256,4) isa(X86_64Avx));
MMMExternKernel!(x86_64; avx_mmm_f32_64x1<f32>(64,1)@(256,4) isa(X86_64Avx));

/// The 256-bit fma f32 kernels are not superseded by the avx512 ones, so they cancel the ladder
/// step between the two and are preferred as peers. Both x86 avx512 cost models score them alongside the
/// avx512 kernels; they cover the small-`n` tiles avx512 has no matching `nr` for (n=2 -> 40x2,
/// n=4 -> 24x4); and avx512 has no 64x1 at all, so a runnable set without them loses the whole r=64
/// packing group, whose 64x3 is the best matrix kernel at small `n`.
const FMA_F32_PEER: fn() -> isize = || crate::isa::peer_of(Isa::X86_64Fma, Isa::X86_64Avx512f);

MMMExternKernel!(x86_64; fma_mmm_f32_8x8 <f32>(8, 8)@(256,4) isa(X86_64Avx, X86_64Fma) boost(FMA_F32_PEER));
MMMExternKernel!(x86_64; fma_mmm_f32_16x6<f32>(16,6)@(256,4) isa(X86_64Avx, X86_64Fma) boost(FMA_F32_PEER));
MMMExternKernel!(x86_64; fma_mmm_f32_16x5<f32>(16,5)@(256,4) isa(X86_64Avx, X86_64Fma) boost(FMA_F32_PEER));
MMMExternKernel!(x86_64; fma_mmm_f32_24x4<f32>(24,4)@(256,4) isa(X86_64Avx, X86_64Fma) boost(FMA_F32_PEER));
MMMExternKernel!(x86_64; fma_mmm_f32_40x2<f32>(40,2)@(256,4) isa(X86_64Avx, X86_64Fma) boost(FMA_F32_PEER));
MMMExternKernel!(x86_64; fma_mmm_f32_64x1<f32>(64,1)@(256,4) isa(X86_64Avx, X86_64Fma) boost(FMA_F32_PEER));

pub fn pq40_r32() -> PackedBlockQuantFormat {
    PackedBlockQuantFormat::new(&Q4_0, 32, 16, false)
}
pub fn pq20t_r32() -> PackedBlockQuantFormat {
    PackedBlockQuantFormat::new(&Q2_0_T, 32, 0, false)
}
MMMExternKernel! { x86_64; fma_mmm_f32_32x1<f32>(32,1)@(256,4) isa(X86_64Avx, X86_64Fma, X86_64F16c)
    packing[1] = q40f32 => |k| k.with_packing_a(pq40_r32());
    packing[2] = q40f16 => |k| k.with_packing(pq40_r32(), f16::packing(1));
    packing[3] = f16f16 => |k| k.with_packing(f16::packing(32), f16::packing(1));
    packing[4] = f16f32 => |k| k.with_packing(f16::packing(32), f32::packing(1));
    packing[5] = f32f16 => |k| k.with_packing(f32::packing(32), f16::packing(1));

    boost(FMA_F32_PEER)
    store(f16)
}
MMMExternKernel!(x86_64; fma_mmm_f32_32x3<f32>(32,3)@(256,4) isa(X86_64Avx, X86_64Fma)
 packing[1] = f32f16 => |k| k.with_packing(f32::packing(32).align(256), f16::packing(3));
 packing[2] = f16f32 => |k| k.with_packing(f16::packing(32).align(256), f32::packing(3));
 packing[3] = f16f16 => |k| k.with_packing(f16::packing(32).align(256), f16::packing(3));

 boost(FMA_F32_PEER)
 store(f16)
);

MMMExternKernel!(x86_64; avx512_mmm_f32_128x1<f32>(128, 1)@(512,4) isa(X86_64Avx512f));
MMMExternKernel!(x86_64; avx512_mmm_f32_16x1 <f32>( 16, 1)@(512,4) isa(X86_64Avx512f));
MMMExternKernel!(x86_64; avx512_mmm_f32_16x12<f32>( 16,12)@(512,4) isa(X86_64Avx512f));
MMMExternKernel!(x86_64; avx512_mmm_f32_16x8 <f32>( 16, 8)@(512,4) isa(X86_64Avx512f));
MMMExternKernel!(x86_64; avx512_mmm_f32_32x6 <f32>( 32, 6)@(512,4) isa(X86_64Avx512f));
MMMExternKernel!(x86_64; avx512_mmm_f32_32x5 <f32>( 32, 5)@(512,4) isa(X86_64Avx512f));
MMMExternKernel!(x86_64; avx512_mmm_f32_48x4 <f32>( 48, 4)@(512,4) isa(X86_64Avx512f));
MMMExternKernel!(x86_64; avx512_mmm_f32_64x3 <f32>( 64, 3)@(512,4) isa(X86_64Avx512f));
MMMExternKernel!(x86_64; avx512_mmm_f32_80x2 <f32>( 80, 2)@(512,4) isa(X86_64Avx512f));

// 128-bit VEX i32 sibling of avx2_mmm_i32_8x8 for the avx-without-avx2 tier:
// same i8i8 widening scheme (i8 products computed in i16 lanes) and the same
// quantization epilogue semantics, on 8x4 xmm column pairs.
MMMExternKernel! { x86_64; avx_mmm_i32_8x4<i32>(8,4)@(256,4) isa(X86_64Avx)
    packing[1] = i8i8 => |k| k.with_packing(PackedFormat::new(DatumType::I8, 8, 256), PackedFormat::new(DatumType::I8, 4, 4));

    store(i8)
}

MMMExternKernel! { x86_64; avx2_mmm_i32_8x8<i32>(8,8)@(256,4) isa(X86_64Avx2)
    packing[1] = i8i8 => |k| k.with_packing(PackedFormat::new(DatumType::I8, 8, 256), PackedFormat::new(DatumType::I8, 8, 4));

    store(i8)
}

// AVX-512 VNNI int8 GEMM: same 8x8 column-accumulator tile and quantization
// epilogue as avx2_mmm_i32_8x8, but the i8i8 matmul inner loop uses VPDPBUSD
// (4-way K dot) over the K=4-inner PackedI8K4 layout. VPDPBUSD is u8*s8, so the
// kernel offsets A by +128 and removes the 128*sum_k(B) bias per column before
// the epilogue, making the i32 accumulators bit-identical to the AVX2 path.
//
// Gated on `tract_avx512vnni` (set by build.rs when the assembler can encode
// `vpdpbusd ymm`; binutils < 2.30 cannot). On old toolchains the kernel is
// omitted entirely and the AVX2 i32 path is used instead.
#[cfg(tract_avx512vnni)]
MMMExternKernel! { x86_64; avx512vnni_mmm_i32_8x8<i32>(8,8)@(256,4) isa(X86_64Avx512f, X86_64Avx512Vnni)
    packing[1] = i8i8 => |k| k.with_packing(PackedI8K4::new(8), PackedI8K4::new(8));

    store(i8)
}

// AVX-512 VNNI int8 GEMM, zmm-wide 16x16 sibling of avx512vnni_mmm_i32_8x8.
// Accumulators are ROW-MAJOR (zmm{m} = row m of C, 16 columns per zmm), so one
// VPDPBUSD covers 16 columns x 4 K and the K=4 inner step issues 16 of them
// (one per row) = 1024 mul-adds/block, 2x the 8x8 ymm kernel's work per
// iteration. Same +128 A-bias / per-column correction as the 8x8 kernel, and
// the same PackedI8K4 layout (r=16 for both A and B). This is the int8
// throughput tier of qmmm_i32 for cores with AVX-512-VNNI but no AMX *and two
// 512-bit FMA ports* (Cascade Lake / Cooper Lake / Ice Lake-SP servers). The
// 2x work/iteration only turns into 2x throughput when the core has two
// 512-bit FMA units to retire two VPDPBUSD/zmm per cycle; single-512-FMA
// client cores (Ice Lake-U / Tiger Lake / Rocket Lake) get no gain and stay on
// the 8x8 ymm kernel -- see `has_dual_avx512_fma()` in `fma_width.rs`.
//
// On dual-FMA cores the boost lifts it above the 8x8 VNNI kernel in the einsum
// kernel-selection scorer for unknown shapes, while staying below the AMX 16x16
// kernels' boost(100) so AMX still wins when both are present; on single-FMA cores
// it goes negative so the 8x8 wins instead.
#[cfg(tract_avx512vnni)]
MMMExternKernel! { x86_64; avx512vnni_mmm_i32_16x16<i32>(16,16)@(64,4) isa(X86_64Avx512f, X86_64Avx512Vnni)
    packing[1] = i8i8 => |k| k.with_packing(PackedI8K4::new(16), PackedI8K4::new(16));

    boost(AVX512VNNI_WIDE_TILE)
    store(i8)
}

// AVX-VNNI ymm int8 GEMM: byte-for-byte the same body as avx512vnni_mmm_i32_8x8
// (8x8 ymm accumulators, PackedI8K4 inner-K, +128 bias trick), but the
// VPDPBUSD instructions are forced to the VEX (AVX-VNNI) encoding via the
// `{vex}` prefix. Runs on Atom-class cores (Alder Lake-E, Sierra Forest,
// Clearwater Forest / Darkmont) which have AVX-VNNI but no AVX-512. On big
// cores with both AVX-512-VNNI and AVX-VNNI (Sapphire Rapids+, some Alder
// Lake P-core SKUs) dispatch prefers the EVEX-encoded kernel above.
#[cfg(tract_avxvnni)]
MMMExternKernel! { x86_64; avxvnni_mmm_i32_8x8<i32>(8,8)@(256,4) isa(X86_64Avx2, X86_64AvxVnni)
    packing[1] = i8i8 => |k| k.with_packing(PackedI8K4::new(8), PackedI8K4::new(8));

    store(i8)
}

// Same epilogue as avx512vnni_mmm_i32_8x8 (8x8 ymm accumulators), but the i8i8
// matmul inner loop uses TDPBSSD (16-M x 16-N x 64-K mul-acc per instruction)
// over AMX tiles. A's packing is novel (PackedAmxA, M-major-within-panel,
// K-padded to multiples of 64); B reuses VNNI's K=4-inner PackedI8K4 layout
// unchanged. TDPBSSD is s8 x s8 so no +128 bias trick — accumulators are
// bit-identical to AVX2/VNNI. Gated by `isa(X86_64Avx512f, X86_64AmxInt8)` (= CPUID amx-int8
// AND Linux XSAVE permission via arch_prctl).
#[cfg(tract_amx_int8)]
MMMExternKernel! { x86_64; avx512amx_mmm_i32_8x8<i32>(8,8)@(64,4) isa(X86_64Avx512f, X86_64AmxInt8)
    packing[1] = i8i8 => |k| k.with_packing(PackedAmxA::new(8), PackedI8K4::new(8));

    store(i8)
}

// 16x16 i32 sibling. One tdpbssd does 16*16*64 = 16384 mul-adds (4x the 8x8).
// Same A/B packing (PackedAmxA, PackedI8K4) just with r=16. Row-major
// accumulators (zmm{m} = row m of C) so the hot path (Clear -> AddMatMul ->
// Store) needs no transpose.
//
// boost(100) pushes this kernel above the AMX 8x8 kernel it shares an instruction-set level
// with, and above the AVX-512-VNNI 8x8 a level below it, when einsum picks among the suitable
// kernels. When more than one dim is symbolic the shape-adaptive `qmmm_i32` picker isn't
// invoked, so the boost is what causes the optimizer to prefer the 16x16 tile for
// unknown-shape matmuls.
#[cfg(tract_amx_int8)]
MMMExternKernel! { x86_64; avx512amx_mmm_i32_16x16<i32>(16,16)@(64,4) isa(X86_64Avx512f, X86_64AmxInt8)
    packing[1] = i8i8 => |k| k.with_packing(PackedAmxA::new(16), PackedI8K4::new(16));

    boost(|| 100)
    store(i8)
}

// AMX bf16 16x16 kernel for f32 matmul: uses TDPBF16PS (bf16 x bf16 -> f32).
// f32 inputs are truncated to bf16 at pack time (round-to-nearest-even, matching
// Intel VCVTNEPS2BF16). One tdpbf16ps consumes 16M x 16N x 32K bf16 = 8192 fma
// per instruction. f32 accumulators differ from a pure-f32 reference by ~1/2^8
// relative per multiply (bf16 = 8 mantissa bits vs f32's 23) -- same precision
// loss profile as oneDNN "fast-math" f32 matmul on AMX, acceptable for
// inference workloads (LLMs, CNNs) that already tolerate bf16.
//
// Default packing[0] (the framework's PackedFormat<f32>) is retained so the
// kernel can still be selected for f32 paths even when the BF16 packer
// isn't a precursor match; packing[1] is the fast bf16-from-f32 path.
// Once opted in, the boost puts this kernel above the AVX-512 f32 and FMA f32 kernels so the
// einsum prefers it, mirroring the i32 16x16 behaviour; see `AMX_BF16_OPT_IN` for the
// default-off half.
#[cfg(tract_amx_bf16)]
MMMExternKernel! { x86_64; avx512amx_mmm_f32_16x16<f32>(16,16)@(64,4) isa(X86_64Avx512f, X86_64AmxBf16)
    packing[1] = f32f32_bf16 => |k| k.with_packing(PackedAmxBf16A::new(16), PackedBf16K2::new(16));

    boost(AMX_BF16_OPT_IN)
}

/// The zmm 16x16 kernel does 2x the work per inner iteration as the ymm 8x8, but that only
/// becomes 2x the *throughput* on cores with two 512-bit FMA ports (Cascade Lake / Cooper Lake /
/// Ice Lake-SP and later Xeons). On a single-512-FMA client core (Ice Lake-U / Tiger Lake /
/// Rocket Lake) one 512-bit VPDPBUSD/cycle delivers the same MAC/s as two 256-bit
/// VPDPBUSD/cycle, so the wider tile is pure overhead (extra A-packing, the 16-column +128 bias
/// correction, a bigger epilogue) and regresses real matmuls -- e.g. -4..-11% on int8 LLM/TDNN
/// prefill on an i9-11900KB. So this tier names the 16x16 only on dual-FMA cores; the einsum
/// scorer is held off it by `AVX512VNNI_WIDE_TILE` going negative elsewhere.
///
/// Where it does name the wide tile the dispatch is shape-adaptive, mirroring the AMX int8 path:
/// the 16x16 is the throughput champion when each of M and N fills at least one tile, and the
/// 8x8 has lower per-call setup and wins on small problems where tile padding dominates. Unknown
/// dims default to the champion. (No K gate: one VPDPBUSD step is only 4 K-bytes, so any K is
/// fine; the choice is about filling the 16-wide M/N tile.)
#[cfg(tract_avx512vnni)]
fn avx512vnni_preferred(
    _isa: &IsaSet,
    dt: DatumType,
    query: &Query,
    _suitable: &[Suitable],
) -> Option<&'static str> {
    match (dt, query.n) {
        (DatumType::I32, Some(1)) => None,
        (DatumType::I32, _) if !has_dual_avx512_fma() => Some(avx512vnni_mmm_i32_8x8.name.as_str()),
        (DatumType::I32, _) => {
            let big = |o: Option<usize>, t: usize| o.is_none_or(|v| v >= t);
            Some(if big(query.m, 16) && big(query.n, 16) {
                avx512vnni_mmm_i32_16x16.name.as_str()
            } else {
                avx512vnni_mmm_i32_8x8.name.as_str()
            })
        }
        _ => None,
    }
}

#[cfg(tract_avx512vnni)]
inventory::submit! {
    MmmTier {
        arch: Some(Arch::X86_64),
        precedence: 6,
        name: "avx512vnni",
        applies: |isa| {
            isa.has(Isa::X86_64Avx2) && isa.has(Isa::X86_64Fma) && isa.has(Isa::X86_64Avx512f)
                && isa.has(Isa::X86_64Avx512Vnni)
        },
        preferred: avx512vnni_preferred,
    }
}

/// On AVX-VNNI-only cores (no AVX-512) this is the int8 throughput champion, over the AVX2
/// emulation below it. Big cores that also have AVX-512-VNNI answer from the higher tier.
#[cfg(tract_avxvnni)]
fn avxvnni_preferred(
    _isa: &IsaSet,
    dt: DatumType,
    query: &Query,
    _suitable: &[Suitable],
) -> Option<&'static str> {
    match (dt, query.n) {
        (DatumType::I32, Some(1)) => None,
        (DatumType::I32, _) => Some(avxvnni_mmm_i32_8x8.name.as_str()),
        _ => None,
    }
}

#[cfg(tract_avxvnni)]
inventory::submit! {
    MmmTier {
        arch: Some(Arch::X86_64),
        precedence: 3,
        name: "avxvnni",
        applies: |isa| isa.has(Isa::X86_64Avx2) && isa.has(Isa::X86_64AvxVnni),
        preferred: avxvnni_preferred,
    }
}

/// The AMX bf16 tile, where it is a good fit: not at small M/N, and not below K 32 — one
/// TDPBF16PS consumes 32 bf16 K-lanes so the panel must have at least one full step. The
/// threshold matches `PackedAmxBf16A::k_alignment()`; below it the AVX-512 / FMA tiers' smaller
/// tiles waste less work, and this tier declines so they answer.
#[cfg(tract_amx_bf16)]
fn amx_bf16_preferred(
    _isa: &IsaSet,
    dt: DatumType,
    query: &Query,
    _suitable: &[Suitable],
) -> Option<&'static str> {
    if dt != DatumType::F32 || query.n == Some(1) {
        return None;
    }
    let big = |o: Option<usize>, t: usize| o.is_none_or(|v| v >= t);
    if big(query.m, 16) && big(query.n, 16) && big(query.k, 32) {
        Some(avx512amx_mmm_f32_16x16.name.as_str())
    } else {
        None
    }
}

#[cfg(tract_amx_bf16)]
inventory::submit! {
    MmmTier {
        arch: Some(Arch::X86_64),
        precedence: 8,
        name: "avx512amx-bf16",
        applies: |isa| {
            crate::knobs::TRACT_AMX_BF16.get()
                && isa.has(Isa::X86_64Avx2)
                && isa.has(Isa::X86_64Fma)
                && isa.has(Isa::X86_64Avx512f)
                && isa.has(Isa::X86_64AmxBf16)
        },
        preferred: amx_bf16_preferred,
    }
}

/// Shape-adaptive AMX int8: the 16x16 hits the full tile (1024 B/tile, 16384 mul-adds per
/// tdpbssd) and is the throughput champion when at least one tile of each dim is fully utilised;
/// the 8x8 has lower per-call setup (1/4 the tile-store scratch, half the prefetch budget,
/// smaller epilogue) and beats it on small problems where tile padding dominates. An unknown dim
/// defaults to the champion. The exact crossover should be re-validated on AMX HW; oneDNN uses
/// similar shape-based MR/NR selection for its BRGEMM ukernel variants.
#[cfg(tract_amx_int8)]
fn amx_int8_preferred(
    _isa: &IsaSet,
    dt: DatumType,
    query: &Query,
    _suitable: &[Suitable],
) -> Option<&'static str> {
    match (dt, query.n) {
        (DatumType::I32, Some(1)) => None,
        (DatumType::I32, _) => {
            let big = |o: Option<usize>, t: usize| o.is_none_or(|v| v >= t);
            Some(if big(query.m, 16) && big(query.n, 16) && big(query.k, 64) {
                avx512amx_mmm_i32_16x16.name.as_str()
            } else {
                avx512amx_mmm_i32_8x8.name.as_str()
            })
        }
        _ => None,
    }
}

#[cfg(tract_amx_int8)]
inventory::submit! {
    MmmTier {
        arch: Some(Arch::X86_64),
        precedence: 7,
        name: "avx512amx-int8",
        applies: |isa| {
            isa.has(Isa::X86_64Avx2)
                && isa.has(Isa::X86_64Fma)
                && isa.has(Isa::X86_64Avx512f)
                && isa.has(Isa::X86_64Avx512Vnni)
                && isa.has(Isa::X86_64AmxInt8)
        },
        preferred: amx_int8_preferred,
    }
}

fn avx2_preferred(
    _isa: &IsaSet,
    dt: DatumType,
    query: &Query,
    _suitable: &[Suitable],
) -> Option<&'static str> {
    match (dt, query.n) {
        (DatumType::I32, Some(1)) => None,
        (DatumType::I32, _) => Some(&mmm::avx2_mmm_i32_8x8.name.as_str()),
        _ => None,
    }
}

inventory::submit! {
    MmmTier {
        arch: Some(Arch::X86_64),
        precedence: 2,
        name: "avx2",
        applies: |isa| isa.has(Isa::X86_64Avx2),
        preferred: avx2_preferred,
    }
}

const AVX_CHOICES: &[KernelChoice] = &[
    KernelChoice { mr: 16, nr: 6, scale: 1.0, name: || avx_mmm_f32_16x6.name.as_str() },
    KernelChoice { mr: 16, nr: 5, scale: 0.98, name: || avx_mmm_f32_16x5.name.as_str() },
    KernelChoice { mr: 24, nr: 4, scale: 0.95, name: || avx_mmm_f32_24x4.name.as_str() },
    KernelChoice { mr: 32, nr: 3, scale: 0.93, name: || avx_mmm_f32_32x3.name.as_str() },
    KernelChoice { mr: 40, nr: 2, scale: 0.90, name: || avx_mmm_f32_40x2.name.as_str() },
    KernelChoice { mr: 8, nr: 8, scale: 0.80, name: || avx_mmm_f32_8x8.name.as_str() },
];

/// f32 and i32 kernels for AVX-capable CPUs that can't run the fma tier (Sandy/Ivy Bridge
/// without fma; AMD Bulldozer-family with fma but no avx2). It sits at the bottom of the x86
/// ladder, so wherever the fma tier can answer it does, and this one is only reached when it
/// cannot.
fn avx_preferred(
    _isa: &IsaSet,
    dt: DatumType,
    query: &Query,
    _suitable: &[Suitable],
) -> Option<&'static str> {
    match (dt, query.n) {
        (DatumType::I32, Some(1)) => None,
        (DatumType::I32, _) => Some(avx_mmm_i32_8x4.name.as_str()),
        (DatumType::F32, _) => match query.n {
            None => Some(avx_mmm_f32_16x6.name.as_str()),
            Some(1) => Some(avx_mmm_f32_64x1.name.as_str()),
            Some(2) => Some(avx_mmm_f32_40x2.name.as_str()),
            Some(3) => Some(avx_mmm_f32_32x3.name.as_str()),
            Some(4) => Some(avx_mmm_f32_24x4.name.as_str()),
            Some(5) => Some(avx_mmm_f32_16x5.name.as_str()),
            Some(6) => Some(avx_mmm_f32_16x6.name.as_str()),
            Some(8) => Some(avx_mmm_f32_8x8.name.as_str()),
            Some(_) => Some(pick_mmm(AVX_CHOICES, query.m, query.n)),
        },
        _ => None,
    }
}

inventory::submit! {
    MmmTier {
        arch: Some(Arch::X86_64),
        precedence: 1,
        name: "avx",
        applies: |isa| isa.has(Isa::X86_64Avx),
        preferred: avx_preferred,
    }
}

/// Fallback for non-Intel/AMD x86: hand-tuned low-N choices, then a generic (M, N)-aware
/// tile-utilisation picker over the same kernels.
const FMA_CHOICES: &[KernelChoice] = &[
    KernelChoice { mr: 8, nr: 8, scale: 44.0 / 60.0, name: || fma_mmm_f32_8x8.name.as_str() },
    KernelChoice { mr: 16, nr: 6, scale: 54.0 / 60.0, name: || fma_mmm_f32_16x6.name.as_str() },
    KernelChoice { mr: 16, nr: 5, scale: 54.0 / 60.0, name: || fma_mmm_f32_16x5.name.as_str() },
    KernelChoice { mr: 24, nr: 4, scale: 54.0 / 60.0, name: || fma_mmm_f32_24x4.name.as_str() },
    KernelChoice { mr: 32, nr: 3, scale: 54.0 / 60.0, name: || fma_mmm_f32_32x3.name.as_str() },
    KernelChoice { mr: 40, nr: 2, scale: 54.0 / 60.0, name: || fma_mmm_f32_40x2.name.as_str() },
];

fn fma_mmm_f32(suitable: &[Suitable], query: &Query) -> Option<&'static str> {
    match super::vendor() {
        super::Vendor::Intel => {
            super::intel_fma_linear::linear_model().preferred(suitable, query.m, query.k, query.n)
        }
        super::Vendor::Amd => {
            super::amd_fma_linear::linear_model().preferred(suitable, query.m, query.k, query.n)
        }
        super::Vendor::Other => match query.n {
            None => Some(fma_mmm_f32_16x6.name.as_str()),
            Some(1) => unreachable!("n == 1 answered above"),
            Some(2) => Some(fma_mmm_f32_40x2.name.as_str()),
            Some(3) => Some(fma_mmm_f32_32x3.name.as_str()),
            Some(4) => Some(fma_mmm_f32_24x4.name.as_str()),
            Some(5) => Some(fma_mmm_f32_16x5.name.as_str()),
            Some(6) => Some(fma_mmm_f32_16x6.name.as_str()),
            Some(8) => Some(fma_mmm_f32_8x8.name.as_str()),
            Some(_) => Some(pick_mmm(FMA_CHOICES, query.m, query.n)),
        },
    }
}

fn fma_preferred(
    _isa: &IsaSet,
    dt: DatumType,
    query: &Query,
    suitable: &[Suitable],
) -> Option<&'static str> {
    match (dt, query.n) {
        // n == 1 has no dedicated n=1-calibrated model on the fma-only path yet; keep the
        // fixed matvec kernel. Routing n==1 through the n>=2-fit mmm model mispicks (matvec
        // kernels are only ever run at n==1, so their mmm coeffs are unrepresentative).
        (DatumType::F32, Some(1)) => Some(fma_mmm_f32_64x1.name.as_str()),
        (DatumType::F32, _) => fma_mmm_f32(suitable, query),
        _ => None,
    }
}

inventory::submit! {
    MmmTier {
        arch: Some(Arch::X86_64),
        precedence: 4,
        name: "fma",
        applies: |isa| isa.has(Isa::X86_64Avx2) && isa.has(Isa::X86_64Fma),
        preferred: fma_preferred,
    }
}

/// The choices span both instruction sets: on avx512 hardware the 256-bit FMA kernels reach
/// comparable f32 throughput and win the small-N tiles the avx512 kernels have no matching `nr`
/// for (e.g. n=2 -> 40x2, n=4 -> 24x4). `scale` is relative throughput at full tile fill,
/// measured together with `tract hwbench 3840,256,120,f32` (M,N divide every mr/nr) and
/// normalised to the fastest kernel.
const X86_F32_CHOICES: &[KernelChoice] = &[
    KernelChoice { mr: 16, nr: 12, scale: 1.000, name: || avx512_mmm_f32_16x12.name.as_str() },
    KernelChoice { mr: 16, nr: 8, scale: 0.995, name: || avx512_mmm_f32_16x8.name.as_str() },
    KernelChoice { mr: 32, nr: 5, scale: 0.992, name: || avx512_mmm_f32_32x5.name.as_str() },
    KernelChoice { mr: 32, nr: 6, scale: 0.990, name: || avx512_mmm_f32_32x6.name.as_str() },
    KernelChoice { mr: 48, nr: 4, scale: 0.978, name: || avx512_mmm_f32_48x4.name.as_str() },
    KernelChoice { mr: 16, nr: 6, scale: 0.964, name: || fma_mmm_f32_16x6.name.as_str() },
    KernelChoice { mr: 24, nr: 4, scale: 0.948, name: || fma_mmm_f32_24x4.name.as_str() },
    KernelChoice { mr: 16, nr: 5, scale: 0.935, name: || fma_mmm_f32_16x5.name.as_str() },
    KernelChoice { mr: 32, nr: 3, scale: 0.919, name: || fma_mmm_f32_32x3.name.as_str() },
    KernelChoice { mr: 64, nr: 3, scale: 0.895, name: || avx512_mmm_f32_64x3.name.as_str() },
    KernelChoice { mr: 40, nr: 2, scale: 0.842, name: || fma_mmm_f32_40x2.name.as_str() },
    KernelChoice { mr: 8, nr: 8, scale: 0.788, name: || fma_mmm_f32_8x8.name.as_str() },
    KernelChoice { mr: 80, nr: 2, scale: 0.766, name: || avx512_mmm_f32_80x2.name.as_str() },
    KernelChoice { mr: 128, nr: 1, scale: 0.378, name: || avx512_mmm_f32_128x1.name.as_str() },
];

/// n==1: below the widest matvec kernel's mr (128) the cost model picks a better-fitting kernel
/// (16x1/32x1); at or above it the 128x1 is already optimal, so keep it. AMD has no
/// n=1-calibrated mmv model yet, and keeps the fixed matvec dispatch.
fn avx512_mmv_f32(suitable: &[Suitable], query: &Query) -> Option<&'static str> {
    match super::vendor() {
        super::Vendor::Intel => match query.m {
            Some(m) if m < 128 => super::intel_avx512_mmv_linear::linear_model().preferred(
                suitable,
                Some(m),
                query.k,
                Some(1),
            ),
            _ => Some(avx512_mmm_f32_128x1.name.as_str()),
        },
        _ => match query.m {
            Some(m) if m < 31 => Some(avx512_mmm_f32_16x1.name.as_str()),
            _ => Some(avx512_mmm_f32_128x1.name.as_str()),
        },
    }
}

fn avx512_mmm_f32(suitable: &[Suitable], query: &Query) -> Option<&'static str> {
    match super::vendor() {
        super::Vendor::Intel => super::intel_avx512_linear::linear_model()
            .preferred(suitable, query.m, query.k, query.n),
        super::Vendor::Amd => {
            super::amd_avx512_linear::linear_model().preferred(suitable, query.m, query.k, query.n)
        }
        super::Vendor::Other => {
            if let Some(1) = query.n {
                unreachable!("n == 1 answered above");
            }
            Some(pick_mmm(X86_F32_CHOICES, query.m, query.n))
        }
    }
}

fn avx512f_preferred(
    _isa: &IsaSet,
    dt: DatumType,
    query: &Query,
    suitable: &[Suitable],
) -> Option<&'static str> {
    match (dt, query.n) {
        (DatumType::F32, Some(1)) => avx512_mmv_f32(suitable, query),
        (DatumType::F32, _) => avx512_mmm_f32(suitable, query),
        _ => None,
    }
}

inventory::submit! {
    MmmTier {
        arch: Some(Arch::X86_64),
        precedence: 5,
        name: "avx512f",
        applies: |isa| isa.has(Isa::X86_64Avx2) && isa.has(Isa::X86_64Fma) && isa.has(Isa::X86_64Avx512f),
        preferred: avx512f_preferred,
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;
    use crate::frame::mmm::{AsInputValue, FusedSpec};
    use tract_data::internal::*;

    #[test]
    fn avx512_128x1_add_unicast_with_strided_c() -> TractResult<()> {
        if !is_x86_feature_detected!("avx512f") {
            return Ok(());
        }
        let (m, k_each, n) = (1000usize, 256usize, 13usize);
        let a0: Vec<f32> = (0..m * k_each).map(|i| ((i % 17) as f32 - 8.0) / 16.0).collect();
        let a1: Vec<f32> = (0..m * k_each).map(|i| ((i % 19) as f32 - 9.0) / 18.0).collect();
        let b0: Vec<f32> = (0..k_each * n).map(|i| ((i % 13) as f32 - 6.0) / 13.0).collect();
        let b1: Vec<f32> = (0..k_each * n).map(|i| ((i % 11) as f32 - 5.0) / 10.0).collect();

        let mut expected = vec![0.0f32; m * n];
        for r in 0..m {
            for c in 0..n {
                let mut acc = 0.0f32;
                for kk in 0..k_each {
                    acc += a0[r * k_each + kk] * b0[kk * n + c];
                    acc += a1[r * k_each + kk] * b1[kk * n + c];
                }
                expected[r * n + c] = acc;
            }
        }

        let ker = avx512_mmm_f32_128x1.mmm();
        let (pack_a, pack_b) = &ker.packings()[0];
        let pack_one =
            |buf: Vec<f32>, rows, cols, m_axis, k_axis, pack: &dyn crate::mmm::MMMInputFormat| {
                let t =
                    tract_ndarray::Array2::from_shape_vec((rows, cols), buf).unwrap().into_tensor();
                pack.prepare_one(&t, k_axis, m_axis).unwrap()
            };
        let pa0 = pack_one(a0, m, k_each, 0, 1, &**pack_a);
        let pa1 = pack_one(a1, m, k_each, 0, 1, &**pack_a);
        let pb0 = pack_one(b0, k_each, n, 1, 0, &**pack_b);
        let pb1 = pack_one(b1, k_each, n, 1, 0, &**pack_b);

        // C-buffer layout with row stride > nr*sizeof, matching squeezenet conv10's
        // (M=1000, spatial=13, N=13) view: M-stride is 169 floats, not nr=1.
        let spatial = 13usize;
        let mut c_backing = Tensor::zero::<f32>(&[m, spatial, n])?;
        let c_spec = unsafe { ker.c_from_data_and_strides(4, (spatial * n) as isize, 1) };

        unsafe {
            let c_view = c_backing.view_mut();
            let c = c_spec.wrap(&c_view);
            let ops: TVec<FusedSpec> = tvec!(
                FusedSpec::AddMatMul {
                    a: AsInputValue::Borrowed(&*pa0),
                    b: AsInputValue::Borrowed(&*pb0),
                    packing: 0,
                },
                FusedSpec::Store(c),
            );
            ker.run(m, n, &ops)?;
        }

        unsafe {
            let c_view = c_backing.view_mut();
            let c_for_unicast = c_spec.wrap(&c_view);
            let c_for_store = c_spec.wrap(&c_view);
            let ops: TVec<FusedSpec> = tvec!(
                FusedSpec::AddMatMul {
                    a: AsInputValue::Borrowed(&*pa1),
                    b: AsInputValue::Borrowed(&*pb1),
                    packing: 0,
                },
                FusedSpec::AddUnicast(c_for_unicast),
                FusedSpec::Store(c_for_store),
            );
            ker.run(m, n, &ops)?;
        }

        let c_slice = c_backing.to_plain_array_view::<f32>()?;
        let mut max_err = 0.0f32;
        let mut wrong_cells = 0;
        for r in 0..m {
            for cc in 0..n {
                let got = c_slice[[r, 0, cc]];
                let exp = expected[r * n + cc];
                let e = (got - exp).abs();
                if e > 1e-3 {
                    wrong_cells += 1;
                }
                max_err = max_err.max(e);
            }
        }
        assert!(
            max_err < 1e-3,
            "avx512_mmm_f32_128x1 wrong output at squeezenet shape: \
             max_err={max_err}, {wrong_cells}/{} cells off",
            m * n,
        );
        Ok(())
    }
}
