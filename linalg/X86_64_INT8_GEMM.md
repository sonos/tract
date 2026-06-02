# x86_64 int8 GEMM kernels

This note documents the int8 (i32-accumulator) matrix-multiply kernel family for
x86_64, for maintainers touching `linalg/src/x86_64_fma/mmm.rs` (Rust
registration + dispatch) and `linalg/x86_64/fma/*.S.j2` (assembly templates).

The kernels form a throughput cascade from the portable AVX2 emulation up to
Intel AMX, with AVX-512-VNNI in between. The right kernel is chosen at runtime
from CPUID + (for selection among ties) the einsum kernel scorer.

## Kernel family

| Kernel | ISA | Tile M×N | Matmul instr | A packing | B packing | Build gate |
|---|---|---|---|---|---|---|
| `avx2_mmm_i32_8x8` | AVX2 | 8×8 (ymm) | `vpmaddubsw` emulation | `PackedFormat` i8 | `PackedFormat` i8 | always |
| `avx512vnni_mmm_i32_8x8` | AVX-512-VNNI | 8×8 (ymm) | `vpdpbusd` | `PackedI8K4(8)` | `PackedI8K4(8)` | always |
| `avx512vnni_mmm_i32_16x16` | AVX-512-VNNI | 16×16 (zmm) | `vpdpbusd` ×16 rows | `PackedI8K4(16)` | `PackedI8K4(16)` | always |
| `avxvnni_mmm_i32_8x8` | AVX-VNNI (VEX) | 8×8 (ymm) | `{vex} vpdpbusd` | `PackedI8K4(8)` | `PackedI8K4(8)` | `tract_avxvnni` |
| `avx512amx_mmm_i32_8x8` | AMX-INT8 | 8×8 | `tdpbssd` | `PackedAmxA(8)` | `PackedI8K4(8)` | `tract_amx_int8` |
| `avx512amx_mmm_i32_16x16` | AMX-INT8 | 16×16 | `tdpbssd` (16384 MACs) | `PackedAmxA(16)` | `PackedI8K4(16)` | `tract_amx_int8` |
| `avx512amx_mmm_f32_16x16` (f32) | AMX-BF16 | 16×16 | `tdpbf16ps` | `PackedAmxBf16A(16)` | `PackedBf16K2(16)` | `tract_amx_bf16` |

The two AVX-512-VNNI kernels and the AVX2 one are always compiled (their
mnemonics are in every supported binutils); the AMX and AVX-VNNI kernels are
behind assembler-probe cfgs (see below).

## The u8×s8 `+128` bias trick (VNNI / AVX-VNNI)

`vpdpbusd` is **u8 × s8** (unsigned first operand). To compute the s8×s8 product
we need, the kernel offsets the A bytes by `+128` (modular `vpaddb`, making them
u8 in `[0,255]`) and then removes the resulting per-column bias
`128 * sum_k(B[n])` after the K loop. The bias is accumulated cheaply during the
loop with a `vpdpbusd` against an all-`0x01` u8 vector.

- **8×8 (ymm)** accumulators are *column-major* (`ymm{n}` = column n), so the
  bias is computed per column and splatted back with `vpermd`.
- **16×16 (zmm)** accumulators are *row-major* (`zmm{m}` = row m, 16 columns in
  the 16 lanes). The per-column bias is then a single lane-aligned vector, so the
  correction is one `vpsubd` per row — cleaner and cheaper than the 8×8 path.

AMX `tdpbssd` is **s8 × s8**, so the AMX int8 kernels need no `+128` trick; their
i32 accumulators are bit-identical to the AVX2 / VNNI reference.

## Packing formats (see `linalg/src/frame/pack.rs`)

- **`PackedI8K4(r)`** — K=4-inner. Per K=4 block, `r` elements × 4 K-bytes (= `4r`
  bytes); element `e` sits at byte offset `e*4` holding `[e, 4kb..4kb+3]`. K is
  zero-padded to a multiple of 4, so kernels read `ceil(k/4)` full blocks safely.
- **`PackedAmxA(r)`** — AMX A layout: per panel of `r` M-rows, row-major within
  the panel, K-bytes contiguous, K padded to a multiple of 64 (one `tdpbssd` step
  consumes 64 K-bytes).
- **`PackedAmxBf16A` / `PackedBf16K2`** — f32 inputs truncated to bf16 at pack
  time (round-to-nearest-even, matching `VCVTNEPS2BF16`) for the AMX-BF16 f32 path.

## Build-time cfg gating (`linalg/build.rs`)

Some mnemonics are too new for old toolchains, so each is guarded by an
**assembler probe** that tries to compile a tiny dummy `.S`. The probe sets a cfg
that gates *both* compiling the kernel template and referencing its Rust symbol:

| cfg | enables | requires |
|---|---|---|
| `tract_amx_int8` | AMX int8 kernels (`tdpbssd`) | gas ≥ 2.34 |
| `tract_amx_bf16` | AMX bf16 kernel (`tdpbf16ps`) | gas ≥ 2.34 |
| `tract_avxvnni` | AVX-VNNI ymm kernel (`{vex}` prefix) | binutils ≥ 2.36 |

Kernel `.S.j2` templates are sorted by filename prefix in `build.rs`:
`avx512amx_*_i32_*` and `*_f32_*` are pulled into their own gated compiles;
`avxvnni_*` likewise; everything else (including `avx512vnni_*`) stays in the
generic `-mfma` bulk compile. **A new `avx512vnni_*` kernel needs no `build.rs`
change** — but note that adding a brand-new template file may not trigger a
`build.rs` re-run on an incremental build (it emits per-file `rerun-if-changed`),
so `touch linalg/build.rs` after creating one.

These cfgs reflect **assembler** capability, not the host CPU. A kernel can be
*compiled* (assembler supports the mnemonic) yet never *run* (CPU lacks the
feature) — which matters for tests (below).

## Dispatch

`plug()` installs kernels in nested feature order, richest ISA last:

```
avx2 → [avxvnni] → fma → avx512f → avx512vnni → [amx_int8]      (int8 path)
                                  → [amx_bf16 overlay]           (f32 path)
```

Each `plug_*` pushes kernels into `ops.mmm_impls` and may set the explicit int8
picker `ops.qmmm_i32`. Because later plugs overwrite `qmmm_i32`, the best
available ISA wins. The pickers are **shape-adaptive**: the 16×16 tile is the
throughput champion when both M and N fill at least one tile; the 8×8 kernel has
lower per-call setup and wins on small problems. (AMX additionally requires
K ≥ 64; VNNI has no K gate since one `vpdpbusd` step is just 4 K-bytes.)

For paths that don't go through `qmmm_i32` (symbolic / unknown shapes via the
einsum kernel scorer), selection among equal-quality kernels uses
`-quality_cost*1000 + boost`. All `ManuallyOptimized` kernels tie on quality, so
`boost` breaks the tie:

| Kernel | boost |
|---|---|
| `avx512amx_mmm_i32_16x16`, `avx512amx_mmm_f32_16x16` | 100 |
| `avx512vnni_mmm_i32_16x16` | 50 |
| all 8×8 kernels | 0 |

So for unknown shapes: AMX 16×16 ≻ VNNI 16×16 ≻ {VNNI/AMX 8×8}. When AMX is
absent, VNNI 16×16 is the int8 champion.

## Testing and a cautionary tale

`MMMExternKernel!` auto-generates a `#[cfg(test)] mod test_<kernel>` with
packed-packed (per packing), fused-op frame, quant-rounding, store, and proptest
coverage. The harness **skips a kernel when `ker.is_supported_here()` is false**
(runtime CPUID). Consequently **AMX kernel tests only run on AMX hardware.**

The usual dev/CI host is Cascade Lake-class (AVX-512-VNNI, no AMX), so the AMX
tests are skipped there. That let a swapped-operand bug in the AMX 16×16 `sub`
fused-op handlers (`scalar_sub` / `per_row_sub` / `per_col_sub` and their
`_flipped` twins computed `acc - operand` instead of the correct `operand - acc`)
go unnoticed — until `avx512vnni_mmm_i32_16x16`, which **reuses the same zmm
row-major epilogue** and *does* run on VNNI hardware, exposed it (negated
results). Takeaway: a VNNI kernel that shares an AMX kernel's epilogue effectively
becomes the on-hardware test for that shared epilogue. The convention for the
non-commutative `sub` lives in `linalg/x86_64/fma/fma_mmm_ymm_ops.j2`
(`scalar` / `per_row` / `per_col` macros, `flipped` flag).

## Possible follow-ups

- A dispatch integration test asserting `qmmm_i32` selects the 16×16 kernel for
  large M,N and the 8×8 for small (no precedent for kernel-selection asserts
  in-tree yet; would need a small helper to read back the chosen `MatMatMul`).
- On Sapphire Rapids+ hardware: validate the AMX `sub` fix end-to-end, benchmark
  the AMX kernels, and re-check the 16×16/8×8 crossover and the `boost` values.
- A wider AVX-512-BF16 (`vdpbf16ps`) f32 kernel for Cooper Lake-class cores, and
  a Q4_0/Q8_0 → s8 packer feeding the AMX/VNNI 16×16 path directly.
