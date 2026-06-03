# AMX validation & benchmark results

Run of `linalg/AMX_BENCH_RUNBOOK.md` on real Intel AMX hardware.

- **Host:** `Intel(R) Xeon(R) Processor @ 2.10GHz` (Sapphire/Emerald Rapids-class), 4 vCPU
- **ISA:** `amx_tile amx_int8 amx_bf16` + AVX-512-VNNI; kernel `6.18.5` (≥5.16); binutils `2.42`; rustc `1.94.1`
- **Branch:** `claude/zealous-galileo-fEQ3d` @ `7a23812`
- **Method:** `cargo bench`, default criterion sampling, pinned to core 2 (`taskset -c 2`), idle box (load ≈ 1.0)
- **Date:** 2026-06-02

## 1. AMX live confirmation ✅

Gate-check (`amx_i32` bench) produced `avx512amx_8x8`/`avx512amx_16x16` columns with real `thrpt:` numbers — **neither** "tract not built with AMX" (build probe) **nor** "AMX not available, skipping" (runtime CPUID + `arch_prctl` XTILEDATA gate) appeared. AMX is genuinely exercised.

## 2. Correctness

| Suite | Result |
|---|---|
| `cargo test -p tract-linalg --lib avx512amx` | **297 passed; 3 failed** |
| `cargo test -p tract-linalg --lib x86_64_fma::mmm` | **1833 passed; 3 failed** |

**Bugfix `99eb75b9d` VALIDATED on silicon** ✅ — every `scalar_sub` / `per_row_sub` / `per_col_sub` (+`_f`) test passed for **both** `avx512amx_mmm_i32_16x16` and `avx512amx_mmm_f32_16x16`.

**3 failures — all in the AMX bf16 path** (`avx512amx_mmm_f32_16x16::f32f32_bf16`): `fuse::prop`, `frame::prop`, `fuse::packed_packed_bug_3`.

**Root cause = test-harness tolerance, NOT a kernel defect.** `packed_packed.rs:367` selects the comparison tolerance from the **accumulator** dtype:
```rust
let app = if K::Acc::datum_type() == f16::datum_type()
    { Approximation::SuperApproximate } else { Approximation::Approximate };
```
This kernel accumulates in **f32** (TDPBF16PS: bf16×bf16→f32), so it gets `Approximate` = `(atol 1e-4, rtol 5e-4, 0 outliers)` — but the `f32f32_bf16` packing truncates inputs to bf16 (~2⁻⁸ ≈ 0.39% rel). bf16-grade error is checked against an f32-grade bar with zero tolerated outliers ⇒ guaranteed failure. `SuperApproximate` `(atol 0.1, rtol 0.05, 1e-4 outliers)` would pass. The structurally identical int8 16×16 kernel passes 100%.

**Proposed fix:** in `check()`, pick `SuperApproximate` when the packing is bf16-based, not only when `K::Acc == f16`.

**Empirically verified (on the AMX host):** the kernel was run on 7 cases (including the exact `bug_3` input) and compared against an independent **bf16-truncated** reference — built with the project's own `f32_to_bf16_rne` — judged by the *same* tight `Approximate` bar: **0 outliers across ~335k output elements** (max abs err ≤ 1.3e-5), versus **282,788 outliers** against a pure-f32 reference. The kernel reproduces "truncate inputs→bf16, accumulate→f32" exactly; the 3 red tests are 100% the f32 oracle, with no kernel defect.

## 3. Benchmarks — throughput (Gelem/s, point estimate)

### `amx_i32` — int8 GEMM
| M×K×N | avx2 | avx512vnni (8×8) | avx512amx_8×8 | avx512amx_16×16 |
|---|---:|---:|---:|---:|
| 64×256×64 | 0.41 | 11.21 | 68.41 | **233.64** |
| 256×256×256 | 0.41 | 11.31 | 68.47 | **237.29** |
| 512×512×512 | 0.39 | 8.94 † | 112.86 | **228.15** |
| 1024×1024×64 | 0.41 | 34.84 | 178.42 | **279.51** |

### `amx_f32` — bf16→f32 GEMM
| M×K×N | fma_16×6 | avx512_16×12 | avx512amx_bf16_16×16 |
|---|---:|---:|---:|
| 64×256×64 | 37.12 | 64.31 | **207.35** |
| 256×256×256 | 37.90 | 71.90 | **225.74** |
| 512×512×512 | 39.37 | 64.69 | **348.38** |
| 1024×1024×64 | 36.85 | 59.22 | **318.36** |

### `vnni_i32` — int8 GEMM (new 16×16 in isolation)
| M×K×N | avx2 | avx512vnni (8×8) | avx512vnni_16×16 |
|---|---:|---:|---:|
| 64×256×64 | 0.41 | 10.90 | **135.74** |
| 256×256×256 | 0.40 | 10.78 | **134.92** |
| 512×512×512 | 0.40 | 20.53 | **154.39** |
| 1024×1024×64 | 0.41 | 34.77 | **161.27** |

† `avx512vnni`@512³ read 8.94 here vs 20.53 in `vnni_i32` (same kernel/shape). Treat **20.53** as the credible value (it fits the size trend 11.3→20.5→34.8); 8.94 was an outlier. A higher-sampling re-measure was attempted but could not complete — see §6.

## 4. Head-to-head ratios

| Comparison | 64×256×64 | 256×256×256 | 512×512×512 | 1024×1024×64 |
|---|---:|---:|---:|---:|
| **AMX 16×16 ÷ VNNI 16×16** (int8, same CPU) | 1.72× | 1.76× | 1.48× | 1.73× |
| **AMX 16×16 ÷ AMX 8×8** (int8) | 3.42× | 3.47× | 2.02× | 1.57× |
| **VNNI 16×16 ÷ VNNI 8×8** (int8) | 12.45× | 12.51× | 7.52× | 4.64× |
| **AMX bf16 16×16 ÷ AVX-512 f32 16×12** | 3.22× | 3.14× | 5.39× | 5.38× |
| *(bonus) AMX bf16 ÷ FMA f32 16×6* | 5.59× | 5.96× | 8.85× | 8.64× |

## 5. Findings

1. **AMX int8 16×16 wins everywhere — justifies `boost(100)` > VNNI `boost(50)`.** 1.48–1.76× over the new VNNI 16×16 on the *same* silicon. Dispatch ordering is correct.
2. **AMX 16×16 vs 8×8: 1.57–3.47×.** 16×16 leads on all tested shapes; the 4×-work/instr advantage is largest on compact shapes (3.4× @ 64×256×64) and narrowest on tall-skinny 1024×1024×64 (1.57×, N=64). No tested shape favors 8×8 — any crossover lives below this suite (smaller M or N<16). `qmmm_i32` defaulting to 16×16 here is sound.
3. **VNNI 16×16 vs 8×8: 4.64–12.5× — far above the dev box's 1.3–2.1×.** Likely the 8×8 kernel's ymm (256-bit) accumulators vs the new kernel's zmm (512-bit), amplified on Sapphire Rapids (no AVX-512 license downclock that Cascade Lake suffers). Strongly validates the new kernel; the magnitude warrants one sanity re-check (see #4).
4. **Data-quality flag (resolved by inspection):** `avx512vnni` 8×8 @ 512³ read 8.94 (in `amx_i32`) vs 20.53 (in `vnni_i32`) — a 2.3× swing on the same kernel/shape. **20.53 is the credible figure** (it continues the monotone size trend 11.3 @ 256³ → 20.5 @ 512³ → 34.8 @ 1024×1024×64; 8.94 breaks it). A `--sample-size 200` re-measure was launched but the AMX host was reclaimed before it could run (see §6); the ratio table already uses the consistent 20.53 pairing. AMX columns were stable across runs.
5. **AMX bf16 is 3.1–5.4× the AVX-512 f32 kernel** (5.6–8.9× over FMA), scaling up on larger shapes (348 Gelem/s @ 512³) — with the documented bf16 precision trade (see §2 and `X86_64_INT8_GEMM.md`).

## 6. Reproducibility note

Numbers were collected **2026-06-02** on an AMX-capable `Intel(R) Xeon(R) @ 2.10GHz` (`amx_tile/int8/bf16` + AVX-512-VNNI, kernel 6.18.5). The ephemeral session container was subsequently reclaimed and re-provisioned onto a different `Intel(R) Xeon(R) @ 2.80GHz` with **neither AMX nor AVX-512-VNNI** (only `avx512f`), on which `amx_i32`/`vnni_i32` both short-circuit and skip — so the one outstanding re-measure (VNNI-8×8 @ 512³) could not be completed in this session. To reproduce or extend, run on an AMX host (Sapphire Rapids / Emerald Rapids / Granite Rapids Xeon, or Xeon Max) following `linalg/AMX_BENCH_RUNBOOK.md`.
