# AMX validation & benchmark runbook

**For: a Claude Code session (or human) on an x86_64 CPU that has Intel AMX.**

The kernel work on branch `claude/zealous-galileo-fEQ3d` was developed on a
Cascade Lake-class container (AVX-512-VNNI, **no AMX**). Everything that can run
without AMX is already validated there. This runbook covers the two things that
box **could not** do and that need a real AMX CPU.

## Your task

**Benchmark every int8 / bf16 GEMM kernel in this tree on this AMX CPU — all the
AMX kernels *and* the AVX-512-VNNI kernels we just improved — and run the AMX
correctness suite.** Full kernel inventory to cover:

| Kernel | ISA | Covered by bench |
|---|---|---|
| `avx512amx_mmm_i32_8x8` | AMX int8 (`tdpbssd`) | `amx_i32` |
| `avx512amx_mmm_i32_16x16` | AMX int8 (`tdpbssd`) | `amx_i32` |
| `avx512amx_mmm_f32_16x16` | AMX bf16→f32 (`tdpbf16ps`) | `amx_f32` |
| `avx512vnni_mmm_i32_8x8` | AVX-512-VNNI (`vpdpbusd`) | `vnni_i32`, `amx_i32` |
| **`avx512vnni_mmm_i32_16x16`** ← new | AVX-512-VNNI (`vpdpbusd`, zmm) | `vnni_i32` |
| `avx2_mmm_i32_8x8` (baseline) | AVX2 | both i32 benches |

Running the three benches in Step 4 covers all of the above. Yes — bench the VNNI
kernels here too: an AMX CPU (Sapphire Rapids+) also has AVX-512-VNNI, so it's the
one place you can measure AMX 16×16 and VNNI 16×16 **on the same silicon** and see
how much AMX actually wins.

In addition, this AMX CPU is the only place that can:

1. **Correctness-test the AMX kernels** — including a recent bugfix to the AMX
   16×16 `sub` fused-op handlers that was invisible on non-AMX hardware.
2. **Benchmark** the AMX int8 / bf16 kernels and the new AVX-512-VNNI 16×16
   kernel head-to-head.

> ⚠️ **Most important caveat:** every AMX kernel test short-circuits to "ok" when
> the host can't run AMX (`is_supported_here()` is false). So a green
> `cargo test` on the wrong box proves **nothing**. You must first confirm AMX is
> actually live (Step 2). The **benchmarks are the authoritative gate-check** —
> they print an explicit "AMX … not available, skipping" message and emit no AMX
> columns if the gate is closed.

---

## 0. Prerequisites

| Requirement | Why | Check |
|---|---|---|
| AMX-capable CPU (Sapphire Rapids / Emerald Rapids / Granite Rapids Xeon, or Xeon Max) | `tdpbssd` / `tdpbf16ps` | `grep -o 'amx[_a-z]*' /proc/cpuinfo \| sort -u` → expect `amx_bf16 amx_int8 amx_tile` |
| Linux kernel ≥ 5.16 | AMX tile-data XSAVE permission via `arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)` | `uname -r` |
| binutils/gas ≥ 2.34 (≥ 2.36 ideal) | assembles AMX mnemonics (and `{vex}` for AVX-VNNI) | `as --version` |
| Rust stable (dev used 1.94–1.96) | build | `cargo --version` |

If `/proc/cpuinfo` shows no `amx_*` flags, this is the wrong machine — stop here.

---

## 1. Get the code

**Fresh clone (preferred):**
```sh
git clone https://github.com/czoli1976/tract.git
cd tract
git checkout claude/zealous-galileo-fEQ3d
```

**Existing checkout:**
```sh
git fetch origin claude/zealous-galileo-fEQ3d
git checkout claude/zealous-galileo-fEQ3d && git pull
# IMPORTANT when pulling into a checkout that was built before: the new kernel
# template (avx512vnni_mmm_i32_16x16.S.j2) may not trigger a build-script rerun
# (build.rs emits per-file rerun-if-changed). Force it once:
touch linalg/build.rs
```
(A fresh clone needs no `touch` — it renders every template on first build.)

---

## 2. Confirm AMX is actually live (do this first)

The AMX kernels are gated by CPUID **and** the kernel granting tile-data XSAVE
permission. The benchmark is the cleanest runtime probe — if AMX is unavailable
it prints a skip line instead of numbers:

```sh
cargo bench -p tract-linalg --bench amx_i32 -- --warm-up-time 0.2 --measurement-time 0.5 --sample-size 10 2>&1 | head -20
```

- ✅ **Good:** you see `avx512amx_8x8` and `avx512amx_16x16` lines with `thrpt:`.
- ❌ **Bad:** `AMX int8 not available (CPUID + arch_prctl gate failed), skipping`
  → AMX isn't usable (check kernel ≥ 5.16, not in a VM that masks AMX, XSAVE
  permission not blocked by a seccomp/container policy). Don't proceed — the
  correctness tests would silently no-op.

Optional: `RUST_LOG=info cargo test -p tract-linalg --lib avx512amx_mmm_i32_16x16 -- --nocapture 2>&1 | grep -i activated`
should log `qmmm_i32: x86_64/avx512amx_int8 (16x16 + 8x8 adaptive) activated`.

---

## 3. Correctness validation (the priority)

Only meaningful once Step 2 confirms AMX is live.

```sh
# All three AMX kernel suites: int8 8x8, int8 16x16, bf16 16x16.
cargo test -p tract-linalg --lib avx512amx 2>&1 | tail -30

# Full x86_64 mmm suite (AMX + VNNI + AVX2 + FMA + AVX-512), for completeness.
cargo test -p tract-linalg --lib x86_64_fma::mmm 2>&1 | tail -5
```

**Expected:** `test result: ok. <N> passed; 0 failed`.

**What this specifically proves (and the dev box couldn't):** the
`scalar_sub` / `per_row_sub` / `per_col_sub` (+ `_flipped`) fused-op tests for
`test_avx512amx_mmm_i32_16x16` and `test_avx512amx_mmm_f32_16x16` **actually
execute**. Those guard commit `99eb75b9d`, which fixed swapped operands in the
AMX `sub` handlers (they were computing `acc − operand` instead of
`operand − acc`, i.e. negated results). This fix is currently only
build-verified — **this run is what confirms it on real silicon.**

---

## 4. Benchmarks

On real hardware use default sampling (drop the reduced flags) and pin a core for
stable numbers. Idle box, turbo/frequency-scaling fixed if you can.

```sh
# int8: AVX2 vs VNNI 8x8 vs AMX 8x8 vs AMX 16x16
taskset -c 2 cargo bench -p tract-linalg --bench amx_i32

# f32 via bf16: FMA 16x6 vs AVX-512 16x12 vs AMX-BF16 16x16
taskset -c 2 cargo bench -p tract-linalg --bench amx_f32

# the new kernel in isolation: AVX2 vs VNNI 8x8 vs VNNI 16x16
taskset -c 2 cargo bench -p tract-linalg --bench vnni_i32
```

Bench layout (group `… /packed_packed`, shapes `64x256x64`, `256x256x256`,
`512x512x512`, `1024x1024x64`, throughput in `Gelem/s`):

| Bench | Columns |
|---|---|
| `amx_i32` | `avx2`, `avx512vnni`, `avx512amx_8x8`, `avx512amx_16x16` |
| `amx_f32` | `fma_16x6`, `avx512_16x12`, `avx512amx_bf16_16x16` |
| `vnni_i32` | `avx2`, `avx512vnni` (8×8), `avx512vnni_16x16` |

Criterion writes HTML reports under `target/criterion/`.

---

## 5. What to report back

**Correctness**
- Confirm AMX was live (Step 2 showed AMX columns / cpuinfo has `amx_int8`).
- `cargo test … avx512amx` result line (`N passed; 0 failed`), confirming the
  AMX `*_sub` fused-op tests passed → bugfix `99eb75b9d` validated on hardware.

**Performance** — the `thrpt:` (Gelem/s) per shape per column for all three
benches, plus these head-to-head reads:

1. **AMX 16×16 vs VNNI 16×16** (compare `amx_i32`'s `avx512amx_16x16` against
   `vnni_i32`'s `avx512vnni_16x16`, same shapes). AMX should win — that justifies
   the dispatch ordering (`boost(100)` for AMX 16×16 > `boost(50)` for VNNI
   16×16). Report the ratio.
2. **AMX 16×16 vs AMX 8×8** — the 4×-work-per-instruction claim and where 8×8
   wins on small shapes (informs the `qmmm_i32` 16/8 crossover).
3. **VNNI 16×16 vs 8×8** — does the ~1.3–2.1× measured on Cascade Lake hold on
   this CPU too?
4. **AMX-BF16 16×16 vs AVX-512 f32 16×12** — bf16 throughput win (with the bf16
   precision trade-off noted in `linalg/X86_64_INT8_GEMM.md`).

---

## Appendix A — one-shot script

```sh
set -e
echo "## CPU AMX flags:"; grep -o 'amx[_a-z]*' /proc/cpuinfo | sort -u || true
echo "## kernel:"; uname -r
echo "## gate check (expect AMX columns, not a skip message):"
cargo bench -p tract-linalg --bench amx_i32 -- --warm-up-time 0.2 --measurement-time 0.5 --sample-size 10 2>&1 | grep -iE "amx|skipping|thrpt" | head
echo "## correctness:"
cargo test -p tract-linalg --lib avx512amx 2>&1 | tail -3
cargo test -p tract-linalg --lib x86_64_fma::mmm 2>&1 | tail -3
echo "## full benches:"
taskset -c 2 cargo bench -p tract-linalg --bench amx_i32
taskset -c 2 cargo bench -p tract-linalg --bench amx_f32
taskset -c 2 cargo bench -p tract-linalg --bench vnni_i32
```

## Appendix B — what's on this branch

Three commits on top of the prior AMX/VNNI work:

| Commit | Summary |
|---|---|
| `9e8f1c5aa` | doc: `linalg/X86_64_INT8_GEMM.md` — the full int8 GEMM kernel cascade |
| `26726db8e` | **feat**: `avx512vnni_mmm_i32_16x16` — zmm-wide int8 VNNI kernel (1.3–2.1× over 8×8 on Cascade Lake) |
| `99eb75b9d` | **fix**: swapped operands in AMX 16×16 `sub` fused-op handlers (int8 + bf16) — **needs AMX to validate** |

Background and the kernel-selection/dispatch model: see
`linalg/X86_64_INT8_GEMM.md`.

> Note on Intel SDE: SDE *can* emulate AMX for **functional/correctness** checks
> on a non-AMX box (`sde64 -spr -- <test-binary>`), but it is **not** a
> performance model — timings under SDE are meaningless. Use it only if no AMX
> hardware is available, and never for the benchmark numbers above.
