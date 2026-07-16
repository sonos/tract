# The matmul kernel cost model

For each `MatMatMul` in an optimized model, tract-linalg picks one microkernel from
the pool available on the running CPU (e.g. on cortex-a53: `12x8_a53`, `8x8_a53`,
`16x4_a53`, `24x4_a53`, … and their `_a55`/`_gen` siblings). The pick is made by an
analytic **`LinearCostModel`**, fit per CPU from on-device measurements.

## The model

`LinearCostModel` (`linalg/src/frame/mmm/cost_model.rs`) predicts each candidate
kernel's runtime at a shape `(m, k, n)` as:

```
time ≈ a · padded_work + b · n_tiles + c
padded_work = ceil(m/mr)·mr · ceil(n/nr)·nr · k     (MACs after tile padding)
n_tiles     = ceil(m/mr) · ceil(n/nr)
```

with three coefficients per kernel: `a` (inverse steady-state throughput), `b`
(per-tile setup), `c` (fixed call overhead). `pick` returns the argmin over the
impls actually present on the target. `mr`/`nr` are read from each impl, so the model
only needs the coefficient table.

The per-target tables are generated Rust files with the dataset each was fit from committed
next to them:
- **arm**: `linalg/src/arm{32,64}/cortex_<cpu>_linear.rs` (+ `cortex_<cpu>.txt`), dispatched by
  `Kind` (CPU part from `/proc/cpuinfo`) in `arm32.rs` / `arm64.rs`.
- **x86**: `linalg/src/x86_64_fma/{intel,amd}_{avx512,fma}_linear.rs` (+ `.txt`). The cohort is
  vendor × ISA tier: the tier is decided by which plug runs (`plug_fma` / `plug_avx512f` in
  `x86_64_fma/mmm.rs`), the vendor by CPUID (`TRACT_X86_KIND=intel|amd|other` overrides).
  Unknown vendors keep the hand-tuned `pick_mmm` fallback.

The coefficients are fit with **non-negative least squares** — times are sums of non-negative
costs, and plain LS can emit a negative coefficient on noisy data that mispicks badly.

## When to regenerate

The coefficients encode measured kernel timings. **Any change to a kernel's assembly
(a retune, added prefetch, a new packing) makes the model stale** — it will keep
avoiding a kernel that has since gotten faster, or vice-versa. Regenerate the affected
CPUs' models after such a change. It is cheap and fully scripted; there is no training
step and no Python.

## Regenerating a CPU's model

There are two ways in, with the same output — a `<platform>_linear.rs` + `.txt` fit from
on-device timings:

- **`cost-model regen`** — one command, for a target you can build *and* run tract on
  directly (x86 desktop/server, Apple, any aarch64/arm dev box). Preferred.
- **manual `gather` + `fit`** — for cross-compiled boards (cortex-a7/a9/a53/a55) that can't
  run the tract source tree: gather on the board, fit on the host.

Both time every f32 kernel with a **fitting-grade fast timing** — coarser than `hwbench
--assert` (~0.1 s warmup + ~0.25 s sampling vs ~1 s + ~1 s), ~3× quicker. The fit averages
over many shapes, so sub-second per-kernel timing is enough to rank kernels; the relative
order (which is what `pick` needs) is robust to the coarser budget. Both exclude `n = 1` —
that is the matrix-vector (`mmv`) path, dispatched separately and not cost-modeled.

### The one-liner: `cost-model regen`

Run from the tract source tree, on the target machine:

```
tract cost-model regen
```

It (1) **detects the platform** and resolves the target `_linear.rs`/`.txt` path + device
class; (2) **gathers** a class-appropriate dataset — the committed seed shapes for the class
(`linalg/cost-model-seeds/<class>.txt`) unioned with a log-uniform random sweep that samples
small `m/k/n` densely, where kernel choice matters most; (3) **fits** with NNLS; (4)
**validates against the currently-installed picker** — for each gathered shape it compares
the new model's pick, the live `ops().mmm(m,k,n)` pick, and the measured oracle, and prints
both regrets; (5) **writes** the result to a side file (provenance header: user, host, date,
CPU, tract version+commit, validation summary) and prints a ready-to-use `mv`.

**It never overwrites in place.** Read the printed `current → new` regret, then run the `mv`
to install. Flags: `--platform <id>` forces the target (else auto), `--size` the random-sweep
count, `--seed` the PRNG seed, `--out-dir` the side-file directory.

**Seed shapes are the "model findings"** — they anchor the fit on shapes that actually occur,
so it never has to extrapolate. A uniform random sweep alone can miss, e.g., the large-`m/k`,
tiny-`n` matmuls a transformer emits and mispick badly there. Extract a model's shapes with
`tract <model> <args> -O dump --mm` and append them under a heading in the class seed file;
that model is then covered on the next regen. The classes are `small32` (armv7),
`small64` (embedded aarch64), and `big64` (x86 / Apple / Neoverse); each seed is both read
from disk (editable in a checkout) and baked into the binary via `include_str!`, so a
standalone build regenerates with the seed even outside the source tree.

### Manual: gather on the board, fit on the host (small devices)

`regen` runs the whole loop in one process, so it needs a target you can build and run tract
on. Cross-compiled boards can't do that — split it:

1. **Build the CLI for the target.** A static-musl build runs on any Linux kernel regardless
   of the board's libc:

   ```
   cargo build --release -p tract-cli --target aarch64-unknown-linux-musl
   ```

2. **Gather on the board** — natively, or through a runner such as `cargo-dinghy`. Keep `n`
   at 2+ (n=1 is the mmv path); `-` streams the dataset to stdout so you can capture a remote
   run. The sweep is deterministic in `--seed`.

   ```
   tract cost-model gather --m 2-192 --k 2-192 --n 2-192 --mkn 4000000 --size 150 -
   ```

3. **Fit on the host and drop the file in:**

   ```
   tract cost-model fit cortex_a53.txt linalg/src/arm64/cortex_a53_linear.rs
   ```

   Commit both `cortex_<cpu>.txt` and the generated `_linear.rs`.

## Wiring a new target

Add `mod <name>_linear;` and install its `pick` in the `mmm_f32` dispatch, keeping a
fallback for unrecognized targets:
- **arm**: a new `Kind::CortexXX` arm in the `arm32.rs` / `arm64.rs` match.
- **Apple**: per-chip, keyed on the CPU brand string (`apple_chip()`), installed *after*
  `apple_amx::plug` / `sme::plug` in `arm64.rs` so it takes precedence over the AMX heuristic
  and the always-SME default; unrecognized Apple chips keep that fallback. Chips differ enough
  to need their own model (M1 has AMX, M4 has SME).
- **x86**: a new cohort is only needed if a vendor/tier wants materially different picks
  (e.g. an Intel-server-AVX-512 model distinct from the client one) — then key it on CPUID
  family/model in `x86_64_fma.rs::vendor` and select it in `plug_fma` / `plug_avx512f`.

```rust
// example (arm)
Kind::CortexXX => {
    let model = cortex_xx_linear::linear_model();
    Box::new(move |m, k, n| model.pick(&impls, m, k, n))
}
```

## Validating the picks

`tract hwbench --assert --tolerance <pct>` times every kernel at a battery of shapes on
the target and fails if the dispatcher's pick lags the fastest available kernel by more
than the tolerance. Run it after regenerating to confirm the new model picks well. See
`doc/kernel-notes.md` for the hwbench battery and tuning knobs.
