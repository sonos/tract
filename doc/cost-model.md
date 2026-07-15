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

Everything goes through the main `tract` CLI (`cost-model` subcommand). Two steps:
`gather` runs **on the target CPU**, `fit` runs anywhere.

1. **Build the CLI for the target** (cross-compile as usual; a static-musl build runs on
   any Linux kernel regardless of the target's libc). Example for aarch64:

   ```
   cargo build --release -p tract-cli --target aarch64-unknown-linux-musl
   ```

2. **Gather a dataset on the target CPU.** Run it on real hardware — natively if you can
   log in, or through a runner such as `cargo-dinghy` for a remote board. `gather` times
   every f32 kernel over a sampled shape sweep and writes `kernel mr nr m k n dur` rows.
   The sweep is deterministic in `--seed`, so a dataset is reproducible.

   ```
   tract cost-model gather --m 1-256 --k 1-256 --n 1-256 --mkn 4000000 --size 150 out.txt
   ```

   `--size` is the number of shapes (each times the whole kernel pool). `--mkn` caps
   problem size so the sweep stays quick; `--m/--k/--n` take a range `lo-hi` or a list
   `a,b,c`. Pass `-` as the output to stream the dataset to stdout (handy when the run
   is remote and you capture its stdout). ~150 shapes is comfortably past the point where
   pick quality plateaus; a run takes a few minutes to ~half an hour depending on the CPU.

   `gather` reuses the exact timing path of `tract hwbench`, so a fitted model and the
   `hwbench --assert` pick-gate agree.

3. **Fit and drop the file in:**

   ```
   tract cost-model fit out.txt linalg/src/arm64/cortex_a53_linear.rs
   ```

   Commit both `out.txt` (as `cortex_<cpu>.txt`) and the generated `_linear.rs`.

## Wiring a new target

Add `mod <name>_linear;` and install its `pick` in the `mmm_f32` dispatch, keeping a
fallback for unrecognized targets:
- **arm**: a new `Kind::CortexXX` arm in the `arm32.rs` / `arm64.rs` match.
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
