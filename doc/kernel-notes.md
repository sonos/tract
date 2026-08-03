# Notes about implementing and working with the kernels

Kernels in tract-linalg are built using templated assembly and via `extern "C"` calling conventions.

The templates are stored in `linalg/$arch`, and in general the file
and main entrypoint share name stem. However, the proc name has a
suffix based on the package version. In order to skip maintaining this
the `extern_kernel!` macro declares the matching function and
re-exports it sans suffix.

Kernels work like a VM. When dispatching a kernel there's a list of
instructions from `FusedKerSpec` that's dispatched in a jump
table. For example; as of writing a MatMatmUl is roughly encoded as
`[Clear, AddMatMul, Store, Done]`. The dispatch is called `non_linear_loop`.

When iterating on assembly; building the code and looking at the
generated assembly under
`target/debug/build/tract-linalg-***/out/fma_mmm_*.S` can be much
easier than tracking the flow through each macro.

If one needs to debug a kernel a useful workflow is to simply insert a
`mov rNN, [0]` at the appropriate point, and configure GDB with
`handle SIGSEGV stop nopass`. This'll pause in GDB but not send the
signal to the program.

## Benchmarking a kernel

`tract hwbench <M,K,N[,dt]>...` times every matmul micro-kernel in the pool at
each shape and prints their flop/s sorted fastest-first, marking the one the
dispatcher currently picks with `<--`. This is the tool for both checking a
dispatch decision and calibrating one. Add `--json` to parse the results, or
`--assert` (with `--tolerance <pct>`, default 5) to fail when a pick lags the
fastest kernel — the basis for a CI that guards kernel selection.

On arm the pick comes from a per-CPU analytic `LinearCostModel` fit from on-device
timings; regenerating it after a kernel change is scripted via `tract cost-model
gather|fit` — see `doc/cost-model.md`.

The x86 picker (`plug_avx512f` / `plug_fma` in `linalg/src/x86_64_fma/mmm.rs`)
scores kernels by `scale * m_util * n_util`, where `scale` is each kernel's
relative throughput once tile-fill is equal. When those `scale`s are all left at
`1.0`, sub-1% tile-padding differences alone decide between otherwise-equal
kernels, which can route large-N / small-K GEMMs onto a narrower, slower tile.
Populate `scale` from an `hwbench` run at a padding-neutral `N` (one divisible by
every `nr`, e.g. `120` — the default `512` unfairly favours power-of-two `nr`):
`tract hwbench 512,512,120`, then normalise the column to the fastest kernel.

## Tuning knobs

A handful of `TRACT_*` env vars steer kernel selection and CPU detection
without recompiling — most usefully `TRACT_LAZY_IM2COL_MIN_KERNEL` /
`TRACT_LAZY_IM2COL_MAX_EAGER_BYTES` for the `Conv` codegen crossover, and
`TRACT_CPU_AARCH64_KIND` / `TRACT_CPU_ARM32_NEON` for forcing detection on
emulated or misreporting targets. Run `tract list-knobs` for the full,
always-current list, or see
[`cli-recipe.md` § Configuration knobs](cli-recipe.md#configuration-knobs)
for the annotated highlights.
