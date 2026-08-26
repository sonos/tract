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

The x86 picker (`plug_avx512f` / `plug_fma` in `linalg/src/x86_64/mmm.rs`)
scores kernels by `scale * m_util * n_util`, where `scale` is each kernel's
relative throughput once tile-fill is equal. When those `scale`s are all left at
`1.0`, sub-1% tile-padding differences alone decide between otherwise-equal
kernels, which can route large-N / small-K GEMMs onto a narrower, slower tile.
Populate `scale` from an `hwbench` run at a padding-neutral `N` (one divisible by
every `nr`, e.g. `120` — the default `512` unfairly favours power-of-two `nr`):
`tract hwbench 512,512,120`, then normalise the column to the fastest kernel.

## Inspecting dispatch

`tract selection` dumps what mmm selection answers for every machine, not just
this one: the declared kernels, each architecture's runnable set, the resolved
tier ladder, and per accumulator and shape the tier's answer, the answer
selection honours, the pick, and the set that survives `retain_best`. The shape
sweep is hard-coded so that two dumps always compare, which is the point — dump
before and after a dispatch change on the same host and diff it, rather than
arguing about what moved.

It is not a golden file. A few tiers read the machine rather than the instruction
set — the Apple chip generation, the Cortex model behind the a53/a55 boosts, the
AMX virtualisation heuristic, `TRACT_AMX_BF16` — so one revision dumps
differently on two hosts. Diff two builds on one host, and run it on the board
when the board is the question.

The sweep is synthetic, so it covers every cohort with no knob set — but for the
architecture this host actually is, a built kernel is kept only when the host's own
instruction set can run it (`runnable` reads the probed set, not the row's). The
native rows therefore describe machines no smaller than this host, and
`TRACT_CPU_ISA` (`+sve2`, `-avx512f`, …) is how to shrink it: it edits the probed
set and moves the picks as lesser hardware would. Foreign rows are unaffected,
their kernels being unbuilt metadata. Either way the effect is the same in two
dumps from one host, so it cancels in a diff.

Nothing conjures a kernel this toolchain never assembled: SVE and SME sit behind
`build.rs` assembler probes, and the `DECL` rows say `built=false` for a tree that
was only declared.

## Tuning knobs

A handful of `TRACT_*` env vars steer kernel selection and CPU detection
without recompiling — most usefully `TRACT_LAZY_IM2COL_MIN_KERNEL` /
`TRACT_LAZY_IM2COL_MAX_EAGER_BYTES` for the `Conv` codegen crossover, and
`TRACT_CPU_AARCH64_KIND` / `TRACT_CPU_ARM32_NEON` for forcing detection on
emulated or misreporting targets. Run `tract list-knobs` for the full,
always-current list, or see
[`cli-recipe.md` § Configuration knobs](cli-recipe.md#configuration-knobs)
for the annotated highlights.
