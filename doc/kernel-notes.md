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

## Tuning knobs

A handful of `TRACT_*` env vars steer kernel selection and CPU detection
without recompiling — most usefully `TRACT_LAZY_IM2COL_MIN_KERNEL` /
`TRACT_LAZY_IM2COL_MAX_EAGER_BYTES` for the `Conv` codegen crossover, and
`TRACT_CPU_AARCH64_KIND` / `TRACT_CPU_ARM32_NEON` for forcing detection on
emulated or misreporting targets. See
[`cli-recipe.md` § Environment variables](cli-recipe.md#environment-variables)
for the full list.
