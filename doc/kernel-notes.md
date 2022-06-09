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
