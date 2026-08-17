// Build-time capability probe for the assembler, used by build.rs
// (assembler_supports_zvfh). Zvfh reached binutils later than base RVV 1.0, so
// a toolchain can assemble the f32 kernels and still reject these. When the
// probe fails we skip the f16 kernels and the `tract_rvv_zvfh` cfg, and f16
// matmul falls back to the generic Rust kernels. Not linked into anything.
.option arch, +v, +zvfh, +zfhmin
.text
.globl tract_rvv_zvfh_probe
tract_rvv_zvfh_probe:
    vsetivli t0, 16, e16, m2, ta, ma
    flh ft0, 0(a0)
    fmv.h.x ft1, zero
    vle16.v v8, (a0)
    vfmacc.vf v16, ft0, v8
    vmfgt.vf v0, v16, ft1
    vsse16.v v16, (a0), a1
    ret
