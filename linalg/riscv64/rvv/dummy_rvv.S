// Build-time capability probe for the assembler, used by build.rs
// (assembler_supports_rvv). `.option arch` landed in binutils 2.36 and the
// ratified RVV 1.0 encodings in 2.38; toolchains older than that either reject
// the directive or silently know only the incompatible 0.7.1 draft. If this
// file fails to assemble, build.rs skips the RVV kernels and the `tract_rvv`
// cfg, and the runtime falls back to the generic Rust kernels. Not linked into
// anything.
//
// The instruction mix is deliberately the one the kernels actually need: the
// vector-scalar FMA that carries the inner loop, and the strided load/store
// pair the tile store and add_unicast are built on.
.option arch, +v
.text
.globl tract_rvv_probe
tract_rvv_probe:
    vsetivli t0, 8, e32, m2, ta, ma
    vle32.v v8, (a0)
    vfmacc.vf v16, ft0, v8
    vfrsub.vf v16, v16, ft0
    vmfgt.vf v0, v16, ft0
    vmerge.vvm v16, v18, v16, v0
    vlse32.v v4, (a0), a1
    vsse32.v v16, (a0), a1
    ret
