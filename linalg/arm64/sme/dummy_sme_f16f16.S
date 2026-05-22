// Build-time capability probe for FEAT_SME_F16F16, used by build.rs
// (assembler_supports_sme_f16f16). The experimental f16 SME kernels use the
// non-widening half-precision outer product `fmopa za.h`, which needs the
// +sme-f16f16 assembler extension. An assembler that supports base SME but not
// FEAT_SME_F16F16 (so the basic dummy_sme.S probe passes but this one fails)
// must still build the f32 SME kernels — so this is a SEPARATE probe gating
// only the f16 unit + the `tract_sme_f16f16` cfg. Not linked into anything.
.arch armv9-a+sme2+sme-f16f16
.text
.globl tract_sme_f16f16_probe
tract_sme_f16f16_probe:
    smstart
    ptrue   p0.h
    fmopa   za0.h, p0/m, p0/m, z0.h, z1.h
    smstop
    ret
