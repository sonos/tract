//! ARMv8.0 f32-roundtrip f16 activations.
//!
//! FEAT_FP16 adds half-precision *arithmetic*; the f16<->f32 conversions
//! (FCVTL / FCVTN) are baseline ASIMD and always available. A core without
//! FEAT_FP16 can therefore still reach NEON-f32 throughput on f16 activations
//! by converting into an f32 scratch, running the existing f32 kernel, and
//! converting back — rather than dropping to the generic scalar path.

use tract_data::internal::f16;

/// f32 scratch length for the f16 round-trip, in elements. Kept small so the
/// scratch stays cache-hot across the three passes over each chunk (convert in,
/// run the f32 kernel in place, convert out) instead of being sized to fill a
/// cache level. The conversions handle any length, so nothing else constrains it.
const CHUNK: usize = 256;

/// Convert `src` (f16) into `dst` (f32) via FCVTL/FCVTL2 for any length: a
/// 32-lane unrolled main loop, an 8-lane fallback loop, then a scalar-step
/// FCVT loop for the final <8 elements — all in asm, no Rust tail. NEON and
/// scalar FCVT are baseline on aarch64, so no target-feature gate is needed.
#[inline]
unsafe fn cvt_f16_to_f32(src: &[f16], dst: &mut [f32]) {
    let n = src.len();
    debug_assert!(dst.len() >= n);
    unsafe {
        let s = src.as_ptr();
        let d = dst.as_mut_ptr();
        let c32 = n / 32;
        let c8 = (n % 32) / 8;
        let c1 = n % 8;
        std::arch::asm!("
                cbz    {c32}, 3f
            2:
                ld1    {{v0.8h, v1.8h, v2.8h, v3.8h}}, [{s}], #64
                fcvtl  v4.4s,  v0.4h
                fcvtl2 v5.4s,  v0.8h
                fcvtl  v6.4s,  v1.4h
                fcvtl2 v7.4s,  v1.8h
                fcvtl  v16.4s, v2.4h
                fcvtl2 v17.4s, v2.8h
                fcvtl  v18.4s, v3.4h
                fcvtl2 v19.4s, v3.8h
                st1    {{v4.4s, v5.4s, v6.4s, v7.4s}}, [{d}], #64
                st1    {{v16.4s, v17.4s, v18.4s, v19.4s}}, [{d}], #64
                subs   {c32}, {c32}, #1
                bne    2b
            3:
                cbz    {c8}, 5f
            4:
                ld1    {{v0.8h}}, [{s}], #16
                fcvtl  v4.4s,  v0.4h
                fcvtl2 v5.4s,  v0.8h
                st1    {{v4.4s, v5.4s}}, [{d}], #32
                subs   {c8}, {c8}, #1
                bne    4b
            5:
                cbz    {c1}, 7f
            6:
                ldr    h0, [{s}], #2
                fcvt   s0, h0
                str    s0, [{d}], #4
                subs   {c1}, {c1}, #1
                bne    6b
            7:
        ",
        s = inout(reg) s => _,
        d = inout(reg) d => _,
        c32 = inout(reg) c32 => _,
        c8 = inout(reg) c8 => _,
        c1 = inout(reg) c1 => _,
        out("v0") _, out("v1") _, out("v2") _, out("v3") _,
        out("v4") _, out("v5") _, out("v6") _, out("v7") _,
        out("v16") _, out("v17") _, out("v18") _, out("v19") _,
        options(nostack),
        );
    }
}

/// Convert `src` (f32) into `dst` (f16) via FCVTN/FCVTN2 for any length: a
/// 32-lane unrolled main loop, an 8-lane fallback loop, then a scalar-step
/// FCVT loop for the final <8 elements — all in asm, no Rust tail. FCVTN and
/// scalar FCVT round to nearest-even under the default FPCR, matching
/// `f16::from_f32`.
#[inline]
unsafe fn cvt_f32_to_f16(src: &[f32], dst: &mut [f16]) {
    let n = src.len();
    debug_assert!(dst.len() >= n);
    unsafe {
        let s = src.as_ptr();
        let d = dst.as_mut_ptr();
        let c32 = n / 32;
        let c8 = (n % 32) / 8;
        let c1 = n % 8;
        std::arch::asm!("
                cbz    {c32}, 3f
            2:
                ld1    {{v4.4s, v5.4s, v6.4s, v7.4s}}, [{s}], #64
                ld1    {{v16.4s, v17.4s, v18.4s, v19.4s}}, [{s}], #64
                fcvtn  v0.4h, v4.4s
                fcvtn2 v0.8h, v5.4s
                fcvtn  v1.4h, v6.4s
                fcvtn2 v1.8h, v7.4s
                fcvtn  v2.4h, v16.4s
                fcvtn2 v2.8h, v17.4s
                fcvtn  v3.4h, v18.4s
                fcvtn2 v3.8h, v19.4s
                st1    {{v0.8h, v1.8h, v2.8h, v3.8h}}, [{d}], #64
                subs   {c32}, {c32}, #1
                bne    2b
            3:
                cbz    {c8}, 5f
            4:
                ld1    {{v4.4s, v5.4s}}, [{s}], #32
                fcvtn  v0.4h, v4.4s
                fcvtn2 v0.8h, v5.4s
                st1    {{v0.8h}}, [{d}], #16
                subs   {c8}, {c8}, #1
                bne    4b
            5:
                cbz    {c1}, 7f
            6:
                ldr    s0, [{s}], #4
                fcvt   h0, s0
                str    h0, [{d}], #2
                subs   {c1}, {c1}, #1
                bne    6b
            7:
        ",
        s = inout(reg) s => _,
        d = inout(reg) d => _,
        c32 = inout(reg) c32 => _,
        c8 = inout(reg) c8 => _,
        c1 = inout(reg) c1 => _,
        out("v0") _, out("v1") _, out("v2") _, out("v3") _,
        out("v4") _, out("v5") _, out("v6") _, out("v7") _,
        out("v16") _, out("v17") _, out("v18") _, out("v19") _,
        options(nostack),
        );
    }
}

ew_impl_f16_via_f32!(
    arm64simd_sigmoid_f16_4n,
    4,
    4,
    CHUNK,
    16,
    cvt_f16_to_f32,
    cvt_f32_to_f16,
    super::arm64simd_sigmoid_f32_4n
);

#[cfg(test)]
pub mod test_arm64simd_sigmoid_f16_4n {
    use super::*;
    sigmoid_frame_tests!(true, f16, arm64simd_sigmoid_f16_4n);
}

// f32-roundtrip f16 SiLU for arm64 cores without FEAT_FP16.
ew_impl_f16_via_f32!(
    arm64simd_silu_f16_4n,
    4,
    4,
    CHUNK,
    16,
    cvt_f16_to_f32,
    cvt_f32_to_f16,
    super::arm64simd_silu_f32_4n_fused
);

#[cfg(test)]
pub mod test_arm64simd_silu_f16_4n {
    use super::*;
    silu_frame_tests!(true, f16, arm64simd_silu_f16_4n);
}
