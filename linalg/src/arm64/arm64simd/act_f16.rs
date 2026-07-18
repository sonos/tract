//! ARMv8.0 f32-roundtrip f16 activations.
//!
//! FEAT_FP16 adds half-precision *arithmetic*; the f16<->f32 conversions
//! (FCVTL / FCVTN) are baseline ASIMD and always available. A core without
//! FEAT_FP16 can therefore still reach NEON-f32 throughput on f16 activations
//! by converting into an f32 scratch, running the existing f32 kernel, and
//! converting back — rather than dropping to the generic scalar path.

use tract_data::internal::f16;

const CHUNK: usize = 256;

#[repr(C, align(16))]
struct AlignedScratch([f32; CHUNK]);

/// Convert `src` (f16) into `dst` (f32) via FCVTL/FCVTL2, 8 lanes per iteration.
/// The 1..8 leftover lanes fall back to scalar. NEON is baseline on aarch64, so
/// no target-feature gate is needed.
#[inline]
unsafe fn cvt_f16_to_f32(src: &[f16], dst: &mut [f32]) {
    let n = src.len();
    debug_assert!(dst.len() >= n);
    let chunks = n / 8;
    if chunks > 0 {
        unsafe {
            let s = src.as_ptr();
            let d = dst.as_mut_ptr();
            let c = chunks;
            std::arch::asm!("
                2:
                    ld1    {{v0.8h}}, [{s}], #16
                    fcvtl  v1.4s, v0.4h
                    fcvtl2 v2.4s, v0.8h
                    st1    {{v1.4s, v2.4s}}, [{d}], #32
                    subs   {c}, {c}, #1
                    bne    2b
            ",
            s = inout(reg) s => _,
            d = inout(reg) d => _,
            c = inout(reg) c => _,
            out("v0") _, out("v1") _, out("v2") _,
            options(nostack),
            );
        }
    }
    for k in (chunks * 8)..n {
        unsafe { *dst.get_unchecked_mut(k) = src.get_unchecked(k).to_f32() };
    }
}

/// Convert `src` (f32) into `dst` (f16) via FCVTN/FCVTN2, 8 lanes per iteration.
/// FCVTN rounds to nearest-even under the default FPCR, matching `f16::from_f32`.
#[inline]
unsafe fn cvt_f32_to_f16(src: &[f32], dst: &mut [f16]) {
    let n = src.len();
    debug_assert!(dst.len() >= n);
    let chunks = n / 8;
    if chunks > 0 {
        unsafe {
            let s = src.as_ptr();
            let d = dst.as_mut_ptr();
            let c = chunks;
            std::arch::asm!("
                2:
                    ld1    {{v1.4s, v2.4s}}, [{s}], #32
                    fcvtn  v0.4h, v1.4s
                    fcvtn2 v0.8h, v2.4s
                    st1    {{v0.8h}}, [{d}], #16
                    subs   {c}, {c}, #1
                    bne    2b
            ",
            s = inout(reg) s => _,
            d = inout(reg) d => _,
            c = inout(reg) c => _,
            out("v0") _, out("v1") _, out("v2") _,
            options(nostack),
            );
        }
    }
    for k in (chunks * 8)..n {
        unsafe { *dst.get_unchecked_mut(k) = f16::from_f32(*src.get_unchecked(k)) };
    }
}

ew_impl_wrap!(
    f16,
    arm64simd_sigmoid_f16_8n,
    8,
    8,
    (),
    #[inline(never)]
    fn run(buf: &mut [f16], _: ()) {
        debug_assert!(buf.len() % Self::nr() == 0);
        if buf.is_empty() {
            return;
        }
        let mut scratch = AlignedScratch([0f32; CHUNK]);
        let s = &mut scratch.0;
        let mut i = 0;
        while i < buf.len() {
            let n = CHUNK.min(buf.len() - i);
            unsafe { cvt_f16_to_f32(&buf[i..i + n], &mut s[..n]) };
            super::arm64simd_sigmoid_f32_4n::run(&mut s[..n], ());
            unsafe { cvt_f32_to_f16(&s[..n], &mut buf[i..i + n]) };
            i += n;
        }
    }
);

#[cfg(test)]
pub mod test_arm64simd_sigmoid_f16_8n {
    use super::*;
    sigmoid_frame_tests!(true, f16, arm64simd_sigmoid_f16_8n);
}
