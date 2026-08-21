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
#[cfg(not(target_arch = "aarch64"))]
unsafe fn cvt_f16_to_f32(_src: &[f16], _dst: &mut [f32]) {
    panic!("cvt_f16_to_f32: not built for this target arch")
}

#[cfg(target_arch = "aarch64")]
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
#[cfg(not(target_arch = "aarch64"))]
unsafe fn cvt_f32_to_f16(_src: &[f32], _dst: &mut [f16]) {
    panic!("cvt_f32_to_f16: not built for this target arch")
}

#[cfg(target_arch = "aarch64")]
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
    sigmoid_frame_tests!(cfg!(target_arch = "aarch64"), f16, arm64simd_sigmoid_f16_4n);
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
    silu_frame_tests!(cfg!(target_arch = "aarch64"), f16, arm64simd_silu_f16_4n);
}

/// Every f16 bit pattern mapped through the NEON f32 SiLU kernel and rounded
/// back, so the activation is one load per element.
///
/// The table is filled by the same kernel the f32-roundtrip path runs, over an
/// aligned whole number of `nr`-blocks — which is the case the element-wise
/// frame hands to that kernel directly — so the two agree bit for bit. Calling
/// the kernel raw rather than through the frame also keeps this off the frame's
/// thread-local scratch, which is already borrowed whenever a kernel runs.
/// 128 KiB, built on first use.
#[cfg(target_arch = "aarch64")]
fn silu_lut() -> &'static [u16; 1 << 16] {
    use crate::frame::element_wise::ElementWiseKer;
    use tract_data::prelude::Tensor;
    static LUT: std::sync::OnceLock<Box<[u16; 1 << 16]>> = std::sync::OnceLock::new();
    LUT.get_or_init(|| {
        let mut values = unsafe {
            Tensor::uninitialized_aligned::<f32>(&[1 << 16], 16)
                .expect("silu lookup table allocation")
        };
        let widened = unsafe { values.as_slice_mut_unchecked::<f32>() };
        widened
            .iter_mut()
            .enumerate()
            .for_each(|(bits, v)| *v = f16::from_bits(bits as u16).to_f32());
        super::arm64simd_silu_f32_4n_fused::run(widened, ());
        let mut lut = Box::new([0u16; 1 << 16]);
        lut.iter_mut()
            .zip(widened.iter())
            .for_each(|(slot, v)| *slot = f16::from_f32(*v).to_bits());
        lut
    })
}

ew_impl_wrap2!(aarch64;
    f16,
    arm64simd_silu_f16_lut_8n,
    8,
    4,
    (),
    #[inline(never)]
    fn run(buf: &mut [f16], _params: ()) {
        let lut = silu_lut();
        buf.iter_mut().for_each(|x| *x = f16::from_bits(lut[x.to_bits() as usize]));
    }
);

#[cfg(all(test, target_arch = "aarch64"))]
mod silu_f16_agreement {
    use super::*;
    use crate::frame::element_wise::ElementWiseKer;

    #[test]
    fn lut_matches_the_f32_roundtrip_on_every_f16() {
        let all: Vec<f16> = (0..=u16::MAX).map(f16::from_bits).collect();
        let mut roundtrip = all.clone();
        let mut lut = all;
        arm64simd_silu_f16_4n::ew().run(&mut roundtrip).unwrap();
        arm64simd_silu_f16_lut_8n::ew().run(&mut lut).unwrap();
        let mismatch = roundtrip
            .iter()
            .zip(&lut)
            .position(|(a, b)| a.to_bits() != b.to_bits())
            .map(|i| (f16::from_bits(i as u16), roundtrip[i], lut[i]));
        assert_eq!(mismatch, None);
    }
}

// f32-roundtrip f16 tanh for arm64 cores without FEAT_FP16.
ew_impl_f16_via_f32!(
    arm64simd_tanh_f16_4n,
    4,
    4,
    CHUNK,
    16,
    cvt_f16_to_f32,
    cvt_f32_to_f16,
    super::arm64simd_tanh_f32_4n
);

#[cfg(test)]
pub mod test_arm64simd_tanh_f16_4n {
    use super::*;
    tanh_frame_tests!(cfg!(target_arch = "aarch64"), f16, arm64simd_tanh_f16_4n);
}
