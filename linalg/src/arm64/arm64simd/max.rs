use std::arch::aarch64::{float32x4_t, vdupq_n_f32, vgetq_lane_f32};

reduce_impl_wrap!(
    f32,
    arm64simd_max_f32_16n,
    16,
    4,
    (),
    f32::MIN,
    #[inline(never)]
    fn run(buf: &[f32], _: ()) -> f32 {
        assert!(buf.len() % 16 == 0);
        assert!(buf.len() > 0);
        unsafe {
            let len = buf.len();
            let ptr = buf.as_ptr();
            let mut out: float32x4_t = vdupq_n_f32(f32::MIN);
            std::arch::asm!("
            and v1.16b, v0.16b, v0.16b
            and v2.16b, v0.16b, v0.16b
            and v3.16b, v0.16b, v0.16b
            2:
                ld1 {{v4.4s, v5.4s, v6.4s, v7.4s}}, [{ptr}], 64
                fmax v0.4s, v0.4s, v4.4s
                fmax v1.4s, v1.4s, v5.4s
                fmax v2.4s, v2.4s, v6.4s
                fmax v3.4s, v3.4s, v7.4s
                subs {len}, {len}, 16
                bne 2b
            fmax v0.4s, v0.4s, v1.4s
            fmax v2.4s, v2.4s, v3.4s
            fmax v0.4s, v0.4s, v2.4s
            fmaxv s0, v0.4s
            ",
            len = inout(reg) len => _,
            ptr = inout(reg) ptr => _,
            inout("v0") out, out("v1") _, out("v2") _, out("v3") _,
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,);
            vgetq_lane_f32(out, 0)
        }
    },
    #[inline(never)]
    fn reduce_two(a: f32, b: f32) -> f32 {
        a.max(b)
    }
);

#[cfg(test)]
mod test_arm64simd_max_f32_16n {
    use super::*;
    crate::max_frame_tests!(true, f32, arm64simd_max_f32_16n);
}
