use tract_data::half::f16;

reduce_impl_wrap!(
    f16,
    arm64fp16_max_f16_32n,
    32,
    8,
    (),
    f16::MIN,
    #[inline(never)]
    fn run(buf: &[f16], _: ()) -> f16 {
        assert!(buf.len() % 32 == 0);
        assert!(buf.len() > 0);
        #[target_feature(enable = "fp16")]
        unsafe fn run(buf: &[f16]) -> f16 {
            let len = buf.len();
            let ptr = buf.as_ptr();
            let mut out: u16;
            std::arch::asm!("
                ins v0.h[0], {min:w}
                dup v0.8h, v0.h[0]
                dup v1.8h, v0.h[0]
                dup v2.8h, v0.h[0]
                dup v3.8h, v0.h[0]

                2:
                    ld1 {{v4.8h, v5.8h, v6.8h, v7.8h}}, [{ptr}], 64
                    fmax v0.8h, v0.8h, v4.8h
                    fmax v1.8h, v1.8h, v5.8h
                    fmax v2.8h, v2.8h, v6.8h
                    fmax v3.8h, v3.8h, v7.8h

                    subs {len}, {len}, 32
                    bne 2b

                fmax v0.8h, v0.8h, v1.8h
                fmax v2.8h, v2.8h, v3.8h
                fmax v0.8h, v0.8h, v2.8h
                fmaxv h0, v0.8h
                ",
                // using v0 as inout triggers https://github.com/rust-lang/rust/issues/120374
                min = in(reg) f16::MIN.to_bits(),
                ptr = inout(reg) ptr => _,
                len = inout(reg) len => _,
                out("v0") out, out("v1") _, out("v2") _, out("v3") _,
                out("v4") _, out("v5") _, out("v6") _, out("v7") _,);
            f16::from_bits(out)
        }
        unsafe { run(buf) }
    },
    #[inline(never)]
    fn reduce_two(a: f16, b: f16) -> f16 {
        a.max(b)
    }
);

#[cfg(test)]
mod test_arm64fp16_max_f16_32n {
    use super::*;
    crate::max_frame_tests!(crate::arm64::has_fp16(), f16, arm64fp16_max_f16_32n);
}
