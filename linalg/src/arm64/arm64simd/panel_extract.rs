use crate::pack::Packing;
#[allow(unused_imports)]
use tract_data::internal::*;

panel_extractor!(aarch64; kernel_packed_32_q40_to_f32 as packed_32_q40_to_f32(
    Box::new(super::q40p32z16se()),
    f32::packing(32).align(16)
));

#[cfg(target_arch = "aarch64")]
unsafe fn kernel_packed_32_q40_to_f32(input: *const u8, output: *mut u8, k: usize) {
    unsafe {
        if k == 0 {
            return;
        }
        let lookup_table: [u8; 16] = [
            0xc8, 0xc7, 0xc6, 0xc5, 0xc4, 0xc2, 0xc0, 0xbc, 0x00, 0x3c, 0x40, 0x42, 0x44, 0x45,
            0x46, 0x47,
        ];
        std::arch::asm!("
    ld1      {{v13.16b}}, [{lookup_table}]
    movi     v15.16b, 15
    eor      v12.16b, v12.16b, v12.16b

    2:
        add     {scales}, {i}, 512  // scales at end: 32 (cols) * 32 (rows) / 2 (half byte)
        ld1     {{v0.8h-v3.8h}}, [{scales}]

        fcvtl   v16.4s, v0.4h
        fcvtl2  v17.4s, v0.8h
        fcvtl   v18.4s, v1.4h
        fcvtl2  v19.4s, v1.8h
        fcvtl   v20.4s, v2.4h
        fcvtl2  v21.4s, v2.8h
        fcvtl   v22.4s, v3.4h
        fcvtl2  v23.4s, v3.8h

        mov     {k2}, 32
    3:
        ld1     {{ v9.16b }}, [{i}], #16

        and     v0.16b, v9.16b, v15.16b
        ushr    v4.16b, v9.16b, 4

        tbl     v0.16b, {{ v13.16b }}, v0.16b
        tbl     v4.16b, {{ v13.16b }}, v4.16b

        zip2    v2.16b, v12.16b, v0.16b
        zip2    v6.16b, v12.16b, v4.16b

        zip1    v0.16b, v12.16b, v0.16b
        zip1    v4.16b, v12.16b, v4.16b

        fcvtl2  v1.4s, v0.8h
        fcvtl   v0.4s, v0.4h
        fcvtl2  v3.4s, v2.8h
        fcvtl   v2.4s, v2.4h
        fcvtl2  v5.4s, v4.8h
        fcvtl   v4.4s, v4.4h
        fcvtl2  v7.4s, v6.8h
        fcvtl   v6.4s, v6.4h

        fmul    v0.4s, v0.4s, v16.4s
        fmul    v1.4s, v1.4s, v17.4s
        fmul    v2.4s, v2.4s, v18.4s
        fmul    v3.4s, v3.4s, v19.4s
        fmul    v4.4s, v4.4s, v20.4s
        fmul    v5.4s, v5.4s, v21.4s
        fmul    v6.4s, v6.4s, v22.4s
        fmul    v7.4s, v7.4s, v23.4s

        st1     {{v0.16b-v3.16b}}, [{o}], #64
        st1     {{v4.16b-v7.16b}}, [{o}], #64

        subs    {k2}, {k2}, #1
        bne     3b

        add     {i}, {i}, 64 // skip scales
        subs    {k}, {k}, 32
        bne     2b
            ",
        lookup_table = in(reg) &lookup_table,
        k = inout(reg) k => _,
        k2 = out(reg) _,
        scales = out(reg) _,
        i = inout(reg) input => _,
        o = inout(reg) output => _,
        out("v0") _, out("v1") _, out("v2") _, out("v3") _,
        out("v4") _, out("v5") _, out("v6") _, out("v7") _,
        out("v8") _, out("v9") _, out("v10") _, out("v11") _,
        out("v12") _, out("v13") _, out("v14") _, out("v15") _,
        out("v16") _, out("v17") _, out("v18") _, out("v19") _,
        out("v20") _, out("v21") _, out("v22") _, out("v23") _,
        );
    }
}

bail_stub!(aarch64; unsafe fn kernel_packed_32_q40_to_f32(*const u8, *mut u8, usize));

panel_extractor!(aarch64; kernel_packed_32_q20t_to_f32 as packed_32_q20t_to_f32(
    Box::new(super::q20tp32()),
    f32::packing(32).align(16)
));

// NEON unpack of a ternary (Q2_0_T) r=32, zip=0 panel into an f32 packing(32) panel.
// Per 32-block: 32 f16 row-scales then 32 k-positions, each with 32 rows of 2-bit codes
// (8 bytes). Row `r` of a position is `(byte[r/4] >> 2*(r%4)) & 3`: a table lookup spreads
// each byte over four lanes and a per-lane shift picks the code, then `(code - 1) * scale`.
#[cfg(target_arch = "aarch64")]
unsafe fn kernel_packed_32_q20t_to_f32(input: *const u8, output: *mut u8, k: usize) {
    use std::arch::aarch64::*;
    unsafe {
        debug_assert!(k % 32 == 0);
        let spread_lo = vld1q_u8([0u8, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3].as_ptr());
        let spread_hi = vld1q_u8([4u8, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7].as_ptr());
        let shifts =
            vld1q_s8([0i8, -2, -4, -6, 0, -2, -4, -6, 0, -2, -4, -6, 0, -2, -4, -6].as_ptr());
        let three = vdupq_n_u8(3);
        let one = vdupq_n_s8(1);
        let mut i = input;
        let mut o = output as *mut f32;
        for _ in 0..k / 32 {
            let mut scales = [0f32; 32];
            for (r, s) in scales.iter_mut().enumerate() {
                *s = (i as *const f16).add(r).read_unaligned().to_f32();
            }
            let scales: [float32x4_t; 8] =
                std::array::from_fn(|q| vld1q_f32(scales.as_ptr().add(4 * q)));
            i = i.add(64);
            for _ in 0..32 {
                let bytes = vld1_u8(i);
                let bytes = vcombine_u8(bytes, bytes);
                for (half, spread) in [spread_lo, spread_hi].into_iter().enumerate() {
                    let codes = vandq_u8(vshlq_u8(vqtbl1q_u8(bytes, spread), shifts), three);
                    let codes = vsubq_s8(vreinterpretq_s8_u8(codes), one);
                    let lo = vmovl_s8(vget_low_s8(codes));
                    let hi = vmovl_high_s8(codes);
                    let quads = [
                        vmovl_s16(vget_low_s16(lo)),
                        vmovl_high_s16(lo),
                        vmovl_s16(vget_low_s16(hi)),
                        vmovl_high_s16(hi),
                    ];
                    for (q, quad) in quads.into_iter().enumerate() {
                        vst1q_f32(o, vmulq_f32(vcvtq_f32_s32(quad), scales[4 * half + q]));
                        o = o.add(4);
                    }
                }
                i = i.add(8);
            }
        }
    }
}

bail_stub!(aarch64; unsafe fn kernel_packed_32_q20t_to_f32(*const u8, *mut u8, usize));
