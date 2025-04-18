use super::FP16;
use crate::block_quant::{PackedBlockQuantFormat, Q4_0};
use crate::pack::Packing;
use crate::Ops;
use tract_data::internal::*;

pub fn plug(ops: &mut Ops) {
    ops.panel_extractors.push(packed_64_q40_to_f16.clone());
}

panel_extractor!(kernel_packed_64_q40_to_f16 as packed_64_q40_to_f16(
    Box::new(PackedBlockQuantFormat::new(&Q4_0, 64, 16, true)),
    f16::packing(64).align(16)
) where(FP16));

#[target_feature(enable = "fp16")]
unsafe fn kernel_packed_64_q40_to_f16(input: *const u8, output: *mut u8, k: usize) {
    if k == 0 {
        return;
    }
    let lookup_table: [u8; 16] = [
        0xc8, 0xc7, 0xc6, 0xc5, 0xc4, 0xc2, 0xc0, 0xbc, 0x00, 0x3c, 0x40, 0x42, 0x44, 0x45, 0x46,
        0x47,
    ];
    std::arch::asm!("
    ld1      {{v13.16b}}, [{lookup_table}]
    movi     v15.16b, 15
    eor      v12.16b, v12.16b, v12.16b

    2:
        add     {scales}, {i}, 1024  // scales at end: 32 (cols) * 64 (rows) / 2 (half byte)
        ld1     {{v16.16b-v19.16b}}, [{scales}], #64
        ld1     {{v20.16b-v23.16b}}, [{scales}]

        mov     {k2}, 32
    3:
        ld1     {{ v9.16b-v10.16b }}, [{i}], #32

        and     v0.16b, v9.16b, v15.16b
        ushr    v2.16b, v9.16b, 4

        and     v4.16b, v10.16b, v15.16b
        ushr    v6.16b, v10.16b, 4

        tbl     v0.16b, {{ v13.16b }}, v0.16b
        tbl     v2.16b, {{ v13.16b }}, v2.16b
        tbl     v4.16b, {{ v13.16b }}, v4.16b
        tbl     v6.16b, {{ v13.16b }}, v6.16b

        zip2    v1.16b, v12.16b, v0.16b
        zip2    v3.16b, v12.16b, v2.16b
        zip2    v5.16b, v12.16b, v4.16b
        zip2    v7.16b, v12.16b, v6.16b

        zip1    v0.16b, v12.16b, v0.16b
        zip1    v2.16b, v12.16b, v2.16b
        zip1    v4.16b, v12.16b, v4.16b
        zip1    v6.16b, v12.16b, v6.16b

        fmul    v0.8h, v0.8h, v16.8h
        fmul    v1.8h, v1.8h, v17.8h
        fmul    v2.8h, v2.8h, v18.8h
        fmul    v3.8h, v3.8h, v19.8h
        fmul    v4.8h, v4.8h, v20.8h
        fmul    v5.8h, v5.8h, v21.8h
        fmul    v6.8h, v6.8h, v22.8h
        fmul    v7.8h, v7.8h, v23.8h

        st1     {{v0.16b-v3.16b}}, [{o}], #64
        st1     {{v4.16b-v7.16b}}, [{o}], #64

        subs    {k2}, {k2}, #1
        bne     3b

        add     {i}, {i}, 128 // skip scales
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
