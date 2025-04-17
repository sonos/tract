use super::*;
use crate::pack::{PackedFormat, Packing};
use crate::Ops;
use tract_data::internal::*;

pub fn plug(ops: &mut Ops) {
    ops.panel_extractors.extend([packed_32_q40_to_f32.clone(), packed_32_f16_to_f32.clone()]);
}

panel_extractor!(kernel_packed_32_q40_to_f32 as packed_32_q40_to_f32(
    Box::new(super::mmm::pq40_r32()),
    f32::packing(32).align(32)
) where(AVX2));

panel_extractor!(kernel_packed_32_f16_to_f32 as packed_32_f16_to_f32(
    Box::new(PackedFormat::new(f16::datum_type(), 32, 32)),
    f32::packing(32).align(32)
) where(AVX2));

#[target_feature(enable = "avx2")]
unsafe fn kernel_packed_32_q40_to_f32(input: *const u8, output: *mut u8, k: usize) {
    if k == 0 {
        return;
    }
    debug_assert!(k % 32 == 0);
    debug_assert!(output as usize % 32 == 0);
    std::arch::asm!("
    vbroadcastss    ymm14, dword ptr [{mask}]
    vbroadcastss    ymm13, dword ptr [{eight}]

    2:
        vmovaps         xmm4, [{i}]
        vmovaps         xmm5, [{i} + 16]
        vmovaps         xmm6, [{i} + 32]
        vmovaps         xmm7, [{i} + 48]
        vcvtph2ps       ymm4, xmm4
        vcvtph2ps       ymm5, xmm5
        vcvtph2ps       ymm6, xmm6
        vcvtph2ps       ymm7, xmm7
        add             {i}, 64

        mov {k2}, 32
    3:
        vmovaps         xmm8, [{i}]            // 32 nibbles
        vpand           xmm10, xmm8, xmm14     // 16 bytes
        vpmovzxbd       ymm9, xmm10            // 8 u32

        vpermilpd       xmm10, xmm10, 1        // swap 64bit halves
        vpmovzxbd       ymm10, xmm10           // 8 u32

        vpsrlw          xmm8, xmm8, 4
        vpand           xmm12, xmm8, xmm14      // 16 bytes
        vpmovzxbd       ymm11, xmm12            // 8 u32
        vpermilpd       xmm12, xmm12, 1         // swap 64bit halves
        vpmovzxbd       ymm12, xmm12            // 8 u32

        vpsubd          ymm9, ymm9, ymm13
        vpsubd          ymm10, ymm10, ymm13
        vpsubd          ymm11, ymm11, ymm13
        vpsubd          ymm12, ymm12, ymm13

        vcvtdq2ps       ymm9, ymm9
        vcvtdq2ps       ymm10, ymm10
        vcvtdq2ps       ymm11, ymm11
        vcvtdq2ps       ymm12, ymm12

        vmulps          ymm9, ymm9, ymm4
        vmulps          ymm10, ymm10, ymm5
        vmulps          ymm11, ymm11, ymm6
        vmulps          ymm12, ymm12, ymm7

        vmovaps         [{o}], ymm9
        vmovaps         [{o}+32], ymm10
        vmovaps         [{o}+64], ymm11
        vmovaps         [{o}+96], ymm12

        add             {i}, 16
        add             {o}, 128
        sub             {k2}, 1
        jnz             3b

        sub {k}, 32
        jnz 2b;
            ",
    mask = in(reg) &0x0F0F0F0F,
    eight = in(reg) &0x08,
    k = inout(reg) k => _,
    k2 = out(reg) _,
    i = inout(reg) input => _,
    o = inout(reg) output => _,
    out("ymm0") _, out("ymm1") _, out("ymm2") _, out("ymm3") _,
    out("ymm4") _, out("ymm5") _, out("ymm6") _, out("ymm7") _,
    out("ymm8") _, out("ymm9") _, out("ymm10") _, out("ymm11") _,
    out("ymm12") _, out("ymm13") _, out("ymm14") _, out("ymm15") _
    );
}

#[target_feature(enable = "avx2")]
unsafe fn kernel_packed_32_f16_to_f32(input: *const u8, output: *mut u8, k: usize) {
    if k == 0 {
        return;
    }
    debug_assert!(output as usize % 32 == 0);
    std::arch::asm!("
    2:
        vmovaps         xmm4, [{i}]
        vmovaps         xmm5, [{i} + 16]
        vmovaps         xmm6, [{i} + 32]
        vmovaps         xmm7, [{i} + 48]

        vcvtph2ps       ymm4, xmm4
        vcvtph2ps       ymm5, xmm5
        vcvtph2ps       ymm6, xmm6
        vcvtph2ps       ymm7, xmm7

        vmovaps         [{o}], ymm4
        vmovaps         [{o}+32], ymm5
        vmovaps         [{o}+64], ymm6
        vmovaps         [{o}+96], ymm7

        add             {i}, 64
        add             {o}, 128

        sub {k}, 1
        jnz 2b;
            ",
    k = inout(reg) k => _,
    i = inout(reg) input => _,
    o = inout(reg) output => _,
    out("ymm4") _, out("ymm5") _, out("ymm6") _, out("ymm7") _,
    );
}
