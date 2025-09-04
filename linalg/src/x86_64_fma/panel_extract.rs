use super::*;
use crate::Ops;
use crate::pack::{PackedFormat, Packing};
use tract_data::internal::*;

pub fn plug_avx2(ops: &mut Ops) {
    ops.panel_extractors.extend([packed_32_q40_to_f32.clone(), packed_32_f16_to_f32.clone()]);
}

pub fn plug_avx512f(ops: &mut Ops) {
    ops.panel_extractors.extend([packed_64_q40_to_f32.clone(), packed_64_f16_to_f32.clone()]);
}

panel_extractor!(kernel_packed_64_q40_to_f32 as packed_64_q40_to_f32(
    Box::new(super::mmm::pq40_r64()),
    f32::packing(64).align(64)
) where(AVX512F));

panel_extractor!(kernel_packed_64_f16_to_f32 as packed_64_f16_to_f32(
    Box::new(PackedFormat::new(f16::datum_type(), 64, 64)),
    f32::packing(64).align(64)
) where(AVX512F));

panel_extractor!(kernel_packed_32_q40_to_f32 as packed_32_q40_to_f32(
    Box::new(super::mmm::pq40_r32()),
    f32::packing(32).align(32)
) where(AVX2));

panel_extractor!(kernel_packed_32_f16_to_f32 as packed_32_f16_to_f32(
    Box::new(PackedFormat::new(f16::datum_type(), 32, 32)),
    f32::packing(32).align(32)
) where(AVX2));

#[target_feature(enable = "avx512f")]
unsafe fn kernel_packed_64_q40_to_f32(input: *const u8, output: *mut u8, k: usize) {
    unsafe {
        if k == 0 {
            return;
        }
        debug_assert!(k % 64 == 0);
        debug_assert!(output as usize % 64 == 0);
        std::arch::asm!("
    vpbroadcastd    zmm30, dword ptr [{mask}]
    vpbroadcastd    zmm31, dword ptr [{eight}]

    2:
        // Load 4x128-bit chunks for scales (16 f16 values each = 64 total)
        vmovaps         xmm4, [{i}]
        vmovaps         xmm5, [{i} + 16]
        vmovaps         xmm6, [{i} + 32]
        vmovaps         xmm7, [{i} + 48]
        vmovaps         xmm8, [{i} + 64]
        vmovaps         xmm9, [{i} + 80]
        vmovaps         xmm10, [{i} + 96]
        vmovaps         xmm11, [{i} + 112]
        
        // Convert f16 to f32 (8 values per conversion)
        vcvtph2ps       zmm4, ymm4
        vcvtph2ps       zmm5, ymm5
        vcvtph2ps       zmm6, ymm6
        vcvtph2ps       zmm7, ymm7
        
        add             {i}, 128

        mov {k2}, 64
    3:
        // Load 64 nibbles (32 bytes)
        vmovdqu8        ymm12, [{i}]
        
        // Extract lower nibbles
        vpandd          ymm13, ymm12, ymm30
        vpmovzxbd       zmm14, xmm13           // First 16 nibbles
        vextracti128    xmm13, ymm13, 1
        vpmovzxbd       zmm15, xmm13           // Next 16 nibbles
        
        // Extract upper nibbles
        vpsrlw          ymm12, ymm12, 4
        vpandd          ymm12, ymm12, ymm30
        vpmovzxbd       zmm16, xmm12           // First 16 upper nibbles
        vextracti128    xmm12, ymm12, 1
        vpmovzxbd       zmm17, xmm12           // Next 16 upper nibbles
        
        // Subtract 8 (dequantize offset)
        vpsubd          zmm14, zmm14, zmm31
        vpsubd          zmm15, zmm15, zmm31
        vpsubd          zmm16, zmm16, zmm31
        vpsubd          zmm17, zmm17, zmm31
        
        // Convert to float
        vcvtdq2ps       zmm14, zmm14
        vcvtdq2ps       zmm15, zmm15
        vcvtdq2ps       zmm16, zmm16
        vcvtdq2ps       zmm17, zmm17
        
        // Apply scales
        vmulps          zmm14, zmm14, zmm4
        vmulps          zmm15, zmm15, zmm5
        vmulps          zmm16, zmm16, zmm6
        vmulps          zmm17, zmm17, zmm7
        
        // Store results
        vmovaps         [{o}], zmm14
        vmovaps         [{o}+64], zmm15
        vmovaps         [{o}+128], zmm16
        vmovaps         [{o}+192], zmm17
        
        add             {i}, 32
        add             {o}, 256
        sub             {k2}, 1
        jnz             3b

        sub {k}, 64
        jnz 2b;
            ",
        mask = in(reg) &0x0F0F0F0F,
        eight = in(reg) &0x08,
        k = inout(reg) k => _,
        k2 = out(reg) _,
        i = inout(reg) input => _,
        o = inout(reg) output => _,
        out("zmm4") _, out("zmm5") _, out("zmm6") _, out("zmm7") _,
        out("zmm8") _, out("zmm9") _, out("zmm10") _, out("zmm11") _,
        out("zmm12") _, out("zmm13") _, out("zmm14") _, out("zmm15") _,
        out("zmm16") _, out("zmm17") _, out("zmm30") _, out("zmm31") _
        );
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn kernel_packed_64_f16_to_f32(input: *const u8, output: *mut u8, k: usize) {
    unsafe {
        if k == 0 {
            return;
        }
        debug_assert!(output as usize % 64 == 0);
        std::arch::asm!("
    2:
        // Load 4 chunks of 16 f16 values each (128 bytes total)
        vmovdqu16       ymm4, [{i}]
        vmovdqu16       ymm5, [{i} + 32]
        vmovdqu16       ymm6, [{i} + 64]
        vmovdqu16       ymm7, [{i} + 96]

        // Convert f16 to f32 (16 values per conversion)
        vcvtph2ps       zmm4, ymm4
        vcvtph2ps       zmm5, ymm5
        vcvtph2ps       zmm6, ymm6
        vcvtph2ps       zmm7, ymm7

        // Store results (64 f32 values = 256 bytes)
        vmovaps         [{o}], zmm4
        vmovaps         [{o}+64], zmm5
        vmovaps         [{o}+128], zmm6
        vmovaps         [{o}+192], zmm7

        add             {i}, 128
        add             {o}, 256

        sub {k}, 1
        jnz 2b;
            ",
        k = inout(reg) k => _,
        i = inout(reg) input => _,
        o = inout(reg) output => _,
        out("zmm4") _, out("zmm5") _, out("zmm6") _, out("zmm7") _,
        );
    }
}

#[target_feature(enable = "avx2")]
unsafe fn kernel_packed_32_q40_to_f32(input: *const u8, output: *mut u8, k: usize) {
    unsafe {
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
        vmovaps         xmm8, [{i}]            
        vpand           xmm10, xmm8, xmm14     
        vpmovzxbd       ymm9, xmm10            

        vpermilpd       xmm10, xmm10, 1        
        vpmovzxbd       ymm10, xmm10           

        vpsrlw          xmm8, xmm8, 4
        vpand           xmm12, xmm8, xmm14      
        vpmovzxbd       ymm11, xmm12            
        vpermilpd       xmm12, xmm12, 1         
        vpmovzxbd       ymm12, xmm12            

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
}

#[target_feature(enable = "avx2")]
unsafe fn kernel_packed_32_f16_to_f32(input: *const u8, output: *mut u8, k: usize) {
    unsafe {
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
}
