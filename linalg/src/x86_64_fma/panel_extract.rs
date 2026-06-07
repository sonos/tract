use super::*;
use crate::Ops;
use crate::pack::{PackedFormat, Packing};
use tract_data::internal::*;

pub fn plug(ops: &mut Ops) {
    ops.panel_extractors.extend([
        packed_32_q40_to_f32.clone(),
        packed_32_q1_58_to_f32.clone(),
        packed_32_f16_to_f32.clone(),
    ]);
}

panel_extractor!(kernel_packed_32_q40_to_f32 as packed_32_q40_to_f32(
    Box::new(super::mmm::pq40_r32()),
    f32::packing(32).align(32)
) where(AVX2));

panel_extractor!(kernel_packed_32_q1_58_to_f32 as packed_32_q1_58_to_f32(
    Box::new(super::mmm::pq1_58_r32()),
    f32::packing(32).align(32)
) where(AVX2));

// AVX2 unpack of a ternary (Q1_58) r=32, zip=0 panel into an f32 packing(32) panel.
// Per 32-block: 32 f16 row-scales then 32 k-positions, each with 32 rows of 2-bit codes
// (8 bytes). For a position, lane `row` wants `(byte[row/4] >> 2*(row%4)) & 3` which the
// per-lane variable shift (vpsrlvd) computes directly after spreading each byte across
// four lanes; then `(code - 1) * scale`.
#[target_feature(enable = "avx2,f16c")]
unsafe fn kernel_packed_32_q1_58_to_f32(input: *const u8, output: *mut u8, k: usize) {
    use std::arch::x86_64::*;
    unsafe {
        if k == 0 {
            return;
        }
        debug_assert!(k % 32 == 0);
        let shift = _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6);
        let three = _mm256_set1_epi32(3);
        let one = _mm256_set1_epi32(1);
        // For group g (rows 8g..8g+7): spread bytes 2g, 2g+1 across 4 lanes each.
        let m = |b: i8| b;
        let masks = [
            _mm_setr_epi8(
                m(0),
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                -128,
                -128,
                -128,
                -128,
                -128,
                -128,
                -128,
                -128,
            ),
            _mm_setr_epi8(
                m(2),
                2,
                2,
                2,
                3,
                3,
                3,
                3,
                -128,
                -128,
                -128,
                -128,
                -128,
                -128,
                -128,
                -128,
            ),
            _mm_setr_epi8(
                m(4),
                4,
                4,
                4,
                5,
                5,
                5,
                5,
                -128,
                -128,
                -128,
                -128,
                -128,
                -128,
                -128,
                -128,
            ),
            _mm_setr_epi8(
                m(6),
                6,
                6,
                6,
                7,
                7,
                7,
                7,
                -128,
                -128,
                -128,
                -128,
                -128,
                -128,
                -128,
                -128,
            ),
        ];
        let mut inp = input;
        let mut out = output as *mut f32;
        let mut kk = k;
        while kk > 0 {
            // 32 f16 row scales -> 4 ymm of 8 f32
            let scales = [
                _mm256_cvtph_ps(_mm_loadu_si128(inp as *const __m128i)),
                _mm256_cvtph_ps(_mm_loadu_si128(inp.add(16) as *const __m128i)),
                _mm256_cvtph_ps(_mm_loadu_si128(inp.add(32) as *const __m128i)),
                _mm256_cvtph_ps(_mm_loadu_si128(inp.add(48) as *const __m128i)),
            ];
            let mut w = inp.add(64);
            for _ in 0..32 {
                let codes = _mm_loadl_epi64(w as *const __m128i);
                for g in 0..4 {
                    let sh = _mm_shuffle_epi8(codes, masks[g]);
                    let mut v = _mm256_cvtepu8_epi32(sh);
                    v = _mm256_srlv_epi32(v, shift);
                    v = _mm256_and_si256(v, three);
                    v = _mm256_sub_epi32(v, one);
                    let f = _mm256_cvtepi32_ps(v);
                    let r = _mm256_mul_ps(f, scales[g]);
                    _mm256_storeu_ps(out.add(g * 8), r);
                }
                out = out.add(32);
                w = w.add(8);
            }
            inp = inp.add(320); // 32 * block_bytes(=10)
            kk -= 32;
        }
    }
}

panel_extractor!(kernel_packed_32_f16_to_f32 as packed_32_f16_to_f32(
    Box::new(PackedFormat::new(f16::datum_type(), 32, 32)),
    f32::packing(32).align(32)
) where(AVX2));

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
