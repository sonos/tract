// AVX-512 (zmm, 16-wide) error function kernel. Mirrors generic/erf.rs::serf
// (Abramowitz & Stegun 7.1.26 six-coefficient polynomial) but runs the
// polynomial via FMA chains over 4 zmm registers per iteration (64 lanes per
// loop step). Validated against the generic scalar reference via
// erf_frame_tests! at SuperApproximate tolerance.
//
// Algorithm (per lane):
//   signum = sign(x);  abs = |x|
//   y = a6
//   y = y*abs + a5            (Horner FMA)
//   y = y*abs + a4
//   y = y*abs + a3
//   y = y*abs + a2
//   y = y*abs + a1
//   y = y * abs               (final factor of abs)
//   y = y + 1
//   y = y^16                  (4 sequential squares)
//   y = 1 / y                 (vdivps, full IEEE precision)
//   y = 1 - y
//   result = copysign(y, x)

routine_ew_rust!(x86_64;
    f32,
    x86_64_avx512_erf_f32_64n,
    64,
    16,
    #[inline(never)]
    fn run(buf: &mut [f32], _: ()) {
        debug_assert!(buf.len() % Self::nr() == 0);
        debug_assert!(buf.as_ptr() as usize % Self::alignment_bytes() == 0);
        if buf.is_empty() {
            return;
        }
        unsafe { x86_64_avx512_erf_f32_64n_run(buf) }
    },
    func(Erf),
    isa(X86_64Avx512f)
);

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn x86_64_avx512_erf_f32_64n_run(buf: &mut [f32]) {
    unsafe {
        let len = buf.len();
        let ptr = buf.as_ptr();
        const A1: f32 = 0.0705230784;
        const A2: f32 = 0.0422820123;
        const A3: f32 = 0.0092705272;
        const A4: f32 = 0.0001520143;
        const A5: f32 = 0.0002765672;
        const A6: f32 = 0.0000430638;
        // 0x7fffffff: positive-finite mask (clears sign bit). As f32 bits, this
        // is NaN; we never use it as a numeric value — only as a bit mask via vandps.
        const ABS_MASK: f32 = f32::from_bits(0x7fffffff);
        const SIGN_MASK: f32 = f32::from_bits(0x80000000);
        std::arch::asm!("
            // broadcast constants (xmmN -> zmmN, broadcast across all 16 lanes)
            vbroadcastss zmm0, xmm0           // a1
            vbroadcastss zmm1, xmm1           // a2
            vbroadcastss zmm2, xmm2           // a3
            vbroadcastss zmm3, xmm3           // a4
            vbroadcastss zmm4, xmm4           // a5
            vbroadcastss zmm5, xmm5           // a6
            vbroadcastss zmm6, xmm6           // 1.0
            vbroadcastss zmm7, xmm7           // abs mask (0x7fffffff)
            vbroadcastss zmm8, xmm8           // sign mask (0x80000000)
            2:
                // load 4 zmm of input
                vmovaps zmm9,  [{ptr}]
                vmovaps zmm10, [{ptr} + 64]
                vmovaps zmm11, [{ptr} + 128]
                vmovaps zmm12, [{ptr} + 192]

                // sign[i] = x[i] & SIGN_MASK   (keeps only the sign bit)
                vandps zmm13, zmm9,  zmm8
                vandps zmm14, zmm10, zmm8
                vandps zmm15, zmm11, zmm8
                vandps zmm16, zmm12, zmm8

                // abs[i] = x[i] & ABS_MASK     (clears the sign bit)
                vandps zmm9,  zmm9,  zmm7
                vandps zmm10, zmm10, zmm7
                vandps zmm11, zmm11, zmm7
                vandps zmm12, zmm12, zmm7

                // y = a6 (in zmm17..20, 4 independent channels)
                vmovaps zmm17, zmm5
                vmovaps zmm18, zmm5
                vmovaps zmm19, zmm5
                vmovaps zmm20, zmm5

                // y = y*abs + a5
                vfmadd213ps zmm17, zmm9,  zmm4
                vfmadd213ps zmm18, zmm10, zmm4
                vfmadd213ps zmm19, zmm11, zmm4
                vfmadd213ps zmm20, zmm12, zmm4

                // y = y*abs + a4
                vfmadd213ps zmm17, zmm9,  zmm3
                vfmadd213ps zmm18, zmm10, zmm3
                vfmadd213ps zmm19, zmm11, zmm3
                vfmadd213ps zmm20, zmm12, zmm3

                // y = y*abs + a3
                vfmadd213ps zmm17, zmm9,  zmm2
                vfmadd213ps zmm18, zmm10, zmm2
                vfmadd213ps zmm19, zmm11, zmm2
                vfmadd213ps zmm20, zmm12, zmm2

                // y = y*abs + a2
                vfmadd213ps zmm17, zmm9,  zmm1
                vfmadd213ps zmm18, zmm10, zmm1
                vfmadd213ps zmm19, zmm11, zmm1
                vfmadd213ps zmm20, zmm12, zmm1

                // y = y*abs + a1
                vfmadd213ps zmm17, zmm9,  zmm0
                vfmadd213ps zmm18, zmm10, zmm0
                vfmadd213ps zmm19, zmm11, zmm0
                vfmadd213ps zmm20, zmm12, zmm0

                // y = y * abs  (final factor)
                vmulps zmm17, zmm17, zmm9
                vmulps zmm18, zmm18, zmm10
                vmulps zmm19, zmm19, zmm11
                vmulps zmm20, zmm20, zmm12

                // y = y + 1
                vaddps zmm17, zmm17, zmm6
                vaddps zmm18, zmm18, zmm6
                vaddps zmm19, zmm19, zmm6
                vaddps zmm20, zmm20, zmm6

                // y^16: square 4 times
                vmulps zmm17, zmm17, zmm17
                vmulps zmm18, zmm18, zmm18
                vmulps zmm19, zmm19, zmm19
                vmulps zmm20, zmm20, zmm20

                vmulps zmm17, zmm17, zmm17
                vmulps zmm18, zmm18, zmm18
                vmulps zmm19, zmm19, zmm19
                vmulps zmm20, zmm20, zmm20

                vmulps zmm17, zmm17, zmm17
                vmulps zmm18, zmm18, zmm18
                vmulps zmm19, zmm19, zmm19
                vmulps zmm20, zmm20, zmm20

                vmulps zmm17, zmm17, zmm17
                vmulps zmm18, zmm18, zmm18
                vmulps zmm19, zmm19, zmm19
                vmulps zmm20, zmm20, zmm20

                // y = 1 / y      (full-precision reciprocal, matches generic .recip())
                vdivps zmm21, zmm6, zmm17
                vdivps zmm22, zmm6, zmm18
                vdivps zmm23, zmm6, zmm19
                vdivps zmm24, zmm6, zmm20

                // y = 1 - y
                vsubps zmm21, zmm6, zmm21
                vsubps zmm22, zmm6, zmm22
                vsubps zmm23, zmm6, zmm23
                vsubps zmm24, zmm6, zmm24

                // copysign: stamp the original sign bit onto the (positive) result
                vorps zmm21, zmm21, zmm13
                vorps zmm22, zmm22, zmm14
                vorps zmm23, zmm23, zmm15
                vorps zmm24, zmm24, zmm16

                // store
                vmovaps [{ptr}],       zmm21
                vmovaps [{ptr} + 64],  zmm22
                vmovaps [{ptr} + 128], zmm23
                vmovaps [{ptr} + 192], zmm24

                add {ptr}, 256
                sub {len}, 64
                jnz 2b
            ",
            len = inout(reg) len => _,
            ptr = inout(reg) ptr => _,
            inout("xmm0") A1 => _,
            inout("xmm1") A2 => _,
            inout("xmm2") A3 => _,
            inout("xmm3") A4 => _,
            inout("xmm4") A5 => _,
            inout("xmm5") A6 => _,
            inout("xmm6") 1f32 => _,
            inout("xmm7") ABS_MASK => _,
            inout("xmm8") SIGN_MASK => _,
            out("zmm9")  _, out("zmm10") _, out("zmm11") _, out("zmm12") _,
            out("zmm13") _, out("zmm14") _, out("zmm15") _, out("zmm16") _,
            out("zmm17") _, out("zmm18") _, out("zmm19") _, out("zmm20") _,
            out("zmm21") _, out("zmm22") _, out("zmm23") _, out("zmm24") _,
        );
    }
}

// AVX2/FMA (ymm, 8-wide) error function kernel. Same polynomial and shape as
// the AVX-512 kernel above, over 4 ymm registers per iteration (32 lanes per
// loop step), for the x86_64 tier without AVX-512.
routine_ew_rust!(x86_64;
    f32,
    x86_64_fma_erf_f32_32n,
    32,
    8,
    #[inline(never)]
    fn run(buf: &mut [f32], _: ()) {
        debug_assert!(buf.len() % Self::nr() == 0);
        debug_assert!(buf.as_ptr() as usize % Self::alignment_bytes() == 0);
        if buf.is_empty() {
            return;
        }
        unsafe { x86_64_fma_erf_f32_32n_run(buf) }
    },
    func(Erf),
    isa(X86_64Fma)
);

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn x86_64_fma_erf_f32_32n_run(buf: &mut [f32]) {
    unsafe {
        use std::arch::x86_64::*;
        const A1: f32 = 0.0705230784;
        const A2: f32 = 0.0422820123;
        const A3: f32 = 0.0092705272;
        const A4: f32 = 0.0001520143;
        const A5: f32 = 0.0002765672;
        const A6: f32 = 0.0000430638;
        // 0x7fffffff / 0x80000000 are bit masks, never numeric values: as f32
        // the first is NaN. Used only through vandps/vorps.
        let abs_mask = _mm256_set1_ps(f32::from_bits(0x7fffffff));
        let sign_mask = _mm256_set1_ps(f32::from_bits(0x80000000));
        let a1 = _mm256_set1_ps(A1);
        let a2 = _mm256_set1_ps(A2);
        let a3 = _mm256_set1_ps(A3);
        let a4 = _mm256_set1_ps(A4);
        let a5 = _mm256_set1_ps(A5);
        let a6 = _mm256_set1_ps(A6);
        let one = _mm256_set1_ps(1.0);

        let erf8 = |x: __m256| -> __m256 {
            let abs = _mm256_and_ps(x, abs_mask);
            let sign = _mm256_and_ps(x, sign_mask);
            let y = _mm256_fmadd_ps(a6, abs, a5);
            let y = _mm256_fmadd_ps(y, abs, a4);
            let y = _mm256_fmadd_ps(y, abs, a3);
            let y = _mm256_fmadd_ps(y, abs, a2);
            let y = _mm256_fmadd_ps(y, abs, a1);
            let y = _mm256_fmadd_ps(y, abs, one);
            let y = _mm256_mul_ps(y, y);
            let y = _mm256_mul_ps(y, y);
            let y = _mm256_mul_ps(y, y);
            let y = _mm256_mul_ps(y, y);
            let y = _mm256_sub_ps(one, _mm256_div_ps(one, y));
            _mm256_or_ps(_mm256_and_ps(y, abs_mask), sign)
        };

        let ptr = buf.as_mut_ptr();
        let mut i = 0;
        while i < buf.len() {
            let p = ptr.add(i);
            let r0 = erf8(_mm256_loadu_ps(p));
            let r1 = erf8(_mm256_loadu_ps(p.add(8)));
            let r2 = erf8(_mm256_loadu_ps(p.add(16)));
            let r3 = erf8(_mm256_loadu_ps(p.add(24)));
            _mm256_storeu_ps(p, r0);
            _mm256_storeu_ps(p.add(8), r1);
            _mm256_storeu_ps(p.add(16), r2);
            _mm256_storeu_ps(p.add(24), r3);
            i += 32;
        }
    }
}
