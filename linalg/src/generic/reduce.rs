// Reduce<max> generic implementation
pub mod max {
    pub use tract_data::internal::f16;

    reduce_impl_wrap!(
        f32,
        SMax4,
        4,
        4,
        (),
        f32::MIN,
        fn run(x: &[f32], _: ()) -> f32 {
            debug_assert!(x.len() % Self::nr() == 0);
            debug_assert!(x.as_ptr() as usize % Self::alignment_bytes() == 0);
            *x.iter().max_by(|a, b| a.total_cmp(b)).unwrap()
        },
        fn reduce_two(a: f32, b: f32) -> f32 {
            a.max(b)
        }
    );

    reduce_impl_wrap!(
        f16,
        HMax8,
        8,
        8,
        (),
        f16::MIN,
        fn run(x: &[f16], _: ()) -> f16 {
            debug_assert!(x.len() % Self::nr() == 0);
            debug_assert!(x.as_ptr() as usize % Self::alignment_bytes() == 0);
            *x.iter().max_by(|a, b| a.total_cmp(b)).unwrap()
        },
        fn reduce_two(a: f16, b: f16) -> f16 {
            a.max(b)
        }
    );

    #[cfg(test)]
    #[macro_use]
    pub mod s {
        crate::max_frame_tests!(true, f32, crate::generic::reduce::max::SMax4);
    }

    #[cfg(test)]
    #[macro_use]
    pub mod h {
        use super::*;
        crate::max_frame_tests!(true, f16, crate::generic::reduce::max::HMax8);
    }
}

// Reduce<min> generic implementation
pub mod min {
    pub use tract_data::internal::f16;

    reduce_impl_wrap!(
        f32,
        SMin4,
        4,
        4,
        (),
        f32::MAX,
        fn run(x: &[f32], _: ()) -> f32 {
            debug_assert!(x.len() % Self::nr() == 0);
            debug_assert!(x.as_ptr() as usize % Self::alignment_bytes() == 0);
            *x.iter().min_by(|a, b| a.total_cmp(b)).unwrap()
        },
        fn reduce_two(a: f32, b: f32) -> f32 {
            a.min(b)
        }
    );

    #[cfg(test)]
    #[macro_use]
    pub mod s {
        crate::min_frame_tests!(true, f32, crate::generic::reduce::min::SMin4);
    }
}

// Reduce<sum> generic implementation
pub mod sum {
    use crate::num_traits::Zero;
    pub use tract_data::internal::f16;

    reduce_impl_wrap!(
        f32,
        SSum4,
        4,
        4,
        (),
        0.0,
        fn run(x: &[f32], _: ()) -> f32 {
            debug_assert!(x.len() % Self::nr() == 0);
            debug_assert!(x.as_ptr() as usize % Self::alignment_bytes() == 0);
            x.iter().sum::<f32>()
        },
        fn reduce_two(a: f32, b: f32) -> f32 {
            a + b
        }
    );

    reduce_impl_wrap!(
        f16,
        HSum8,
        8,
        8,
        (),
        f16::zero(),
        fn run(x: &[f16], _: ()) -> f16 {
            debug_assert!(x.len() % Self::nr() == 0);
            debug_assert!(x.as_ptr() as usize % Self::alignment_bytes() == 0);
            // f32 accumulator: a row long enough for the running sum to outgrow
            // its own terms stalls in f16. The vector kernels are shielded by
            // holding one partial per lane; this one is not.
            f16::from_f32(x.iter().map(|v| v.to_f32()).sum::<f32>())
        },
        fn reduce_two(a: f16, b: f16) -> f16 {
            a + b
        }
    );

    #[cfg(test)]
    #[macro_use]
    pub mod s {
        crate::sum_frame_tests!(true, f32, crate::generic::reduce::sum::SSum4);
    }

    #[cfg(test)]
    #[macro_use]
    pub mod h {
        use super::*;
        crate::sum_frame_tests!(true, f16, crate::generic::reduce::sum::HSum8);
    }
}

// Softmax generic implementation
pub mod softmax_l2 {

    /// exp(x - max) with a Cody-Waite reduction and a degree-6 fit: accurate to
    /// about 1e-7 relative while still being all FMAs so the row loop vectorizes.
    #[inline(always)]
    pub fn accurate_exp_f32(x: f32) -> f32 {
        const LOG2E: f32 = 1.442_695_04;
        const LN2_HI: f32 = 0.693_145_75;
        const LN2_LO: f32 = 1.428_606_8e-6;
        const MAGIC: f32 = 12_582_912.0;
        let kf = (x * LOG2E + MAGIC) - MAGIC;
        let rr = kf.mul_add(-LN2_LO, kf.mul_add(-LN2_HI, x));
        let mut q = 1.383684405e-03f32;
        q = q.mul_add(rr, 8.374815793e-03);
        q = q.mul_add(rr, 4.166822560e-02);
        q = q.mul_add(rr, 1.666642017e-01);
        q = q.mul_add(rr, 4.999999208e-01);
        q = q.mul_add(rr, 1.000000036e+00);
        q = q.mul_add(rr, 1.000000001e+00);
        let k = kf as i32;
        let scale = f32::from_bits(((k + 127).clamp(1, 254) as u32) << 23);
        // The argument is `v - max` over a row, so it is never positive; a
        // positive value only arises from the f32::MIN lanes a short row is
        // padded with, and below -103 exp underflows to zero. Selecting here
        // rather than returning early keeps the loop vectorizable. Not a range
        // check: `contains` is true-by-negation for NaN, which would return zero
        // where a fully masked row must still reduce to NaN.
        #[allow(clippy::manual_range_contains)]
        if x < -103.0 || x > 0.0 { 0.0 } else { q * scale }
    }

    /// exp(x - max) over a row, returning the sum. Every lane of the load, the
    /// polynomial and the accumulation stays four wide; auto-vectorization
    /// leaves the integer half of the scale reconstruction scalar, which costs
    /// most of the throughput.
    #[cfg(target_arch = "aarch64")]
    #[inline]
    fn exp_sum_impl(x: &mut [f32], max: f32) -> f32 {
        use std::arch::aarch64::*;
        unsafe {
            #[inline(always)]
            unsafe fn exp_ps(x: float32x4_t) -> float32x4_t {
                unsafe {
                    let kf = vrndnq_f32(vmulq_f32(x, vdupq_n_f32(1.442_695_04)));
                    let mut rr = vfmsq_f32(x, kf, vdupq_n_f32(0.693_145_75));
                    rr = vfmsq_f32(rr, kf, vdupq_n_f32(1.428_606_8e-6));
                    let mut q = vdupq_n_f32(8.297653546e-03);
                    q = vfmaq_f32(vdupq_n_f32(4.191538191e-02), q, rr);
                    q = vfmaq_f32(vdupq_n_f32(1.666757475e-01), q, rr);
                    q = vfmaq_f32(vdupq_n_f32(4.999889485e-01), q, rr);
                    q = vfmaq_f32(vdupq_n_f32(9.999996920e-01), q, rr);
                    q = vfmaq_f32(vdupq_n_f32(1.000000072e+00), q, rr);
                    let k = vcvtq_s32_f32(kf);
                    // The biased exponent must stay a valid field: k + 127 goes
                    // non-positive around x = -88, and shifting that in would
                    // build -inf instead of a small number, poisoning the sum.
                    let biased = vmaxq_s32(
                        vminq_s32(vaddq_s32(k, vdupq_n_s32(127)), vdupq_n_s32(254)),
                        vdupq_n_s32(1),
                    );
                    let scale = vreinterpretq_f32_s32(vshlq_n_s32(biased, 23));
                    let out = vorrq_u32(
                        vcltq_f32(x, vdupq_n_f32(-103.0)),
                        vcgtq_f32(x, vdupq_n_f32(0.0)),
                    );
                    vbslq_f32(out, vdupq_n_f32(0.0), vmulq_f32(q, scale))
                }
            }
            let vm = vdupq_n_f32(max);
            let mut vsum = vdupq_n_f32(0.0);
            let mut i = 0;
            while i + 4 <= x.len() {
                let y = exp_ps(vsubq_f32(vld1q_f32(x.as_ptr().add(i)), vm));
                vst1q_f32(x.as_mut_ptr().add(i), y);
                vsum = vaddq_f32(vsum, y);
                i += 4;
            }
            let mut sum = vaddvq_f32(vsum);
            for v in &mut x[i..] {
                let y = accurate_exp_f32(*v - max);
                *v = y;
                sum += y;
            }
            sum
        }
    }

    /// simd128 counterpart. wasm has no fused multiply-add outside relaxed-simd,
    /// so the polynomial is a separate multiply and add per step; the reduction
    /// is what matters here, since LLVM does not vectorize f32 reductions on
    /// this target at all.
    #[cfg(all(target_family = "wasm", target_feature = "simd128"))]
    #[inline]
    fn exp_sum_impl(x: &mut [f32], max: f32) -> f32 {
        use std::arch::wasm32::*;
        #[inline(always)]
        fn exp_ps(x: v128) -> v128 {
            let kf = f32x4_nearest(f32x4_mul(x, f32x4_splat(1.442_695_04)));
            let mut rr = f32x4_sub(x, f32x4_mul(kf, f32x4_splat(0.693_145_75)));
            rr = f32x4_sub(rr, f32x4_mul(kf, f32x4_splat(1.428_606_8e-6)));
            let mut q = f32x4_splat(8.297653546e-03);
            q = f32x4_add(f32x4_splat(4.191538191e-02), f32x4_mul(q, rr));
            q = f32x4_add(f32x4_splat(1.666757475e-01), f32x4_mul(q, rr));
            q = f32x4_add(f32x4_splat(4.999889485e-01), f32x4_mul(q, rr));
            q = f32x4_add(f32x4_splat(9.999996920e-01), f32x4_mul(q, rr));
            q = f32x4_add(f32x4_splat(1.000000072e+00), f32x4_mul(q, rr));
            let k = i32x4_trunc_sat_f32x4(kf);
            // Same guard as the scalar and NEON forms: an out-of-range biased
            // exponent would shift in as -inf rather than a small number.
            let biased = i32x4_max(
                i32x4_min(i32x4_add(k, i32x4_splat(127)), i32x4_splat(254)),
                i32x4_splat(1),
            );
            let scale = i32x4_shl(biased, 23);
            let out = v128_or(f32x4_lt(x, f32x4_splat(-103.0)), f32x4_gt(x, f32x4_splat(0.0)));
            v128_bitselect(f32x4_splat(0.0), f32x4_mul(q, scale), out)
        }
        let vm = f32x4_splat(max);
        let mut vsum = f32x4_splat(0.0);
        let mut i = 0;
        while i + 4 <= x.len() {
            let y = exp_ps(f32x4_sub(unsafe { v128_load(x.as_ptr().add(i) as *const v128) }, vm));
            unsafe { v128_store(x.as_mut_ptr().add(i) as *mut v128, y) };
            vsum = f32x4_add(vsum, y);
            i += 4;
        }
        let mut sum = f32x4_extract_lane::<0>(vsum)
            + f32x4_extract_lane::<1>(vsum)
            + f32x4_extract_lane::<2>(vsum)
            + f32x4_extract_lane::<3>(vsum);
        for v in &mut x[i..] {
            let y = accurate_exp_f32(*v - max);
            *v = y;
            sum += y;
        }
        sum
    }

    #[cfg(not(any(
        target_arch = "aarch64",
        all(target_family = "wasm", target_feature = "simd128")
    )))]
    #[inline]
    fn exp_sum_impl(x: &mut [f32], max: f32) -> f32 {
        let mut acc = [0f32; 4];
        let mut it = x.chunks_exact_mut(4);
        for c in &mut it {
            for (j, v) in c.iter_mut().enumerate() {
                let y = accurate_exp_f32(*v - max);
                *v = y;
                acc[j] += y;
            }
        }
        let mut sum = (acc[0] + acc[1]) + (acc[2] + acc[3]);
        for v in it.into_remainder().iter_mut() {
            let y = accurate_exp_f32(*v - max);
            *v = y;
            sum += y;
        }
        sum
    }

    map_reduce_impl_wrap!(
        f32,
        SSoftMaxL2Accurate,
        4,
        4,
        f32,
        f32::NEG_INFINITY,
        0.0,
        fn run(x: &mut [f32], max: f32) -> f32 {
            debug_assert!(x.len() % Self::nr() == 0);
            debug_assert!(x.as_ptr() as usize % Self::alignment_bytes() == 0);
            exp_sum_impl(x, max)
        },
        fn reduce_two(a: f32, b: f32) -> f32 {
            a + b
        }
    );

    #[cfg(test)]
    #[macro_use]
    pub mod s {
        crate::softmax_l2_frame_tests!(true, f32, super::SSoftMaxL2Accurate);
    }
}

#[cfg(test)]
mod f16_accumulators {
    use super::*;
    use crate::frame::reduce::ReduceKer;
    use tract_data::internal::f16;

    /// The returned row sum must stay close to the same sum taken in f32. A row
    /// long enough for the running total to outgrow its own terms is the case an
    /// f16 accumulator silently drops.
    #[test]
    fn plain_sum_keeps_long_rows() {
        for len in [1024usize, 4096, 8192] {
            let row: Vec<f16> = vec![f16::from_f32(1.0); len];
            let got = sum::HSum8::red().run(&row).unwrap().to_f32();
            let err = (got - len as f32).abs() / len as f32;
            assert!(err < 0.01, "len {len}: summed to {got}, rel {err}");
        }
    }
}
