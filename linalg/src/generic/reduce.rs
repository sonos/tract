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
            x.iter().sum::<f16>()
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
    use crate::num_traits::Zero;
    use tract_data::internal::f16;

    map_reduce_impl_wrap!(
        f32,
        SSoftMaxL2,
        4,
        4,
        f32,
        f32::MIN,
        0.0,
        fn run(x: &mut [f32], max: f32) -> f32 {
            debug_assert!(x.len() % Self::nr() == 0);
            debug_assert!(x.as_ptr() as usize % Self::alignment_bytes() == 0);
            let mut sum = 0.;
            for v in x.iter_mut() {
                let y = *v - max;
                let y = fast_compact_exp_f32(y);
                *v = y;
                sum += y;
            }
            sum
        },
        fn reduce_two(a: f32, b: f32) -> f32 {
            a + b
        }
    );

    map_reduce_impl_wrap!(
        f16,
        HSoftMaxL2,
        8,
        8,
        f16,
        f16::MIN,
        f16::zero(),
        fn run(x: &mut [f16], max: f16) -> f16 {
            debug_assert!(x.len() % Self::nr() == 0);
            debug_assert!(x.as_ptr() as usize % Self::alignment_bytes() == 0);
            let mut sum = f16::zero();
            for v in x.iter_mut() {
                let y = *v - max;
                let y = f16::from_f32(fast_compact_exp_f32(y.to_f32()));
                *v = y;
                sum += y;
            }
            sum
        },
        fn reduce_two(a: f16, b: f16) -> f16 {
            a + b
        }
    );

    // Accurate (~1-2 ulp) softmax kernels: same map+reduce shape as the
    // FastCompact ones above, but using `accurate_exp_f32` instead of the
    // coarse Schraudolph approximation, and padding the SIMD tail with
    // -inf (so masked/padding lanes contribute exactly 0 to the sum).
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
            let mut sum = 0.;
            for v in x.iter_mut() {
                let y = accurate_exp_f32(*v - max);
                *v = y;
                sum += y;
            }
            sum
        },
        fn reduce_two(a: f32, b: f32) -> f32 {
            a + b
        }
    );

    map_reduce_impl_wrap!(
        f16,
        HSoftMaxL2Accurate,
        8,
        8,
        f16,
        f16::NEG_INFINITY,
        f16::zero(),
        fn run(x: &mut [f16], max: f16) -> f16 {
            debug_assert!(x.len() % Self::nr() == 0);
            debug_assert!(x.as_ptr() as usize % Self::alignment_bytes() == 0);
            let mut sum = f16::zero();
            for v in x.iter_mut() {
                let y = f16::from_f32(accurate_exp_f32((*v - max).to_f32()));
                *v = y;
                sum += y;
            }
            sum
        },
        fn reduce_two(a: f16, b: f16) -> f16 {
            a + b
        }
    );

    // ported from https://github.com/gnuradio/volk/blob/master/kernels/volk/volk_32f_expfast_32f.h
    // probably inspired from https://nic.schraudolph.org/pubs/Schraudolph99.pdf
    // not that the cast to u32 deals with negative right, while implem in volk code are wrong in some
    // corner cases (need a max(0,x) before the u32 conversion)
    pub fn fast_compact_exp_f32(v: f32) -> f32 {
        const MLN2: f32 = 0.6931471805f32;
        const A: f32 = 8388608.0f32;
        const B: f32 = 1065353216.0f32;
        const C: f32 = 60801.0f32;
        const SLOPE: f32 = A / MLN2;
        const OFFSET: f32 = B - C;
        f32::from_bits(((SLOPE * v) + OFFSET) as u32)
    }

    /// Accurate (~1-2 ulp) vectorizable `exp`, used by the SSoftMaxL2Accurate /
    /// HSoftMaxL2Accurate softmax kernels.
    ///
    /// Cephes-style range reduction `x = n*ln2 + r` (Cody-Waite two-part ln2)
    /// followed by a degree-6 polynomial on `r` in `[-ln2/2, ln2/2]` and a
    /// scaling by `2^n` via direct exponent construction. Branch-free apart
    /// from one final `select`, so the caller's per-lane loop auto-vectorizes.
    ///
    /// Guarantees relied upon by softmax:
    /// * `exp(0) == 1.0` exactly (the row max normalises cleanly);
    /// * `exp(-inf) == 0.0` exactly and deep underflow flushes to `0.0`, so a
    ///   fully-masked row sums to 0 and the dispatch's `0·(1/0)` yields `NaN`,
    ///   matching the libc path and the numpy reference (rather than the
    ///   spurious finite 0 the old `f32::MIN` padding produced).
    ///
    /// Softmax only ever evaluates `x = element - max ≤ 0`; the positive domain
    /// is clamped purely for safety.
    #[inline(always)]
    pub fn accurate_exp_f32(x: f32) -> f32 {
        const LOG2E: f32 = 1.44269504088896341f32;
        // Cody-Waite split of ln2: C1 + C2 == ln2 to extra precision.
        const C1: f32 = 0.693359375f32;
        const C2: f32 = -2.12194440e-4f32;
        const HI: f32 = 88.3762626647949f32;
        const LO: f32 = -88.3762626647949f32;
        // polynomial coefficients (Cephes expf)
        const P0: f32 = 1.9875691500e-4f32;
        const P1: f32 = 1.3981999507e-3f32;
        const P2: f32 = 8.3334519073e-3f32;
        const P3: f32 = 4.1665795894e-2f32;
        const P4: f32 = 1.6666665459e-1f32;
        const P5: f32 = 5.0000001201e-1f32;

        let xc = x.clamp(LO, HI);
        let n = xc.mul_add(LOG2E, 0.5).floor();
        let g = (-n).mul_add(C2, (-n).mul_add(C1, xc));
        let z = g * g;
        let mut y = P0;
        y = y.mul_add(g, P1);
        y = y.mul_add(g, P2);
        y = y.mul_add(g, P3);
        y = y.mul_add(g, P4);
        y = y.mul_add(g, P5);
        y = y.mul_add(z, g + 1.0);
        // 2ⁿ by exponent construction; n ∈ [-127, 127] thanks to the clamp.
        let pow2n = f32::from_bits((((n as i32) + 127) as u32) << 23);
        let e = y * pow2n;
        // Branchless underflow flush keeps the dataflow vectorizable:
        // 0 when x < LO, else e (NaN propagates: `NaN < LO` is false).
        let flush = -((x < LO) as i32) as u32;
        f32::from_bits(e.to_bits() & !flush)
    }

    #[cfg(test)]
    #[macro_use]
    pub mod s {
        crate::softmax_l2_frame_tests!(true, f32, super::SSoftMaxL2);
    }

    #[cfg(test)]
    #[macro_use]
    pub mod h {
        use super::*;
        crate::softmax_l2_frame_tests!(true, f16, HSoftMaxL2);
    }

    #[cfg(test)]
    mod accurate {
        use super::{SSoftMaxL2Accurate, accurate_exp_f32};
        use crate::frame::reduce::MapReduceKer;

        // Non-tautological: the accurate kernel's exp is validated against libc
        // exp (the FastCompact kernels are only ever compared against
        // fast_compact itself, which can't catch divergence from true softmax).
        #[test]
        fn accurate_exp_matches_libc() {
            // softmax only feeds x <= 0; sweep that domain densely.
            let mut x = 0.0f32;
            while x > -60.0 {
                let got = accurate_exp_f32(x);
                let want = x.exp();
                let rel = (got - want).abs() / want;
                assert!(rel < 2e-6, "x={x}: accurate={got} libc={want} rel={rel}");
                x -= 0.0007;
            }
        }

        #[test]
        fn accurate_exp_edges() {
            assert_eq!(accurate_exp_f32(0.0), 1.0); // row max normalises cleanly
            assert_eq!(accurate_exp_f32(f32::NEG_INFINITY), 0.0); // fully-masked -> 0
            assert_eq!(accurate_exp_f32(-1000.0), 0.0); // deep underflow flushes
            assert!(accurate_exp_f32(f32::NAN).is_nan()); // NaN propagates
        }

        // A fully-masked row (every element -inf) must sum to exactly 0 -> the
        // dispatch's 1/0 = inf, 0*inf = NaN, matching the libc path and the numpy
        // reference. Driven through the frame (`red().run_with_params`) so the
        // kernel always sees a correctly-aligned slice (the raw `K::run` has a
        // debug_assert on alignment) and the SIMD tail is padded with
        // `map_neutral` (= -inf, exp -> 0), not f32::MIN (exp -> ~1, which would
        // make the row sum to a spurious nonzero and yield a finite 0). Length 6
        // exercises both an aligned 4-lane chunk and a 2-lane padded tail.
        #[test]
        fn fully_masked_row_sums_to_zero() {
            use crate::frame::reduce::MapReduce;
            let op = SSoftMaxL2Accurate::red();
            let mut row = vec![f32::NEG_INFINITY; 6];
            // the max reduce returns f32::MIN (its neutral) for an all -inf row.
            let sum = op.run_with_params(&mut row, f32::MIN).unwrap();
            assert_eq!(sum, 0.0, "fully-masked row must sum to 0, got {sum}");
            assert!(row.iter().all(|&v| v == 0.0));
        }

        // The SIMD tail is padded with `map_neutral`; it must be -inf so padding
        // lanes contribute exp(-inf - max) = 0. With the old f32::MIN padding, a
        // fully-masked row's lanes computed exp(f32::MIN - f32::MIN) = exp(0) ≈ 1,
        // inflating the sum and turning the result into a spurious finite 0
        // instead of NaN.
        #[test]
        fn map_neutral_is_neg_inf() {
            assert_eq!(
                <SSoftMaxL2Accurate as MapReduceKer<f32, f32>>::map_neutral(),
                f32::NEG_INFINITY
            );
        }
    }
}
