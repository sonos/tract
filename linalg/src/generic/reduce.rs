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
            // f32 accumulator: a row long enough for the running sum to outgrow
            // its own terms stalls in f16, matching what the AVX-512 f16 kernel
            // already does.
            let mut sum = 0f32;
            for v in x.iter_mut() {
                let y = *v - max;
                let y = f16::from_f32(fast_compact_exp_f32(y.to_f32()));
                *v = y;
                sum += y.to_f32();
            }
            f16::from_f32(sum)
        },
        fn reduce_two(a: f16, b: f16) -> f16 {
            a + b
        }
    );

    /// exp(x - max) with a Cody-Waite reduction and a degree-6 fit: accurate to
    /// about 1e-7 relative, against [`fast_compact_exp_f32`]'s 6e-2, while still
    /// being all FMAs so the row loop vectorizes.
    #[inline(always)]
    pub fn accurate_exp_f32(x: f32) -> f32 {
        const LOG2E: f32 = 1.442_695_04;
        const LN2_HI: f32 = 0.693_145_75;
        const LN2_LO: f32 = 1.428_606_8e-6;
        const MAGIC: f32 = 12_582_912.0;
        // The argument is `v - max` over a row, so it is never positive; a
        // positive value only arises from the f32::MIN lanes a short row is
        // padded with, and below -103 exp underflows to zero anyway. Both
        // comparisons are false for NaN, so a fully masked row still reduces to
        // NaN as the scalar form does.
        // Not a range check: `contains` is true-by-negation for NaN, which would
        // return zero where the scalar form propagates it.
        #[allow(clippy::manual_range_contains)]
        if x < -103.0 || x > 0.0 {
            return 0.0;
        }
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
        q * scale
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
        },
        fn reduce_two(a: f32, b: f32) -> f32 {
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
}

#[cfg(test)]
mod f16_accumulators {
    use super::*;
    use crate::frame::reduce::{MapReduceKer, ReduceKer};
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

    #[test]
    fn long_rows_keep_their_sum() {
        for len in [1024usize, 4096, 8192] {
            let max = f16::from_f32(0.0);
            let mut row: Vec<f16> = vec![f16::from_f32(0.0); len];
            let got = softmax_l2::HSoftMaxL2::red().run_with_params(&mut row, max).unwrap();

            let want: f32 = row.iter().map(|v| v.to_f32()).sum();
            let err = (got.to_f32() - want).abs() / want;
            assert!(err < 0.01, "len {len}: kernel returned {got}, row sums to {want}, rel {err}");
        }
    }
}
