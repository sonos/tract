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
