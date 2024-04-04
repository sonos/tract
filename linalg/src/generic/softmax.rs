use num_traits::Zero;
use tract_data::internal::f16;

use crate::frame::reduce::MapReduceKer;

#[derive(Clone, Debug)]
pub struct SSoftMaxL2;

impl MapReduceKer<f32, f32> for SSoftMaxL2 {
    fn name() -> &'static str {
        "generic"
    }

    fn alignment_items() -> usize {
        4
    }

    fn nr() -> usize {
        4
    }

    fn map_neutral() -> f32 {
        f32::MIN
    }

    fn reduce_neutral() -> f32 {
        0.
    }

    fn reduce_two(a: f32, b: f32) -> f32 {
        a + b
    }

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
    }
}

#[derive(Clone, Debug)]
pub struct HSoftMaxL2;

impl MapReduceKer<f16, f16> for HSoftMaxL2 {
    fn name() -> &'static str {
        "generic"
    }

    fn alignment_items() -> usize {
        8
    }

    fn nr() -> usize {
        8
    }

    fn map_neutral() -> f16 {
        f16::MIN
    }

    fn reduce_neutral() -> f16 {
        f16::zero()
    }

    fn reduce_two(a: f16, b: f16) -> f16 {
        a + b
    }

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
    }
}

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
    softmax_l2_frame_tests!(true, f32, crate::generic::softmax::SSoftMaxL2);
}

#[cfg(test)]
#[macro_use]
pub mod h {
    use super::*;
    softmax_l2_frame_tests!(true, f16, crate::generic::softmax::HSoftMaxL2);
}
