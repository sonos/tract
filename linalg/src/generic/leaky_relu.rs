#![allow(clippy::excessive_precision)]
use crate::frame::element_wise::ElementWiseKer;
use tract_data::internal::*;
use tract_num_traits::Zero;

#[derive(Clone, Debug)]
pub struct SLeakyRelu4;

impl ElementWiseKer<f32, f32> for SLeakyRelu4 {
    fn name() -> &'static str {
        "generic"
    }

    fn alignment_bytes() -> usize {
        16
    }

    fn alignment_items() -> usize {
        4
    }

    fn nr() -> usize {
        4
    }

    fn run(x: &mut [f32], alpha: f32) {
        debug_assert!(x.len() % Self::nr() == 0);
        debug_assert!(x.as_ptr() as usize % Self::alignment_bytes() == 0);
        x.iter_mut().for_each(|px| *px = if *px < 0. { *px * alpha } else { *px });
    }
}

#[derive(Clone, Debug)]
pub struct HLeakyRelu8;

impl ElementWiseKer<f16, f16> for HLeakyRelu8 {
    fn name() -> &'static str {
        "generic"
    }

    fn alignment_bytes() -> usize {
        16
    }

    fn alignment_items() -> usize {
        4
    }

    fn nr() -> usize {
        8
    }

    fn run(x: &mut [f16], alpha: f16) {
        debug_assert!(x.len() % Self::nr() == 0);
        debug_assert!(x.as_ptr() as usize % Self::alignment_bytes() == 0);
        x.iter_mut().for_each(|px| *px = if *px < f16::zero() { *px * alpha } else { *px })
    }
}

#[cfg(test)]
#[macro_use]
pub mod s {
    leaky_relu_frame_tests!(true, f32, crate::generic::leaky_relu::SLeakyRelu4);
}

#[cfg(test)]
#[macro_use]
pub mod h {
    leaky_relu_frame_tests!(
        true,
        tract_data::internal::f16,
        crate::generic::leaky_relu::HLeakyRelu8
    );
}
