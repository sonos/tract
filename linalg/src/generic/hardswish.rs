#![allow(clippy::excessive_precision)]
use crate::frame::element_wise::ElementWiseKer;
use tract_data::internal::*;
use tract_num_traits::Zero;

#[derive(Clone, Debug)]
pub struct SHardSwish4;

impl ElementWiseKer<f32> for SHardSwish4 {
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

    fn run(x: &mut [f32], _: ()) {
        debug_assert!(x.len() % Self::nr() == 0);
        debug_assert!(x.as_ptr() as usize % Self::alignment_bytes() == 0);
        const INV6: f32 = 1.0 / 6.0;
        x.iter_mut().for_each(|px| {
            let relu6 = ((*px + 3.0).min(6.0)).max(0.0);
            *px = *px * relu6 * INV6;
        });
    }
}

#[derive(Clone, Debug)]
pub struct HHardSwish8;

impl ElementWiseKer<f16> for HHardSwish8 {
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

    fn run(x: &mut [f16], _: ()) {
        debug_assert!(x.len() % Self::nr() == 0);
        debug_assert!(x.as_ptr() as usize % Self::alignment_bytes() == 0);
        let three = f16::from_f32(3.0);
        let six = f16::from_f32(6.0);
        let inv6 = f16::from_f32(1.0 / 6.0);
        x.iter_mut().for_each(|px| {
            let relu6 = ((*px + three).min(six)).max(f16::zero());
            *px = *px * relu6 * inv6;
        });
    }
}

#[cfg(test)]
#[macro_use]
pub mod s {
    hardswish_frame_tests!(true, f32, crate::generic::hardswish::SHardSwish4);
}

#[cfg(test)]
#[macro_use]
pub mod h {
    hardswish_frame_tests!(true, tract_data::internal::f16, crate::generic::hardswish::HHardSwish8);
}
