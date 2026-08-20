#![allow(clippy::excessive_precision)]
use crate::frame::element_wise::ElementWiseKer;
use crate::generic::sigmoid::ssigmoid;
use tract_data::internal::*;

#[derive(Clone, Debug)]
pub struct SSiLU4;

impl ElementWiseKer<f32> for SSiLU4 {
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
        x.iter_mut().for_each(|px| *px *= ssigmoid(*px));
    }
}

#[derive(Clone, Debug)]
pub struct HSiLU8;

impl ElementWiseKer<f16> for HSiLU8 {
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
        x.iter_mut().for_each(|px| {
            let x_f32 = px.to_f32();
            *px = f16::from_f32(x_f32 * ssigmoid(x_f32));
        });
    }
}

#[cfg(test)]
#[macro_use]
pub mod s {
    silu_frame_tests!(true, f32, crate::generic::silu::SSiLU4);
}

#[cfg(test)]
#[macro_use]
pub mod h {
    silu_frame_tests!(true, tract_data::internal::f16, crate::generic::silu::HSiLU8);
}
