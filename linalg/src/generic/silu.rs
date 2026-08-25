#![allow(clippy::excessive_precision)]
use crate::frame::element_wise::ElementWiseKer;
use crate::generic::sigmoid::{LOW, ssigmoid};
use tract_data::internal::*;

/// f32 SiLU, as `max(x, LOW) * ssigmoid(x)`.
///
/// The factor is floored at [`LOW`] rather than left as `x`: past the clamp [`ssigmoid`]
/// returns the constant `4.8e-7` instead of exactly 0, so an unfloored factor would take
/// the negative tail to `-inf` instead of decaying to 0. Floored, `x < LOW` saturates at
/// `LOW * ssigmoid(LOW)` ~= `-6.9e-6`, which the true SiLU approaches from below.
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
        x.iter_mut().for_each(|px| *px = px.max(LOW) * ssigmoid(*px));
    }
}

/// f16 SiLU, evaluated on the f32 fit and narrowed, and floored like [`SSiLU4`].
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
            *px = f16::from_f32(x_f32.max(LOW) * ssigmoid(x_f32));
        });
    }
}

submit_routine!(F32, Silu, SSiLU4);
submit_routine!(F16, Silu, HSiLU8);

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
