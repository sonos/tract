#![allow(clippy::excessive_precision)]
use crate::frame::element_wise::ElementWiseKer;
use tract_data::internal::*;

// Tanh-form GELU approximation matching tract's GeluApproximate (pow=3, the
// canonical Hendrycks-Gimpel/Open-AI form):
//
//     gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//
// The fast variant (pow=2) is not exposed here; the graph op falls back to
// scalar when fast_impl=true.

const SQRT_2_OVER_PI: f32 = 0.7978845608028654;
const COEF: f32 = 0.044715;

#[derive(Clone, Debug)]
pub struct SGelu4;

impl ElementWiseKer<f32> for SGelu4 {
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
        x.iter_mut().for_each(|px| {
            let v = *px;
            let inner = SQRT_2_OVER_PI * (v + COEF * v * v * v);
            *px = 0.5 * v * (1.0 + inner.tanh());
        });
    }
}

#[derive(Clone, Debug)]
pub struct HGelu8;

impl ElementWiseKer<f16> for HGelu8 {
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
            let v = px.to_f32();
            let inner = SQRT_2_OVER_PI * (v + COEF * v * v * v);
            *px = f16::from_f32(0.5 * v * (1.0 + inner.tanh()));
        });
    }
}

#[cfg(test)]
#[macro_use]
pub mod s {
    gelu_frame_tests!(true, f32, crate::generic::gelu::SGelu4);
}

#[cfg(test)]
#[macro_use]
pub mod h {
    gelu_frame_tests!(true, tract_data::internal::f16, crate::generic::gelu::HGelu8);
}
