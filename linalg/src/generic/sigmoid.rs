#![allow(clippy::excessive_precision)]
use crate::frame::element_wise::ElementWiseKer;
use tract_data::internal::*;

pub fn ssigmoid(x: f32) -> f32 {
    const LOW: f32 = -18.6;
    const HIGH: f32 = -LOW;

    const ALPHA_13: f32 = -4.433153405e-18;
    const ALPHA_11: f32 = 1.169974371e-14;
    const ALPHA_9: f32 = -1.875289645e-11;
    const ALPHA_7: f32 = 4.257889523e-8;
    const ALPHA_5: f32 = 0.00004811817576;
    const ALPHA_3: f32 = 0.008163842030;
    const ALPHA_1: f32 = 0.2499999971;
    const BETA_6: f32 = 3.922935744e-6;
    const BETA_4: f32 = 0.001524872358;
    const BETA_2: f32 = 0.1159886749;
    const BETA_0: f32 = 1.0;

    let x = x.clamp(LOW, HIGH);

    let x2 = x * x;

    let p = ALPHA_13;
    let p = x2 * p + ALPHA_11;
    let p = x2 * p + ALPHA_9;
    let p = x2 * p + ALPHA_7;
    let p = x2 * p + ALPHA_5;
    let p = x2 * p + ALPHA_3;
    let p = x2 * p + ALPHA_1;
    let p = p * x;

    let q = BETA_6;
    let q = x2 * q + BETA_4;
    let q = x2 * q + BETA_2;
    let q = x2 * q + BETA_0;

    p / q + 0.5
}

pub fn hsigmoid(x: f16) -> f16 {
    /*
     * (x (0.249895 + x^2 (0.00400222 - 0.0000124702 x^2)))
     * /
     * (1. + 0.098734 x^2)
     */

    const LOW: f16 = f16::from_f32_const(-6.92);
    const HIGH: f16 = f16::from_f32_const(6.92);

    const ALPHA_5: f16 = f16::from_f32_const(-0.0000124702);
    const ALPHA_3: f16 = f16::from_f32_const(0.00400222);
    const ALPHA_1: f16 = f16::from_f32_const(0.249895);

    const BETA_2: f16 = f16::from_f32_const(0.098734);
    const BETA_0: f16 = f16::from_f32_const(1.0);

    let x = x.clamp(LOW, HIGH);

    let x2 = x * x;

    let p = ALPHA_5;
    let p = x2 * p + ALPHA_3;
    let p = x2 * p + ALPHA_1;
    let p = p * x;

    let q = BETA_2;
    let q = x2 * q + BETA_0;

    p / q + f16::from_f32_const(0.5)
}

#[derive(Clone, Debug)]
pub struct SSigmoid4;

impl ElementWiseKer<f32> for SSigmoid4 {
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
        x.iter_mut().for_each(|px| *px = ssigmoid(*px))
    }
}

#[derive(Clone, Debug)]
pub struct HSigmoid8;

impl ElementWiseKer<f16> for HSigmoid8 {
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
        x.iter_mut().for_each(|px| *px = hsigmoid(*px))
    }
}

#[cfg(test)]
#[macro_use]
pub mod s {
    sigmoid_frame_tests!(true, f32, crate::generic::sigmoid::SSigmoid4);
}

#[cfg(test)]
#[macro_use]
pub mod h {
    sigmoid_frame_tests!(true, tract_data::internal::f16, crate::generic::sigmoid::HSigmoid8);
}
