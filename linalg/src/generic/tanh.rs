#![allow(clippy::excessive_precision)]
use crate::frame::element_wise::ElementWiseKer;
use tract_data::internal::*;

pub fn stanh(x: f32) -> f32 {
    const LOW: f32 = -8.9;
    const HIGH: f32 = 8.9;

    const ALPHA_13: f32 = -8.488492677e-14;
    const ALPHA_11: f32 = 5.277853000e-11;
    const ALPHA_9: f32 = -2.022500419e-8;
    const ALPHA_7: f32 = 0.00001115424833;
    const ALPHA_5: f32 = 0.003103950131;
    const ALPHA_3: f32 = 0.1308400453;
    const ALPHA_1: f32 = 0.9999999934;

    const BETA_6: f32 = 0.0002546136580;
    const BETA_4: f32 = 0.02449515379;
    const BETA_2: f32 = 0.4641733162;
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

    p / q
}

pub fn htanh(x: f16) -> f16 {
    const LOW: f16 = f16::from_f32_const(-3.84);
    const HIGH: f16 = f16::from_f32_const(3.84);

    const ALPHA_3: f16 = f16::from_f32_const(0.082654955);
    const ALPHA_1: f16 = f16::from_f32_const(0.99963124);

    const BETA_4: f16 = f16::from_f32_const(0.0065383179);
    const BETA_2: f16 = f16::from_f32_const(0.41401828);
    const BETA_0: f16 = f16::from_f32_const(1.0);

    let x = x.clamp(LOW, HIGH);

    let x2 = x * x;

    let p = ALPHA_3;
    let p = x2 * p + ALPHA_1;
    let p = p * x;

    let q = BETA_4;
    let q = x2 * q + BETA_2;
    let q = x2 * q + BETA_0;

    p / q
}

#[derive(Clone, Debug)]
pub struct STanh4;

impl ElementWiseKer<f32> for STanh4 {
    fn name() -> &'static str {
        "generic"
    }

    fn alignment_items() -> usize {
        16
    }

    fn alignment_bytes() -> usize {
        16
    }

    fn nr() -> usize {
        4
    }

    fn run(x: &mut [f32], _: ()) {
        debug_assert!(x.len() % Self::nr() == 0);
        debug_assert!(x.as_ptr() as usize % Self::alignment_bytes() == 0);
        x.iter_mut().for_each(|px| *px = stanh(*px))
    }
}

#[cfg(test)]
#[macro_use]
pub mod s {
    tanh_frame_tests!(true, f32, crate::generic::tanh::STanh4);
}

#[derive(Clone, Debug)]
pub struct HTanh8;

impl ElementWiseKer<f16> for HTanh8 {
    fn name() -> &'static str {
        "generic"
    }

    fn alignment_items() -> usize {
        16
    }

    fn alignment_bytes() -> usize {
        16
    }

    fn nr() -> usize {
        8
    }

    fn run(x: &mut [f16], _: ()) {
        debug_assert!(x.len() % Self::nr() == 0);
        debug_assert!(x.as_ptr() as usize % Self::alignment_bytes() == 0);
        x.iter_mut().for_each(|px| *px = htanh(*px))
    }
}

#[cfg(test)]
#[macro_use]
pub mod h {
    tanh_frame_tests!(true, tract_data::internal::f16, crate::generic::tanh::HTanh8);
}
