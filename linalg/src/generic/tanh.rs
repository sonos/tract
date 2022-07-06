use crate::frame::element_wise::ElementWiseKer;

/* (x (1. + x^2 (0.129499 + x^2 (0.00295387 + x^2 (9.57366*10^-6 + x^2 (-1.37187*10^-8 + 2.07234*10^-11 x^2))))))
 * ---------------------------------------------------------------------------------------------------------------
 *          (1. + x^2 (0.462832 + x^2 (0.0238983 + 0.000232702 x^2)))
 */

const LOW: f32 = -8.5;
const HIGH: f32 = 8.5;

const ALPHA_11: f32 = 2.07234e-11;
const ALPHA_9: f32 = -1.37187e-8;
const ALPHA_7: f32 = 9.57366e-6;
const ALPHA_5: f32 = 0.00295387;
const ALPHA_3: f32 = 0.129499;
const ALPHA_1: f32 = 1.0;
const BETA_6: f32 = 0.000232702;
const BETA_4: f32 = 0.0238983;
const BETA_2: f32 = 0.462832;
const BETA_0: f32 = 1.0;

pub fn stanh(x: f32) -> f32 {
    let x = x.max(LOW).min(HIGH);

    let x2 = x * x;

    let p = ALPHA_11;
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

    fn run(x: &mut [f32]) {
        debug_assert!(x.len() % Self::nr() == 0);
        debug_assert!(x.as_ptr() as usize % Self::alignment_bytes() == 0);
        x.iter_mut().for_each(|px| *px = stanh(*px))
    }
}

#[cfg(test)]
#[macro_use]
pub mod test {
    tanh_frame_tests!(true, crate::generic::tanh::STanh4);
}
