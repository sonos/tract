use crate::frame::element_wise::ElementWiseKer;

/*
 * (x (0.25 + x^2 (0.00809368 + x^2 (0.0000461542 + x^2 (3.73971*10^-8 + x^2 (-1.33972*10^-11 + 5.05941*10^-15 x^2))))))
 * --------------------------------------------------------------------------------------------------------------------- + 0.5
 *              (1. + x^2 (0.115708 + x^2 (0.00149364 + 3.63597*10^-6 x^2)))
 */

const LOW: f32 = -17.0;
const HIGH: f32 = 17.0;

const ALPHA_11: f32 = 5.05941e-15;
const ALPHA_9: f32 = -1.33972e-11;
const ALPHA_7: f32 = 3.73971e-8;
const ALPHA_5: f32 = 0.0000461542;
const ALPHA_3: f32 = 0.00809368;
const ALPHA_1: f32 = 0.25;
const BETA_6: f32 = 3.63597e-6;
const BETA_4: f32 = 0.00149364;
const BETA_2: f32 = 0.115708;
const BETA_0: f32 = 1.0;

pub fn ssigmoid(x: f32) -> f32 {
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

    p / q + 0.5
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

    fn run(x: &mut [f32]) {
        debug_assert!(x.len() % Self::nr() == 0);
        debug_assert!(x.as_ptr() as usize % Self::alignment_bytes() == 0);
        x.iter_mut().for_each(|px| *px = ssigmoid(*px))
    }
}

#[cfg(test)]
#[macro_use]
pub mod test {
    sigmoid_frame_tests!(true, crate::generic::sigmoid::SSigmoid4);
}
