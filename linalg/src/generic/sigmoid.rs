use crate::frame::sigmoid::SigmoidKer;

const LOW: f32 = -18.0;
const HIGH: f32 = 18.0;
const ALPHA_9: f32 = 4.37031012579801e-11;
const ALPHA_7: f32 = 1.15627324459942e-07;
const ALPHA_5: f32 = 6.08574864600143e-05;
const ALPHA_3: f32 = 8.51377133304701e-03;
const ALPHA_1: f32 = 2.48287947061529e-01;
const BETA_10: f32 = 6.10247389755681e-13;
const BETA_8: f32 = 5.76102136993427e-09;
const BETA_6: f32 = 6.29106785017040e-06;
const BETA_4: f32 = 1.70198817374094e-03;
const BETA_2: f32 = 1.16817656904453e-01;
const BETA_0: f32 = 9.93151921023180e-01;

pub fn ssigmoid(x: f32) -> f32 {
    let x = x.max(LOW).min(HIGH);

    let x2 = x * x;

    let p = ALPHA_9;
    let p = x2 * p + ALPHA_7;
    let p = x2 * p + ALPHA_5;
    let p = x2 * p + ALPHA_3;
    let p = x2 * p + ALPHA_1;
    let p = p * x;

    let q = BETA_10;
    let q = x2 * q + BETA_8;
    let q = x2 * q + BETA_6;
    let q = x2 * q + BETA_4;
    let q = x2 * q + BETA_2;
    let q = x2 * q + BETA_0;

    p / q + 0.5
}

#[derive(Clone, Debug)]
pub struct SSigmoid4;

impl SigmoidKer<f32> for SSigmoid4 {
    fn name() -> &'static str {
        "generic"
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
        x.iter_mut().for_each(|px| *px = ssigmoid(*px))
    }
}

#[cfg(test)]
#[macro_use]
pub mod test {
    sigmoid_frame_tests!(true, crate::generic::sigmoid::SSigmoid4);
}
