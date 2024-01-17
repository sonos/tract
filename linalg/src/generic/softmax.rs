use crate::frame::reduce::MapReduceKer;

#[derive(Clone, Debug)]
pub struct SSoftMaxL2;

impl MapReduceKer<f32, f32> for SSoftMaxL2 {
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

    fn neutral() -> f32 {
        f32::MIN
    }

    fn reduce_two(a: f32, b: f32) -> f32 {
        a + b
    }

    fn run(x: &mut [f32], max: f32) -> f32 {
        debug_assert!(x.len() % Self::nr() == 0);
        debug_assert!(x.as_ptr() as usize % Self::alignment_bytes() == 0);
        let mut sum = 0.;
        for x in x.iter_mut() {
            *x = (*x - max).exp();
            sum += *x;
        }
        sum
    }
}

#[cfg(test)]
#[macro_use]
pub mod s {
    softmax_l2_frame_tests!(true, f32, crate::generic::softmax::SSoftMaxL2);
}
