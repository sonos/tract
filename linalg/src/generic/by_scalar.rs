use crate::element_wise::ElementWiseKer;


#[derive(Clone, Debug)]
pub struct SMulByScalar4;

impl ElementWiseKer<f32, f32> for SMulByScalar4 {
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

    fn run(x: &mut [f32], s: f32) {
        debug_assert!(x.len() % Self::nr() == 0);
        debug_assert!(x.as_ptr() as usize % Self::alignment_bytes() == 0);
        x.iter_mut().for_each(|px| *px *= s)
    }
}

#[cfg(test)]
#[macro_use]
pub mod s {
    mul_by_scalar_frame_tests!(true, f32, crate::generic::by_scalar::SMulByScalar4);
}
