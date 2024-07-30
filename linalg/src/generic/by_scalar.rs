use tract_data::internal::f16;

use crate::element_wise::ElementWiseKer;

#[derive(Clone, Debug)]
pub struct SMulByScalar4;

impl ElementWiseKer<f32, f32> for SMulByScalar4 {
    fn name() -> &'static str {
        "generic"
    }

    fn alignment_items() -> usize {
        4
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
pub mod mul_by_scalar_f32 {
    mul_by_scalar_frame_tests!(true, f32, crate::generic::by_scalar::SMulByScalar4);
}

#[derive(Clone, Debug)]
pub struct HMulByScalar8;

impl ElementWiseKer<f16, f16> for HMulByScalar8 {
    fn name() -> &'static str {
        "generic"
    }

    fn alignment_items() -> usize {
        8
    }

    fn nr() -> usize {
        8
    }

    fn run(x: &mut [f16], s: f16) {
        debug_assert!(x.len() % Self::nr() == 0);
        debug_assert!(x.as_ptr() as usize % Self::alignment_bytes() == 0);
        x.iter_mut().for_each(|px| *px *= s)
    }
}

#[cfg(test)]
#[macro_use]
pub mod mul_by_scalar_f16 {
    use super::*;
    mul_by_scalar_frame_tests!(true, f16, crate::generic::by_scalar::HMulByScalar8);
}
