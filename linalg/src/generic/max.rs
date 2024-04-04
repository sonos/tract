use tract_data::internal::f16;

use crate::frame::reduce::ReduceKer;

#[derive(Clone, Debug)]
pub struct SMax4;

impl ReduceKer<f32> for SMax4 {
    fn name() -> &'static str {
        "generic"
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
        a.max(b)
    }

    fn run(x: &[f32], _: ()) -> f32 {
        debug_assert!(x.len() % Self::nr() == 0);
        debug_assert!(x.as_ptr() as usize % Self::alignment_bytes() == 0);
        *x.iter().max_by(|a, b| a.total_cmp(b)).unwrap()
    }
}

#[derive(Clone, Debug)]
pub struct HMax8;

impl ReduceKer<f16> for HMax8 {
    fn name() -> &'static str {
        "generic"
    }

    fn alignment_items() -> usize {
        8
    }

    fn nr() -> usize {
        8
    }

    fn neutral() -> f16 {
        f16::MIN
    }

    fn reduce_two(a: f16, b: f16) -> f16 {
        a.max(b)
    }

    fn run(x: &[f16], _: ()) -> f16 {
        debug_assert!(x.len() % Self::nr() == 0);
        debug_assert!(x.as_ptr() as usize % Self::alignment_bytes() == 0);
        *x.iter().max_by(|a, b| a.total_cmp(b)).unwrap()
    }
}

#[cfg(test)]
#[macro_use]
pub mod s {
    max_frame_tests!(true, f32, crate::generic::max::SMax4);
}

#[cfg(test)]
#[macro_use]
pub mod h {
    use super::*;
    max_frame_tests!(true, f16, crate::generic::max::HMax8);
}
