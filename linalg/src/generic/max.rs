use crate::frame::reduce::ReduceKer;

#[derive(Clone, Debug)]
pub struct SMax4;

impl ReduceKer<f32> for SMax4 {
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
        a.max(b)
    }

    fn run(x: &[f32], _: ()) -> f32 {
        debug_assert!(x.len() % Self::nr() == 0);
        debug_assert!(x.as_ptr() as usize % Self::alignment_bytes() == 0);
        *x.iter().max_by(|a, b| a.total_cmp(b)).unwrap()
    }
}

#[cfg(test)]
#[macro_use]
pub mod tmax {
    max_frame_tests!(true, f32, crate::generic::max::SMax4);
}
