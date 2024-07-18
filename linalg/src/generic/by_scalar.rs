use tract_data::internal::f16;

by_scalar_impl_wrap!(
    f32,
    SMulByScalar4,
    4,
    4,
    f32,
    fn run(x: &mut [f32], s: f32) {
        debug_assert!(x.len() % Self::nr() == 0);
        debug_assert!(x.as_ptr() as usize % Self::alignment_bytes() == 0);
        x.iter_mut().for_each(|px| *px *= s)
    }
);

#[cfg(test)]
#[macro_use]
pub mod mul_by_scalar_f32 {
    mul_by_scalar_frame_tests!(true, f32, crate::generic::by_scalar::SMulByScalar4);
}

by_scalar_impl_wrap!(
    f16,
    HMulByScalar8,
    8,
    8,
    f16,
    fn run(x: &mut [f16], s: f16) {
        debug_assert!(x.len() % Self::nr() == 0);
        debug_assert!(x.as_ptr() as usize % Self::alignment_bytes() == 0);
        x.iter_mut().for_each(|px| *px *= s)
    }
);

#[cfg(test)]
#[macro_use]
pub mod mul_by_scalar_f16 {
    use super::*;
    mul_by_scalar_frame_tests!(true, f16, crate::generic::by_scalar::HMulByScalar8);
}
