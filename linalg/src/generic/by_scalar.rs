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

by_scalar_impl_wrap!(
    f32,
    SAddByScalar4,
    4,
    4,
    f32,
    fn run(x: &mut [f32], s: f32) {
        debug_assert!(x.len() % Self::nr() == 0);
        debug_assert!(x.as_ptr() as usize % Self::alignment_bytes() == 0);
        x.iter_mut().for_each(|px| *px += s)
    }
);

by_scalar_impl_wrap!(
    f32,
    SSubByScalar4,
    4,
    4,
    f32,
    fn run(x: &mut [f32], s: f32) {
        debug_assert!(x.len() % Self::nr() == 0);
        debug_assert!(x.as_ptr() as usize % Self::alignment_bytes() == 0);
        x.iter_mut().for_each(|px| *px -= s)
    }
);

by_scalar_impl_wrap!(
    f32,
    SSubFByScalar4,
    4,
    4,
    f32,
    fn run(x: &mut [f32], s: f32) {
        debug_assert!(x.len() % Self::nr() == 0);
        debug_assert!(x.as_ptr() as usize % Self::alignment_bytes() == 0);
        x.iter_mut().for_each(|px| *px = s - *px)
    }
);

by_scalar_impl_wrap!(
    f32,
    SMinByScalar4,
    4,
    4,
    f32,
    fn run(x: &mut [f32], s: f32) {
        debug_assert!(x.len() % Self::nr() == 0);
        debug_assert!(x.as_ptr() as usize % Self::alignment_bytes() == 0);
        x.iter_mut().for_each(|px| *px = px.min(s))
    }
);

by_scalar_impl_wrap!(
    f32,
    SMaxByScalar4,
    4,
    4,
    f32,
    fn run(x: &mut [f32], s: f32) {
        debug_assert!(x.len() % Self::nr() == 0);
        debug_assert!(x.as_ptr() as usize % Self::alignment_bytes() == 0);
        x.iter_mut().for_each(|px| *px = px.max(s))
    }
);

#[cfg(test)]
#[macro_use]
pub mod mul_by_scalar_f32 {
    use super::*;
    by_scalar_frame_tests!(true, f32, SMulByScalar4, |a, b| a * b);
    by_scalar_frame_tests!(true, f32, SAddByScalar4, |a, b| a + b );
    by_scalar_frame_tests!(true, f32, SSubByScalar4, |a, b| a - b);
    by_scalar_frame_tests!(true, f32, SSubFByScalar4, |a, b| b - a);
    by_scalar_frame_tests!(true, f32, SMinByScalar4, |a, b| a.min(b));
    by_scalar_frame_tests!(true, f32, SMaxByScalar4, |a, b| a.max(b));
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

by_scalar_impl_wrap!(
    f16,
    HAddByScalar8,
    8,
    8,
    f16,
    fn run(x: &mut [f16], s: f16) {
        debug_assert!(x.len() % Self::nr() == 0);
        debug_assert!(x.as_ptr() as usize % Self::alignment_bytes() == 0);
        x.iter_mut().for_each(|px| *px += s)
    }
);

by_scalar_impl_wrap!(
    f16,
    HSubByScalar8,
    8,
    8,
    f16,
    fn run(x: &mut [f16], s: f16) {
        debug_assert!(x.len() % Self::nr() == 0);
        debug_assert!(x.as_ptr() as usize % Self::alignment_bytes() == 0);
        x.iter_mut().for_each(|px| *px -= s)
    }
);

by_scalar_impl_wrap!(
    f16,
    HSubFByScalar8,
    8,
    8,
    f16,
    fn run(x: &mut [f16], s: f16) {
        debug_assert!(x.len() % Self::nr() == 0);
        debug_assert!(x.as_ptr() as usize % Self::alignment_bytes() == 0);
        x.iter_mut().for_each(|px| *px = s - *px)
    }
);

by_scalar_impl_wrap!(
    f16,
    HMinByScalar8,
    8,
    8,
    f16,
    fn run(x: &mut [f16], s: f16) {
        debug_assert!(x.len() % Self::nr() == 0);
        debug_assert!(x.as_ptr() as usize % Self::alignment_bytes() == 0);
        x.iter_mut().for_each(|px| *px = px.min(s))
    }
);

by_scalar_impl_wrap!(
    f16,
    HMaxByScalar8,
    8,
    8,
    f16,
    fn run(x: &mut [f16], s: f16) {
        debug_assert!(x.len() % Self::nr() == 0);
        debug_assert!(x.as_ptr() as usize % Self::alignment_bytes() == 0);
        x.iter_mut().for_each(|px| *px = px.max(s))
    }
);

#[cfg(test)]
#[macro_use]
pub mod mul_by_scalar_f16 {
    use super::*;
    by_scalar_frame_tests!(true, f16, HMulByScalar8, |a, b| a * b);
    by_scalar_frame_tests!(true, f16, HAddByScalar8, |a, b| a + b);
    by_scalar_frame_tests!(true, f16, HSubByScalar8, |a, b| a - b);
    by_scalar_frame_tests!(true, f16, HSubFByScalar8, |a, b| b - a);
    by_scalar_frame_tests!(true, f16, HMinByScalar8, |a, b| a.min(b));
    by_scalar_frame_tests!(true, f16, HMaxByScalar8, |a, b| a.max(b));
}
