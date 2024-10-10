pub use tract_data::internal::f16;
unicast_impl_wrap!(
    f32,
    SUnicastMul4,
    4,
    4,
    fn run(a: &mut [f32], b: &[f32]) {
        debug_assert!(a.len() == b.len());
        debug_assert!(a.len() % Self::nr() == 0);
        debug_assert!(a.as_ptr() as usize % Self::alignment_bytes() == 0);
        debug_assert!(b.as_ptr() as usize % Self::alignment_bytes() == 0);
        a.iter_mut().zip(b.iter()).for_each(|(a, b)| *a *= b)
    }
);

unicast_impl_wrap!(
    f16,
    HUnicastMul8,
    8,
    8,
    fn run(a: &mut [f16], b: &[f16]) {
        debug_assert!(a.len() == b.len());
        debug_assert!(a.len() % Self::nr() == 0);
        debug_assert!(a.as_ptr() as usize % Self::alignment_bytes() == 0);
        debug_assert!(b.as_ptr() as usize % Self::alignment_bytes() == 0);
        a.iter_mut().zip(b.iter()).for_each(|(a, b)| *a *= b)
    }
);

unicast_impl_wrap!(
    f32,
    SUnicastAdd4,
    4,
    4,
    fn run(a: &mut [f32], b: &[f32]) {
        debug_assert!(a.len() == b.len());
        debug_assert!(a.len() % Self::nr() == 0);
        debug_assert!(a.as_ptr() as usize % Self::alignment_bytes() == 0);
        debug_assert!(b.as_ptr() as usize % Self::alignment_bytes() == 0);
        a.iter_mut().zip(b.iter()).for_each(|(a, b)| *a += b)
    }
);

unicast_impl_wrap!(
    f16,
    HUnicastAdd8,
    8,
    8,
    fn run(a: &mut [f16], b: &[f16]) {
        debug_assert!(a.len() == b.len());
        debug_assert!(a.len() % Self::nr() == 0);
        debug_assert!(a.as_ptr() as usize % Self::alignment_bytes() == 0);
        debug_assert!(b.as_ptr() as usize % Self::alignment_bytes() == 0);
        a.iter_mut().zip(b.iter()).for_each(|(a, b)| *a += b)
    }
);

unicast_impl_wrap!(
    f32,
    SUnicastSub4,
    4,
    4,
    fn run(a: &mut [f32], b: &[f32]) {
        debug_assert!(a.len() == b.len());
        debug_assert!(a.len() % Self::nr() == 0);
        debug_assert!(a.as_ptr() as usize % Self::alignment_bytes() == 0);
        debug_assert!(b.as_ptr() as usize % Self::alignment_bytes() == 0);
        a.iter_mut().zip(b.iter()).for_each(|(a, b)| *a -= b)
    }
);

unicast_impl_wrap!(
    f16,
    HUnicastSub8,
    8,
    8,
    fn run(a: &mut [f16], b: &[f16]) {
        debug_assert!(a.len() == b.len());
        debug_assert!(a.len() % Self::nr() == 0);
        debug_assert!(a.as_ptr() as usize % Self::alignment_bytes() == 0);
        debug_assert!(b.as_ptr() as usize % Self::alignment_bytes() == 0);
        a.iter_mut().zip(b.iter()).for_each(|(a, b)| *a -= b)
    }
);

unicast_impl_wrap!(
    f32,
    SUnicastSubF4,
    4,
    4,
    fn run(a: &mut [f32], b: &[f32]) {
        debug_assert!(a.len() == b.len());
        debug_assert!(a.len() % Self::nr() == 0);
        debug_assert!(a.as_ptr() as usize % Self::alignment_bytes() == 0);
        debug_assert!(b.as_ptr() as usize % Self::alignment_bytes() == 0);
        a.iter_mut().zip(b.iter()).for_each(|(a, b)| *a = *b - *a)
    }
);

unicast_impl_wrap!(
    f16,
    HUnicastSubF8,
    8,
    8,
    fn run(a: &mut [f16], b: &[f16]) {
        debug_assert!(a.len() == b.len());
        debug_assert!(a.len() % Self::nr() == 0);
        debug_assert!(a.as_ptr() as usize % Self::alignment_bytes() == 0);
        debug_assert!(b.as_ptr() as usize % Self::alignment_bytes() == 0);
        a.iter_mut().zip(b.iter()).for_each(|(a, b)| *a = *b - *a)
    }
);

unicast_impl_wrap!(
    f32,
    SUnicastMin4,
    4,
    4,
    fn run(a: &mut [f32], b: &[f32]) {
        debug_assert!(a.len() == b.len());
        debug_assert!(a.len() % Self::nr() == 0);
        debug_assert!(a.as_ptr() as usize % Self::alignment_bytes() == 0);
        debug_assert!(b.as_ptr() as usize % Self::alignment_bytes() == 0);
        a.iter_mut().zip(b.iter()).for_each(|(a, b)| *a = a.min(*b))
    }
);

unicast_impl_wrap!(
    f16,
    HUnicastMin8,
    8,
    8,
    fn run(a: &mut [f16], b: &[f16]) {
        debug_assert!(a.len() == b.len());
        debug_assert!(a.len() % Self::nr() == 0);
        debug_assert!(a.as_ptr() as usize % Self::alignment_bytes() == 0);
        debug_assert!(b.as_ptr() as usize % Self::alignment_bytes() == 0);
        a.iter_mut().zip(b.iter()).for_each(|(a, b)| *a = a.min(*b))
    }
);

unicast_impl_wrap!(
    f32,
    SUnicastMax4,
    4,
    4,
    fn run(a: &mut [f32], b: &[f32]) {
        debug_assert!(a.len() == b.len());
        debug_assert!(a.len() % Self::nr() == 0);
        debug_assert!(a.as_ptr() as usize % Self::alignment_bytes() == 0);
        debug_assert!(b.as_ptr() as usize % Self::alignment_bytes() == 0);
        a.iter_mut().zip(b.iter()).for_each(|(a, b)| *a = a.max(*b))
    }
);

unicast_impl_wrap!(
    f16,
    HUnicastMax8,
    8,
    8,
    fn run(a: &mut [f16], b: &[f16]) {
        debug_assert!(a.len() == b.len());
        debug_assert!(a.len() % Self::nr() == 0);
        debug_assert!(a.as_ptr() as usize % Self::alignment_bytes() == 0);
        debug_assert!(b.as_ptr() as usize % Self::alignment_bytes() == 0);
        a.iter_mut().zip(b.iter()).for_each(|(a, b)| *a = a.max(*b))
    }
);

#[cfg(test)]
#[macro_use]
pub mod s {
    use super::*;
    use proptest::strategy::Strategy;
    crate::unicast_frame_tests!(true, f32, SUnicastMul4, |a, b| a * b);
    crate::unicast_frame_tests!(true, f32, SUnicastAdd4, |a, b| a + b);
    crate::unicast_frame_tests!(true, f32, SUnicastSub4, |a, b| a - b);
    crate::unicast_frame_tests!(true, f32, SUnicastSubF4, |a, b| b - a);
    crate::unicast_frame_tests!(true, f32, SUnicastMin4, |a, b| a.min(b));
    crate::unicast_frame_tests!(true, f32, SUnicastMax4, |a, b| a.max(b));
}

#[cfg(test)]
#[macro_use]
pub mod h {
    use super::*;
    use proptest::strategy::Strategy;
    crate::unicast_frame_tests!(true, f16, HUnicastMul8, |a, b| a * b);
    crate::unicast_frame_tests!(true, f16, HUnicastAdd8, |a, b| a + b);
    crate::unicast_frame_tests!(true, f16, HUnicastSub8, |a, b| a - b);
    crate::unicast_frame_tests!(true, f16, HUnicastSubF8, |a, b| b - a);
    crate::unicast_frame_tests!(true, f16, HUnicastMin8, |a, b| a.min(b));
    crate::unicast_frame_tests!(true, f16, HUnicastMax8, |a, b| a.max(b));
}
