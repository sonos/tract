pub use tract_data::internal::f16;
binary_impl_wrap!(
    f32,
    SBinaryMul4,
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

binary_impl_wrap!(
    f16,
    HBinaryMul8,
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

#[cfg(test)]
#[macro_use]
pub mod s {
    crate::binary_mul_frame_tests!(true, f32, crate::generic::binary::SBinaryMul4);
}

#[cfg(test)]
#[macro_use]
pub mod h {
    use super::*;
    crate::binary_mul_frame_tests!(true, f16, crate::generic::binary::HBinaryMul8);
}
