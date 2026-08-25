pub use tract_data::internal::f16;
routine_unicast_rust!(portable;
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
    },
    op(Mul)
);

routine_unicast_rust!(portable;
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
    },
    op(Add)
);

routine_unicast_rust!(portable;
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
    },
    op(Sub)
);

routine_unicast_rust!(portable;
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
    },
    op(SubF)
);

routine_unicast_rust!(portable;
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
    },
    op(Min)
);

routine_unicast_rust!(portable;
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
    },
    op(Max)
);
