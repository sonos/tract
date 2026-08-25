use tract_data::internal::f16;

routine_by_scalar_rust!(generic;
    f32,
    SMulByScalar4,
    4,
    4,
    fn run(x: &mut [f32], s: f32) {
        debug_assert!(x.len() % Self::nr() == 0);
        debug_assert!(x.as_ptr() as usize % Self::alignment_bytes() == 0);
        x.iter_mut().for_each(|px| *px *= s)
    },
    op(Mul),
    bin,
    param(MulByScalar)
);

routine_by_scalar_rust!(generic;
    f32,
    SAddByScalar4,
    4,
    4,
    fn run(x: &mut [f32], s: f32) {
        debug_assert!(x.len() % Self::nr() == 0);
        debug_assert!(x.as_ptr() as usize % Self::alignment_bytes() == 0);
        x.iter_mut().for_each(|px| *px += s)
    },
    op(Add),
    bin
);

routine_by_scalar_rust!(generic;
    f32,
    SSubByScalar4,
    4,
    4,
    fn run(x: &mut [f32], s: f32) {
        debug_assert!(x.len() % Self::nr() == 0);
        debug_assert!(x.as_ptr() as usize % Self::alignment_bytes() == 0);
        x.iter_mut().for_each(|px| *px -= s)
    },
    op(Sub),
    bin
);

routine_by_scalar_rust!(generic;
    f32,
    SSubFByScalar4,
    4,
    4,
    fn run(x: &mut [f32], s: f32) {
        debug_assert!(x.len() % Self::nr() == 0);
        debug_assert!(x.as_ptr() as usize % Self::alignment_bytes() == 0);
        x.iter_mut().for_each(|px| *px = s - *px)
    },
    op(SubF),
    bin
);

routine_by_scalar_rust!(generic;
    f32,
    SMinByScalar4,
    4,
    4,
    fn run(x: &mut [f32], s: f32) {
        debug_assert!(x.len() % Self::nr() == 0);
        debug_assert!(x.as_ptr() as usize % Self::alignment_bytes() == 0);
        x.iter_mut().for_each(|px| *px = px.min(s))
    },
    op(Min),
    bin
);

routine_by_scalar_rust!(generic;
    f32,
    SMaxByScalar4,
    4,
    4,
    fn run(x: &mut [f32], s: f32) {
        debug_assert!(x.len() % Self::nr() == 0);
        debug_assert!(x.as_ptr() as usize % Self::alignment_bytes() == 0);
        x.iter_mut().for_each(|px| *px = px.max(s))
    },
    op(Max),
    bin
);

routine_by_scalar_rust!(generic;
    f16,
    HMulByScalar8,
    8,
    8,
    fn run(x: &mut [f16], s: f16) {
        debug_assert!(x.len() % Self::nr() == 0);
        debug_assert!(x.as_ptr() as usize % Self::alignment_bytes() == 0);
        x.iter_mut().for_each(|px| *px *= s)
    },
    op(Mul),
    param(MulByScalar)
);
