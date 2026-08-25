#![allow(clippy::excessive_precision)]
use tract_data::internal::*;
use tract_num_traits::Zero;

routine_ew_rust!(generic;
    f32,
    generic_leaky_relu_f32_4n,
    4,
    4,
    fn run(x: &mut [f32], alpha: f32) {
        debug_assert!(x.len() % Self::nr() == 0);
        debug_assert!(x.as_ptr() as usize % Self::alignment_bytes() == 0);
        x.iter_mut().for_each(|px| *px = if *px < 0. { *px * alpha } else { *px });
    },
    func(LeakyRelu),
    param
);

routine_ew_rust!(generic;
    f16,
    generic_leaky_relu_f16_8n,
    8,
    8,
    fn run(x: &mut [f16], alpha: f16) {
        debug_assert!(x.len() % Self::nr() == 0);
        debug_assert!(x.as_ptr() as usize % Self::alignment_bytes() == 0);
        x.iter_mut().for_each(|px| *px = if *px < f16::zero() { *px * alpha } else { *px })
    },
    func(LeakyRelu),
    param
);
