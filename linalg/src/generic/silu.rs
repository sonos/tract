#![allow(clippy::excessive_precision)]
use crate::generic::sigmoid::{LOW, ssigmoid};
use tract_data::internal::*;

// f32 SiLU, as `max(x, LOW) * ssigmoid(x)`.
//
// The factor is floored at [`LOW`] rather than left as `x`: past the clamp [`ssigmoid`]
// returns the constant `4.8e-7` instead of exactly 0, so an unfloored factor would take
// the negative tail to `-inf` instead of decaying to 0. Floored, `x < LOW` saturates at
// `LOW * ssigmoid(LOW)` ~= `-6.9e-6`, which the true SiLU approaches from below.
routine_ew_rust!(generic;
    f32,
    generic_silu_f32_4n,
    4,
    4,
    fn run(x: &mut [f32], _: ()) {
        debug_assert!(x.len() % Self::nr() == 0);
        debug_assert!(x.as_ptr() as usize % Self::alignment_bytes() == 0);
        x.iter_mut().for_each(|px| *px = px.max(LOW) * ssigmoid(*px));
    },
    func(Silu)
);

// f16 SiLU, evaluated on the f32 fit and narrowed, and floored the same way.
routine_ew_rust!(generic;
    f16,
    generic_silu_f16_8n,
    8,
    8,
    fn run(x: &mut [f16], _: ()) {
        debug_assert!(x.len() % Self::nr() == 0);
        debug_assert!(x.as_ptr() as usize % Self::alignment_bytes() == 0);
        x.iter_mut().for_each(|px| {
            let x_f32 = px.to_f32();
            *px = f16::from_f32(x_f32.max(LOW) * ssigmoid(x_f32));
        });
    },
    func(Silu)
);
