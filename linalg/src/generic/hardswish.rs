#![allow(clippy::excessive_precision)]
use tract_data::internal::*;
use tract_num_traits::Zero;

routine_ew_rust!(generic;
    f32,
    generic_hardswish_f32_4n,
    4,
    4,
    fn run(x: &mut [f32], _: ()) {
        debug_assert!(x.len() % Self::nr() == 0);
        debug_assert!(x.as_ptr() as usize % Self::alignment_bytes() == 0);
        const INV6: f32 = 1.0 / 6.0;
        x.iter_mut().for_each(|px| {
            let relu6 = (*px + 3.0).clamp(0.0, 6.0);
            *px = *px * relu6 * INV6;
        });
    },
    func(Hardswish)
);

routine_ew_rust!(generic;
    f16,
    generic_hardswish_f16_8n,
    8,
    8,
    fn run(x: &mut [f16], _: ()) {
        debug_assert!(x.len() % Self::nr() == 0);
        debug_assert!(x.as_ptr() as usize % Self::alignment_bytes() == 0);
        let three = f16::from_f32(3.0);
        let six = f16::from_f32(6.0);
        let inv6 = f16::from_f32(1.0 / 6.0);
        x.iter_mut().for_each(|px| {
            let relu6 = ((*px + three).min(six)).max(f16::zero());
            *px = *px * relu6 * inv6;
        });
    },
    func(Hardswish)
);
