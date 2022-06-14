use crate::internal::*;
use tract_core::ops;

mod broadcast;
mod cast;
mod downsample;
mod gather;
mod one_hot;
mod qconv;
mod qmatmul;
mod range;
mod reduce;
mod scan;
mod scatter;
mod source;

pub fn register(registry: &mut Registry) {
    registry.register_unit_element_wise("tract_core_round_even", &ops::math::RoundHalfToEven {});

    registry.register_binary("tract_core_xor", &ops::logic::Xor {});

    registry.register_binary_with_flipped(
        "tract_shl",
        &ops::math::ShiftLeft,
        &ops::math::FlippedShiftLeft,
    );
    registry.register_binary_with_flipped(
        "tract_shr",
        &ops::math::ShiftRight,
        &ops::math::FlippedShiftRight,
    );
    broadcast::register(registry);
    cast::register(registry);
    downsample::register(registry);
    gather::register(registry);
    one_hot::register(registry);
    qconv::register(registry);
    qmatmul::register(registry);
    reduce::register(registry);
    scatter::register(registry);
    scan::register(registry);
    source::register(registry);
    range::register(registry);
}
