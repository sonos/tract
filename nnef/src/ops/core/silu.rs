use crate::internal::*;
use tract_core::ops::nn::silu::Silu;

pub fn register(registry: &mut Registry) {
    registry.register_unit_element_wise("tract_core_silu", &Silu {});
    // Backward compatibility alias
    registry.register_unit_element_wise("tract_transformers_silu", &Silu {});
}
