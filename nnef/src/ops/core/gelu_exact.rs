use crate::internal::*;
use tract_core::ops::nn::gelu_exact::GeluExact;

pub fn register(registry: &mut Registry) {
    registry.register_unit_element_wise("tract_core_gelu_exact", &GeluExact {});
}
