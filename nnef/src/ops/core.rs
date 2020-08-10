use crate::internal::*;

pub fn register(registry: &mut Registry) {
    registry.register_element_wise("tract_core_tan", &tract_core::ops::math::Tan {});
    registry.register_element_wise("tract_core_acos", &tract_core::ops::math::Acos {});
    registry.register_element_wise("tract_core_asin", &tract_core::ops::math::Asin {});
    registry.register_element_wise("tract_core_atan", &tract_core::ops::math::Atan {});
    registry.register_element_wise("tract_core_acosh", &tract_core::ops::math::Acosh {});
    registry.register_element_wise("tract_core_asinh", &tract_core::ops::math::Asinh {});
    registry.register_element_wise("tract_core_atanh", &tract_core::ops::math::Atanh {});
}
