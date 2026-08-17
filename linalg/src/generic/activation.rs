//! Portable-reference activation descriptors. Always compiled; every cell's floor.

use crate::activation::{ActFactory, ActivationFn, ActivationImpl, Tier};
use crate::frame::element_wise::ElementWiseKer;
use tract_data::prelude::DatumType;

inventory::submit! {
    ActivationImpl {
        func: ActivationFn::Sigmoid, dt: DatumType::F32, target: "generic",
        feature: None, tier: Tier::Generic, isa_rank: 0, kernel: "SSigmoid4",
        check: || true,
        factory: Some(ActFactory::F32(|| crate::generic::SSigmoid4::ew())),
    }
}
inventory::submit! {
    ActivationImpl {
        func: ActivationFn::Sigmoid, dt: DatumType::F16, target: "generic",
        feature: None, tier: Tier::Generic, isa_rank: 0, kernel: "HSigmoid8",
        check: || true,
        factory: Some(ActFactory::F16(|| crate::generic::HSigmoid8::ew())),
    }
}
inventory::submit! {
    ActivationImpl {
        func: ActivationFn::Silu, dt: DatumType::F32, target: "generic",
        feature: None, tier: Tier::Generic, isa_rank: 0, kernel: "SSiLU4",
        check: || true,
        factory: Some(ActFactory::F32(|| crate::generic::SSiLU4::ew())),
    }
}
inventory::submit! {
    ActivationImpl {
        func: ActivationFn::Silu, dt: DatumType::F16, target: "generic",
        feature: None, tier: Tier::Generic, isa_rank: 0, kernel: "HSiLU8",
        check: || true,
        factory: Some(ActFactory::F16(|| crate::generic::HSiLU8::ew())),
    }
}
