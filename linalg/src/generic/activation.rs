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
inventory::submit! {
    ActivationImpl {
        func: ActivationFn::Tanh, dt: DatumType::F32, target: "generic",
        feature: None, tier: Tier::Generic, isa_rank: 0, kernel: "STanh4",
        check: || true,
        factory: Some(ActFactory::F32(|| crate::generic::STanh4::ew())),
    }
}
inventory::submit! {
    ActivationImpl {
        func: ActivationFn::Tanh, dt: DatumType::F16, target: "generic",
        feature: None, tier: Tier::Generic, isa_rank: 0, kernel: "HTanh8",
        check: || true,
        factory: Some(ActFactory::F16(|| crate::generic::HTanh8::ew())),
    }
}
inventory::submit! {
    ActivationImpl {
        func: ActivationFn::Erf, dt: DatumType::F32, target: "generic",
        feature: None, tier: Tier::Generic, isa_rank: 0, kernel: "SErf4",
        check: || true,
        factory: Some(ActFactory::F32(|| crate::generic::SErf4::ew())),
    }
}
inventory::submit! {
    ActivationImpl {
        func: ActivationFn::HardSwish, dt: DatumType::F32, target: "generic",
        feature: None, tier: Tier::Generic, isa_rank: 0, kernel: "SHardSwish4",
        check: || true,
        factory: Some(ActFactory::F32(|| crate::generic::SHardSwish4::ew())),
    }
}
inventory::submit! {
    ActivationImpl {
        func: ActivationFn::HardSwish, dt: DatumType::F16, target: "generic",
        feature: None, tier: Tier::Generic, isa_rank: 0, kernel: "HHardSwish8",
        check: || true,
        factory: Some(ActFactory::F16(|| crate::generic::HHardSwish8::ew())),
    }
}
inventory::submit! {
    ActivationImpl {
        func: ActivationFn::Gelu, dt: DatumType::F32, target: "generic",
        feature: None, tier: Tier::Generic, isa_rank: 0, kernel: "SGelu4",
        check: || true,
        factory: Some(ActFactory::F32(|| crate::generic::SGelu4::ew())),
    }
}
inventory::submit! {
    ActivationImpl {
        func: ActivationFn::Gelu, dt: DatumType::F16, target: "generic",
        feature: None, tier: Tier::Generic, isa_rank: 0, kernel: "HGelu8",
        check: || true,
        factory: Some(ActFactory::F16(|| crate::generic::HGelu8::ew())),
    }
}
inventory::submit! {
    ActivationImpl {
        func: ActivationFn::LeakyRelu, dt: DatumType::F32, target: "generic",
        feature: None, tier: Tier::Generic, isa_rank: 0, kernel: "SLeakyRelu4",
        check: || true,
        factory: Some(ActFactory::F32Param(|| crate::generic::SLeakyRelu4::ew())),
    }
}
inventory::submit! {
    ActivationImpl {
        func: ActivationFn::LeakyRelu, dt: DatumType::F16, target: "generic",
        feature: None, tier: Tier::Generic, isa_rank: 0, kernel: "HLeakyRelu8",
        check: || true,
        factory: Some(ActFactory::F16Param(|| crate::generic::HLeakyRelu8::ew())),
    }
}
inventory::submit! {
    ActivationImpl {
        func: ActivationFn::MulByScalar, dt: DatumType::F32, target: "generic",
        feature: None, tier: Tier::Generic, isa_rank: 0, kernel: "SMulByScalar4",
        check: || true,
        factory: Some(ActFactory::F32Param(|| crate::generic::SMulByScalar4::ew())),
    }
}
inventory::submit! {
    ActivationImpl {
        func: ActivationFn::MulByScalar, dt: DatumType::F16, target: "generic",
        feature: None, tier: Tier::Generic, isa_rank: 0, kernel: "HMulByScalar8",
        check: || true,
        factory: Some(ActFactory::F16Param(|| crate::generic::HMulByScalar8::ew())),
    }
}
