//! Portable-reference activation descriptors. Always compiled; every cell's floor.

use crate::activation::{ActFactory, ActivationFn, ActivationImpl, Tier};
use crate::frame::element_wise::ElementWiseKer;
use crate::frame::reduce::{MapReduceKer, ReduceKer};
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
inventory::submit! {
    ActivationImpl {
        func: ActivationFn::Max, dt: DatumType::F32, target: "generic",
        feature: None, tier: Tier::Generic, isa_rank: 0, kernel: "SMax4",
        check: || true,
        factory: Some(ActFactory::F32Reduce(|| crate::generic::reduce::max::SMax4::red())),
    }
}
inventory::submit! {
    ActivationImpl {
        func: ActivationFn::Max, dt: DatumType::F16, target: "generic",
        feature: None, tier: Tier::Generic, isa_rank: 0, kernel: "HMax8",
        check: || true,
        factory: Some(ActFactory::F16Reduce(|| crate::generic::reduce::max::HMax8::red())),
    }
}
inventory::submit! {
    ActivationImpl {
        func: ActivationFn::Min, dt: DatumType::F32, target: "generic",
        feature: None, tier: Tier::Generic, isa_rank: 0, kernel: "SMin4",
        check: || true,
        factory: Some(ActFactory::F32Reduce(|| crate::generic::reduce::min::SMin4::red())),
    }
}
inventory::submit! {
    ActivationImpl {
        func: ActivationFn::Sum, dt: DatumType::F32, target: "generic",
        feature: None, tier: Tier::Generic, isa_rank: 0, kernel: "SSum4",
        check: || true,
        factory: Some(ActFactory::F32Reduce(|| crate::generic::reduce::sum::SSum4::red())),
    }
}
inventory::submit! {
    ActivationImpl {
        func: ActivationFn::Sum, dt: DatumType::F16, target: "generic",
        feature: None, tier: Tier::Generic, isa_rank: 0, kernel: "HSum8",
        check: || true,
        factory: Some(ActFactory::F16Reduce(|| crate::generic::reduce::sum::HSum8::red())),
    }
}
inventory::submit! {
    ActivationImpl {
        func: ActivationFn::Softmax, dt: DatumType::F32, target: "generic",
        feature: None, tier: Tier::Generic, isa_rank: 0, kernel: "SSoftMaxL2",
        check: || true,
        factory: Some(ActFactory::F32MapReduce(
            || crate::generic::reduce::softmax_l2::SSoftMaxL2::red(),
        )),
    }
}
inventory::submit! {
    ActivationImpl {
        func: ActivationFn::Softmax, dt: DatumType::F16, target: "generic",
        feature: None, tier: Tier::Generic, isa_rank: 0, kernel: "HSoftMaxL2",
        check: || true,
        factory: Some(ActFactory::F16MapReduce(
            || crate::generic::reduce::softmax_l2::HSoftMaxL2::red(),
        )),
    }
}
