//! Portable-reference activation descriptors. Always compiled; every cell's floor.

use crate::BinOp;
use crate::frame::by_scalar::ByScalarKer;
use crate::frame::element_wise::ElementWiseKer;
use crate::frame::reduce::{MapReduceKer, ReduceKer};
use crate::frame::unicast::UnicastKer;
use crate::routines::{Routine, RoutineFactory, RoutineImpl, Tier};
use tract_data::prelude::DatumType;

inventory::submit! {
    RoutineImpl {
        func: Routine::Sigmoid, dt: DatumType::F32, target: "generic",
        feature: None, tier: Tier::Generic, isa_rank: 0, kernel: "SSigmoid4",
        check: || true,
        factory: Some(RoutineFactory::F32(|| crate::generic::SSigmoid4::ew())),
    }
}
inventory::submit! {
    RoutineImpl {
        func: Routine::Sigmoid, dt: DatumType::F16, target: "generic",
        feature: None, tier: Tier::Generic, isa_rank: 0, kernel: "HSigmoid8",
        check: || true,
        factory: Some(RoutineFactory::F16(|| crate::generic::HSigmoid8::ew())),
    }
}
inventory::submit! {
    RoutineImpl {
        func: Routine::Silu, dt: DatumType::F32, target: "generic",
        feature: None, tier: Tier::Generic, isa_rank: 0, kernel: "SSiLU4",
        check: || true,
        factory: Some(RoutineFactory::F32(|| crate::generic::SSiLU4::ew())),
    }
}
inventory::submit! {
    RoutineImpl {
        func: Routine::Silu, dt: DatumType::F16, target: "generic",
        feature: None, tier: Tier::Generic, isa_rank: 0, kernel: "HSiLU8",
        check: || true,
        factory: Some(RoutineFactory::F16(|| crate::generic::HSiLU8::ew())),
    }
}
inventory::submit! {
    RoutineImpl {
        func: Routine::Tanh, dt: DatumType::F32, target: "generic",
        feature: None, tier: Tier::Generic, isa_rank: 0, kernel: "STanh4",
        check: || true,
        factory: Some(RoutineFactory::F32(|| crate::generic::STanh4::ew())),
    }
}
inventory::submit! {
    RoutineImpl {
        func: Routine::Tanh, dt: DatumType::F16, target: "generic",
        feature: None, tier: Tier::Generic, isa_rank: 0, kernel: "HTanh8",
        check: || true,
        factory: Some(RoutineFactory::F16(|| crate::generic::HTanh8::ew())),
    }
}
inventory::submit! {
    RoutineImpl {
        func: Routine::Erf, dt: DatumType::F32, target: "generic",
        feature: None, tier: Tier::Generic, isa_rank: 0, kernel: "SErf4",
        check: || true,
        factory: Some(RoutineFactory::F32(|| crate::generic::SErf4::ew())),
    }
}
inventory::submit! {
    RoutineImpl {
        func: Routine::HardSwish, dt: DatumType::F32, target: "generic",
        feature: None, tier: Tier::Generic, isa_rank: 0, kernel: "SHardSwish4",
        check: || true,
        factory: Some(RoutineFactory::F32(|| crate::generic::SHardSwish4::ew())),
    }
}
inventory::submit! {
    RoutineImpl {
        func: Routine::HardSwish, dt: DatumType::F16, target: "generic",
        feature: None, tier: Tier::Generic, isa_rank: 0, kernel: "HHardSwish8",
        check: || true,
        factory: Some(RoutineFactory::F16(|| crate::generic::HHardSwish8::ew())),
    }
}
inventory::submit! {
    RoutineImpl {
        func: Routine::Gelu, dt: DatumType::F32, target: "generic",
        feature: None, tier: Tier::Generic, isa_rank: 0, kernel: "SGelu4",
        check: || true,
        factory: Some(RoutineFactory::F32(|| crate::generic::SGelu4::ew())),
    }
}
inventory::submit! {
    RoutineImpl {
        func: Routine::Gelu, dt: DatumType::F16, target: "generic",
        feature: None, tier: Tier::Generic, isa_rank: 0, kernel: "HGelu8",
        check: || true,
        factory: Some(RoutineFactory::F16(|| crate::generic::HGelu8::ew())),
    }
}
inventory::submit! {
    RoutineImpl {
        func: Routine::LeakyRelu, dt: DatumType::F32, target: "generic",
        feature: None, tier: Tier::Generic, isa_rank: 0, kernel: "SLeakyRelu4",
        check: || true,
        factory: Some(RoutineFactory::F32Param(|| crate::generic::SLeakyRelu4::ew())),
    }
}
inventory::submit! {
    RoutineImpl {
        func: Routine::LeakyRelu, dt: DatumType::F16, target: "generic",
        feature: None, tier: Tier::Generic, isa_rank: 0, kernel: "HLeakyRelu8",
        check: || true,
        factory: Some(RoutineFactory::F16Param(|| crate::generic::HLeakyRelu8::ew())),
    }
}
inventory::submit! {
    RoutineImpl {
        func: Routine::MulByScalar, dt: DatumType::F32, target: "generic",
        feature: None, tier: Tier::Generic, isa_rank: 0, kernel: "SMulByScalar4",
        check: || true,
        factory: Some(RoutineFactory::F32Param(|| crate::generic::SMulByScalar4::ew())),
    }
}
inventory::submit! {
    RoutineImpl {
        func: Routine::MulByScalar, dt: DatumType::F16, target: "generic",
        feature: None, tier: Tier::Generic, isa_rank: 0, kernel: "HMulByScalar8",
        check: || true,
        factory: Some(RoutineFactory::F16Param(|| crate::generic::HMulByScalar8::ew())),
    }
}
inventory::submit! {
    RoutineImpl {
        func: Routine::ReduceMax, dt: DatumType::F32, target: "generic",
        feature: None, tier: Tier::Generic, isa_rank: 0, kernel: "SMax4",
        check: || true,
        factory: Some(RoutineFactory::F32Reduce(|| crate::generic::reduce::max::SMax4::red())),
    }
}
inventory::submit! {
    RoutineImpl {
        func: Routine::ReduceMax, dt: DatumType::F16, target: "generic",
        feature: None, tier: Tier::Generic, isa_rank: 0, kernel: "HMax8",
        check: || true,
        factory: Some(RoutineFactory::F16Reduce(|| crate::generic::reduce::max::HMax8::red())),
    }
}
inventory::submit! {
    RoutineImpl {
        func: Routine::ReduceMin, dt: DatumType::F32, target: "generic",
        feature: None, tier: Tier::Generic, isa_rank: 0, kernel: "SMin4",
        check: || true,
        factory: Some(RoutineFactory::F32Reduce(|| crate::generic::reduce::min::SMin4::red())),
    }
}
inventory::submit! {
    RoutineImpl {
        func: Routine::ReduceSum, dt: DatumType::F32, target: "generic",
        feature: None, tier: Tier::Generic, isa_rank: 0, kernel: "SSum4",
        check: || true,
        factory: Some(RoutineFactory::F32Reduce(|| crate::generic::reduce::sum::SSum4::red())),
    }
}
inventory::submit! {
    RoutineImpl {
        func: Routine::ReduceSum, dt: DatumType::F16, target: "generic",
        feature: None, tier: Tier::Generic, isa_rank: 0, kernel: "HSum8",
        check: || true,
        factory: Some(RoutineFactory::F16Reduce(|| crate::generic::reduce::sum::HSum8::red())),
    }
}
inventory::submit! {
    RoutineImpl {
        func: Routine::Softmax, dt: DatumType::F32, target: "generic",
        feature: None, tier: Tier::Generic, isa_rank: 0, kernel: "SSoftMaxL2",
        check: || true,
        factory: Some(RoutineFactory::F32MapReduce(
            || crate::generic::reduce::softmax_l2::SSoftMaxL2::red(),
        )),
    }
}
inventory::submit! {
    RoutineImpl {
        func: Routine::Softmax, dt: DatumType::F16, target: "generic",
        feature: None, tier: Tier::Generic, isa_rank: 0, kernel: "HSoftMaxL2",
        check: || true,
        factory: Some(RoutineFactory::F16MapReduce(
            || crate::generic::reduce::softmax_l2::HSoftMaxL2::red(),
        )),
    }
}

// Binary ops (by-scalar and unicast layouts), the generic scalar floor. f16 is
// gated on `has_fp16()` to mirror the old `bin_by_scalar`/`bin_unicast` guard,
// which fell back to the non-linalg path for f16 without hardware fp16.
macro_rules! gbin {
    ($op:ident, $bs32:path, $bs16:path, $uc32:path, $uc16:path) => {
        inventory::submit! { RoutineImpl {
            func: Routine::BinByScalar(BinOp::$op), dt: DatumType::F32, target: "generic",
            feature: None, tier: Tier::Generic, isa_rank: 0, kernel: stringify!($op),
            check: || true, factory: Some(RoutineFactory::Bin(|| <$bs32>::bin())),
        }}
        inventory::submit! { RoutineImpl {
            func: Routine::BinByScalar(BinOp::$op), dt: DatumType::F16, target: "generic",
            feature: None, tier: Tier::Generic, isa_rank: 0, kernel: stringify!($op),
            check: || crate::has_fp16(), factory: Some(RoutineFactory::Bin(|| <$bs16>::bin())),
        }}
        inventory::submit! { RoutineImpl {
            func: Routine::BinUnicast(BinOp::$op), dt: DatumType::F32, target: "generic",
            feature: None, tier: Tier::Generic, isa_rank: 0, kernel: stringify!($op),
            check: || true, factory: Some(RoutineFactory::Bin(|| <$uc32>::bin())),
        }}
        inventory::submit! { RoutineImpl {
            func: Routine::BinUnicast(BinOp::$op), dt: DatumType::F16, target: "generic",
            feature: None, tier: Tier::Generic, isa_rank: 0, kernel: stringify!($op),
            check: || crate::has_fp16(), factory: Some(RoutineFactory::Bin(|| <$uc16>::bin())),
        }}
    };
}

use crate::generic::by_scalar as bs;
use crate::generic::unicast as uc;
gbin!(Mul, bs::SMulByScalar4, bs::HMulByScalar8, uc::SUnicastMul4, uc::HUnicastMul8);
gbin!(Add, bs::SAddByScalar4, bs::HAddByScalar8, uc::SUnicastAdd4, uc::HUnicastAdd8);
gbin!(Sub, bs::SSubByScalar4, bs::HSubByScalar8, uc::SUnicastSub4, uc::HUnicastSub8);
gbin!(SubF, bs::SSubFByScalar4, bs::HSubFByScalar8, uc::SUnicastSubF4, uc::HUnicastSubF8);
gbin!(Min, bs::SMinByScalar4, bs::HMinByScalar8, uc::SUnicastMin4, uc::HUnicastMin8);
gbin!(Max, bs::SMaxByScalar4, bs::HMaxByScalar8, uc::SUnicastMax4, uc::HUnicastMax8);
