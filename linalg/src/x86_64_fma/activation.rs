//! x86_64 activation descriptors. Compiled for x86_64, or under
//! `registry-all-targets` on any host (with `factory: None`).

use crate::activation::{ActivationFn, ActivationImpl, Tier};
#[cfg(target_arch = "x86_64")]
use crate::{activation::ActFactory, frame::element_wise::ElementWiseKer};
use tract_data::prelude::DatumType;

#[cfg(target_arch = "x86_64")]
macro_rules! factory {
    (F32, $k:path) => {
        Some(ActFactory::F32(|| <$k>::ew()))
    };
    (F16, $k:path) => {
        Some(ActFactory::F16(|| <$k>::ew()))
    };
}
#[cfg(not(target_arch = "x86_64"))]
macro_rules! factory {
    ($dt:ident, $k:path) => {
        None
    };
}

// `is_x86_feature_detected!` needs a literal argument in its own expansion; it
// rejects one arriving through a macro metavariable. So probe with literal calls
// here, and stub to `false` off-target.
#[cfg(target_arch = "x86_64")]
mod probe {
    pub fn avx() -> bool {
        std::is_x86_feature_detected!("avx")
    }
    pub fn fma() -> bool {
        std::is_x86_feature_detected!("fma")
    }
    pub fn avx512f() -> bool {
        std::is_x86_feature_detected!("avx512f")
    }
}
#[cfg(not(target_arch = "x86_64"))]
mod probe {
    pub fn avx() -> bool {
        false
    }
    pub fn fma() -> bool {
        false
    }
    pub fn avx512f() -> bool {
        false
    }
}

inventory::submit! {
    ActivationImpl {
        func: ActivationFn::Sigmoid, dt: DatumType::F32, target: "x86_64",
        feature: Some("avx"), tier: Tier::Native, isa_rank: 10,
        kernel: "avx_sigmoid_f32",
        check: || probe::avx(),
        factory: factory!(F32, crate::x86_64_fma::avx_sigmoid_f32),
    }
}
inventory::submit! {
    ActivationImpl {
        func: ActivationFn::Sigmoid, dt: DatumType::F32, target: "x86_64",
        feature: Some("fma"), tier: Tier::Native, isa_rank: 20,
        kernel: "fma_sigmoid_f32",
        check: || probe::fma(),
        factory: factory!(F32, crate::x86_64_fma::fma_sigmoid_f32),
    }
}
inventory::submit! {
    ActivationImpl {
        func: ActivationFn::Sigmoid, dt: DatumType::F32, target: "x86_64",
        feature: Some("avx512f"), tier: Tier::Native, isa_rank: 30,
        kernel: "avx512_sigmoid_f32",
        check: || probe::avx512f(),
        factory: factory!(F32, crate::x86_64_fma::avx512_sigmoid_f32),
    }
}
inventory::submit! {
    ActivationImpl {
        func: ActivationFn::Sigmoid, dt: DatumType::F16, target: "x86_64",
        feature: Some("avx512f"), tier: Tier::Via("f32"), isa_rank: 30,
        kernel: "x86_64_avx512_sigmoid_f16_16n",
        check: || probe::avx512f(),
        factory: factory!(F16, crate::x86_64_fma::act_f16::x86_64_avx512_sigmoid_f16_16n),
    }
}

// silu needs FMA — there is no avx-only silu, so a plain-avx box falls to generic.
inventory::submit! {
    ActivationImpl {
        func: ActivationFn::Silu, dt: DatumType::F32, target: "x86_64",
        feature: Some("fma"), tier: Tier::Native, isa_rank: 20,
        kernel: "fma_silu_f32",
        check: || probe::fma(),
        factory: factory!(F32, crate::x86_64_fma::fma_silu_f32),
    }
}
inventory::submit! {
    ActivationImpl {
        func: ActivationFn::Silu, dt: DatumType::F32, target: "x86_64",
        feature: Some("avx512f"), tier: Tier::Native, isa_rank: 30,
        kernel: "x86_64_avx512_silu_f32_16n",
        check: || probe::avx512f(),
        factory: factory!(F32, crate::x86_64_fma::act::x86_64_avx512_silu_f32_16n),
    }
}
inventory::submit! {
    ActivationImpl {
        func: ActivationFn::Silu, dt: DatumType::F16, target: "x86_64",
        feature: Some("avx512f"), tier: Tier::Via("f32"), isa_rank: 30,
        kernel: "x86_64_avx512_silu_f16_16n",
        check: || probe::avx512f(),
        factory: factory!(F16, crate::x86_64_fma::act_f16::x86_64_avx512_silu_f16_16n),
    }
}
