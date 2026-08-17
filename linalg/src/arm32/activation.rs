//! armv7 (target_arch = "arm") activation descriptors. Compiled for arm, or under
//! `registry-all-targets` on any host (with `factory: None`).

use crate::activation::{ActivationFn, ActivationImpl, Tier};
#[cfg(target_arch = "arm")]
use crate::{activation::ActFactory, frame::element_wise::ElementWiseKer};
use tract_data::prelude::DatumType;

#[cfg(target_arch = "arm")]
macro_rules! factory {
    (F32, $k:path) => {
        Some(ActFactory::F32(|| <$k>::ew()))
    };
}
#[cfg(not(target_arch = "arm"))]
macro_rules! factory {
    ($dt:ident, $k:path) => {
        None
    };
}
#[cfg(target_arch = "arm")]
macro_rules! check {
    ($e:expr) => {
        $e
    };
}
#[cfg(not(target_arch = "arm"))]
macro_rules! check {
    ($e:expr) => {
        false
    };
}

inventory::submit! {
    ActivationImpl {
        func: ActivationFn::Sigmoid, dt: DatumType::F32, target: "armv7",
        feature: Some("neon"), tier: Tier::Native, isa_rank: 10,
        kernel: "armv7neon_sigmoid_f32_4n",
        check: || check!(crate::arm32::has_neon()),
        factory: factory!(F32, crate::arm32::armv7neon::armv7neon_sigmoid_f32_4n),
    }
}
inventory::submit! {
    ActivationImpl {
        func: ActivationFn::Silu, dt: DatumType::F32, target: "armv7",
        feature: Some("neon"), tier: Tier::Native, isa_rank: 10,
        kernel: "armv7neon_silu_f32_4n",
        check: || check!(crate::arm32::has_neon()),
        factory: factory!(F32, crate::arm32::armv7neon::armv7neon_silu_f32_4n),
    }
}
inventory::submit! {
    ActivationImpl {
        func: ActivationFn::Tanh, dt: DatumType::F32, target: "armv7",
        feature: Some("neon"), tier: Tier::Native, isa_rank: 10,
        kernel: "armv7neon_tanh_f32_4n",
        check: || check!(crate::arm32::has_neon()),
        factory: factory!(F32, crate::arm32::armv7neon::armv7neon_tanh_f32_4n),
    }
}
