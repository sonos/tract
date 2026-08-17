//! wasm activation descriptors. Compiled for wasm, or under `registry-all-targets`
//! on any host (with `factory: None`).
//!
//! The native sigmoid is a relaxed-simd kernel, only compiled into a wasm build with
//! `+relaxed-simd`. wasm feature detection is compile-time, so `check` is a `cfg!`.

use crate::routines::{Routine, RoutineImpl, Tier};
#[cfg(all(target_family = "wasm", target_feature = "relaxed-simd"))]
use crate::{frame::element_wise::ElementWiseKer, routines::RoutineFactory};
use tract_data::prelude::DatumType;

#[cfg(all(target_family = "wasm", target_feature = "relaxed-simd"))]
macro_rules! relaxed_factory {
    ($k:path) => {
        Some(RoutineFactory::F32(|| <$k>::ew()))
    };
}
#[cfg(not(all(target_family = "wasm", target_feature = "relaxed-simd")))]
macro_rules! relaxed_factory {
    ($k:path) => {
        None
    };
}

inventory::submit! {
    RoutineImpl {
        func: Routine::Sigmoid, dt: DatumType::F32, target: "wasm",
        feature: Some("relaxed-simd"), tier: Tier::Native, isa_rank: 10,
        kernel: "WasmSigmoid4Relaxed",
        check: || cfg!(all(target_family = "wasm", target_feature = "relaxed-simd")),
        factory: relaxed_factory!(crate::wasm::WasmSigmoid4Relaxed),
    }
}
inventory::submit! {
    RoutineImpl {
        func: Routine::Tanh, dt: DatumType::F32, target: "wasm",
        feature: Some("relaxed-simd"), tier: Tier::Native, isa_rank: 10,
        kernel: "WasmTanh4Relaxed",
        check: || cfg!(all(target_family = "wasm", target_feature = "relaxed-simd")),
        factory: relaxed_factory!(crate::wasm::WasmTanh4Relaxed),
    }
}
