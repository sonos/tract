//! wasm activation descriptors. Compiled for wasm, or under `registry-all-targets`
//! on any host (with `factory: None`).
//!
//! The native sigmoid is a relaxed-simd kernel, only compiled into a wasm build with
//! `+relaxed-simd`. wasm feature detection is compile-time, so `check` is a `cfg!`.

use crate::activation::{ActivationFn, ActivationImpl, Tier};
#[cfg(all(target_family = "wasm", target_feature = "relaxed-simd"))]
use crate::{activation::ActFactory, frame::element_wise::ElementWiseKer};
use tract_data::prelude::DatumType;

#[cfg(all(target_family = "wasm", target_feature = "relaxed-simd"))]
macro_rules! relaxed_factory {
    () => {
        Some(ActFactory::F32(|| crate::wasm::WasmSigmoid4Relaxed::ew()))
    };
}
#[cfg(not(all(target_family = "wasm", target_feature = "relaxed-simd")))]
macro_rules! relaxed_factory {
    () => {
        None
    };
}

inventory::submit! {
    ActivationImpl {
        func: ActivationFn::Sigmoid, dt: DatumType::F32, target: "wasm",
        feature: Some("relaxed-simd"), tier: Tier::Native, isa_rank: 10,
        kernel: "WasmSigmoid4Relaxed",
        check: || cfg!(all(target_family = "wasm", target_feature = "relaxed-simd")),
        factory: relaxed_factory!(),
    }
}
