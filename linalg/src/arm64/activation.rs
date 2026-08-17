//! aarch64 activation descriptors. Compiled for aarch64, or under
//! `registry-all-targets` on any host (with `factory: None`).

use crate::activation::{ActivationFn, ActivationImpl, Tier};
#[cfg(target_arch = "aarch64")]
use crate::{activation::ActFactory, frame::element_wise::ElementWiseKer};
use tract_data::prelude::DatumType;

#[cfg(target_arch = "aarch64")]
macro_rules! factory {
    (F32, $k:path) => {
        Some(ActFactory::F32(|| <$k>::ew()))
    };
    (F16, $k:path) => {
        Some(ActFactory::F16(|| <$k>::ew()))
    };
}
#[cfg(not(target_arch = "aarch64"))]
macro_rules! factory {
    ($dt:ident, $k:path) => {
        None
    };
}
#[cfg(target_arch = "aarch64")]
macro_rules! check {
    ($e:expr) => {
        $e
    };
}
#[cfg(not(target_arch = "aarch64"))]
macro_rules! check {
    ($e:expr) => {
        false
    };
}

inventory::submit! {
    ActivationImpl {
        func: ActivationFn::Sigmoid, dt: DatumType::F32, target: "aarch64",
        feature: None, tier: Tier::Native, isa_rank: 10,
        kernel: "arm64simd_sigmoid_f32_4n",
        check: || check!(true),
        factory: factory!(F32, crate::arm64::arm64simd_sigmoid_f32_4n),
    }
}
inventory::submit! {
    ActivationImpl {
        func: ActivationFn::Sigmoid, dt: DatumType::F16, target: "aarch64",
        feature: Some("fp16"), tier: Tier::Native, isa_rank: 20,
        kernel: "arm64fp16_sigmoid_f16_8n",
        check: || check!(crate::arm64::has_fp16()),
        factory: factory!(F16, crate::arm64::arm64fp16_sigmoid_f16_8n),
    }
}
// Always available on aarch64 (baseline NEON); when fp16 is present the native
// kernel above outranks it by tier, so no `!has_fp16()` gate is needed.
inventory::submit! {
    ActivationImpl {
        func: ActivationFn::Sigmoid, dt: DatumType::F16, target: "aarch64",
        feature: None, tier: Tier::Via("f32"), isa_rank: 10,
        kernel: "arm64simd_sigmoid_f16_4n",
        check: || check!(true),
        factory: factory!(F16, crate::arm64::arm64simd_sigmoid_f16_4n),
    }
}

inventory::submit! {
    ActivationImpl {
        func: ActivationFn::Silu, dt: DatumType::F32, target: "aarch64",
        feature: None, tier: Tier::Native, isa_rank: 10,
        kernel: "arm64simd_silu_f32_4n_fused",
        check: || check!(true),
        factory: factory!(F32, crate::arm64::arm64simd_silu_f32_4n_fused),
    }
}
// No native fp16 silu kernel exists (arm64.rs uses this via-f32 kernel even with
// fp16 hardware), so this is the best available on aarch64 regardless of fp16.
inventory::submit! {
    ActivationImpl {
        func: ActivationFn::Silu, dt: DatumType::F16, target: "aarch64",
        feature: None, tier: Tier::Via("f32"), isa_rank: 10,
        kernel: "arm64simd_silu_f16_4n",
        check: || check!(true),
        factory: factory!(F16, crate::arm64::arm64simd_silu_f16_4n),
    }
}

inventory::submit! {
    ActivationImpl {
        func: ActivationFn::Tanh, dt: DatumType::F32, target: "aarch64",
        feature: None, tier: Tier::Native, isa_rank: 10,
        kernel: "arm64simd_tanh_f32_4n",
        check: || check!(true),
        factory: factory!(F32, crate::arm64::arm64simd_tanh_f32_4n),
    }
}
// f16 tanh is native only with fp16; without it there is no via-f32 fallback (falls
// to generic), matching arm64.rs which leaves tanh_f16 at generic when !has_fp16.
inventory::submit! {
    ActivationImpl {
        func: ActivationFn::Tanh, dt: DatumType::F16, target: "aarch64",
        feature: Some("fp16"), tier: Tier::Native, isa_rank: 20,
        kernel: "arm64fp16_tanh_f16_8n",
        check: || check!(crate::arm64::has_fp16()),
        factory: factory!(F16, crate::arm64::arm64fp16_tanh_f16_8n),
    }
}
inventory::submit! {
    ActivationImpl {
        func: ActivationFn::HardSwish, dt: DatumType::F32, target: "aarch64",
        feature: None, tier: Tier::Native, isa_rank: 10,
        kernel: "arm64simd_hardswish_f32_8n",
        check: || check!(true),
        factory: factory!(F32, crate::arm64::arm64simd_hardswish_f32_8n),
    }
}
inventory::submit! {
    ActivationImpl {
        func: ActivationFn::Gelu, dt: DatumType::F32, target: "aarch64",
        feature: None, tier: Tier::Native, isa_rank: 10,
        kernel: "arm64simd_gelu_f32_4n_fused",
        check: || check!(true),
        factory: factory!(F32, crate::arm64::arm64simd_gelu_f32_4n_fused),
    }
}
