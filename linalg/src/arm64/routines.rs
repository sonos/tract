//! aarch64 activation descriptors. Compiled for aarch64, or under
//! `registry-all-targets` on any host (with `factory: None`).

use crate::routines::{Routine, RoutineImpl, Tier};
#[cfg(target_arch = "aarch64")]
use crate::{
    frame::element_wise::ElementWiseKer,
    frame::reduce::{MapReduceKer, ReduceKer},
    routines::RoutineFactory,
};
use tract_data::prelude::DatumType;

#[cfg(target_arch = "aarch64")]
macro_rules! factory {
    (F32, $k:path) => {
        Some(RoutineFactory::F32(|| <$k>::ew()))
    };
    (F16, $k:path) => {
        Some(RoutineFactory::F16(|| <$k>::ew()))
    };
    (F32Param, $k:path) => {
        Some(RoutineFactory::F32Param(|| <$k>::ew()))
    };
    (F16Param, $k:path) => {
        Some(RoutineFactory::F16Param(|| <$k>::ew()))
    };
    (F32Reduce, $k:path) => {
        Some(RoutineFactory::F32Reduce(|| <$k>::red()))
    };
    (F16Reduce, $k:path) => {
        Some(RoutineFactory::F16Reduce(|| <$k>::red()))
    };
    (F32MapReduce, $k:path) => {
        Some(RoutineFactory::F32MapReduce(|| <$k>::red()))
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
    RoutineImpl {
        func: Routine::Sigmoid, dt: DatumType::F32, target: "aarch64",
        feature: None, tier: Tier::Native, isa_rank: 10,
        kernel: "arm64simd_sigmoid_f32_4n",
        check: || check!(true),
        factory: factory!(F32, crate::arm64::arm64simd_sigmoid_f32_4n),
    }
}
inventory::submit! {
    RoutineImpl {
        func: Routine::Sigmoid, dt: DatumType::F16, target: "aarch64",
        feature: Some("fp16"), tier: Tier::Native, isa_rank: 20,
        kernel: "arm64fp16_sigmoid_f16_8n",
        check: || check!(crate::arm64::has_fp16()),
        factory: factory!(F16, crate::arm64::arm64fp16_sigmoid_f16_8n),
    }
}
// Always available on aarch64 (baseline NEON); when fp16 is present the native
// kernel above outranks it by tier, so no `!has_fp16()` gate is needed.
inventory::submit! {
    RoutineImpl {
        func: Routine::Sigmoid, dt: DatumType::F16, target: "aarch64",
        feature: None, tier: Tier::Via("f32"), isa_rank: 10,
        kernel: "arm64simd_sigmoid_f16_4n",
        check: || check!(true),
        factory: factory!(F16, crate::arm64::arm64simd_sigmoid_f16_4n),
    }
}

inventory::submit! {
    RoutineImpl {
        func: Routine::Silu, dt: DatumType::F32, target: "aarch64",
        feature: None, tier: Tier::Native, isa_rank: 10,
        kernel: "arm64simd_silu_f32_4n_fused",
        check: || check!(true),
        factory: factory!(F32, crate::arm64::arm64simd_silu_f32_4n_fused),
    }
}
// No native fp16 silu kernel exists (arm64.rs uses this via-f32 kernel even with
// fp16 hardware), so this is the best available on aarch64 regardless of fp16.
inventory::submit! {
    RoutineImpl {
        func: Routine::Silu, dt: DatumType::F16, target: "aarch64",
        feature: None, tier: Tier::Via("f32"), isa_rank: 10,
        kernel: "arm64simd_silu_f16_4n",
        check: || check!(true),
        factory: factory!(F16, crate::arm64::arm64simd_silu_f16_4n),
    }
}

inventory::submit! {
    RoutineImpl {
        func: Routine::Tanh, dt: DatumType::F32, target: "aarch64",
        feature: None, tier: Tier::Native, isa_rank: 10,
        kernel: "arm64simd_tanh_f32_4n",
        check: || check!(true),
        factory: factory!(F32, crate::arm64::arm64simd_tanh_f32_4n),
    }
}
// f16 tanh is native only with fp16; without it there is no via-f32 fallback (falls
// to generic), matching arm64.rs which leaves tanh_f16 at generic when !has_fp16.
inventory::submit! {
    RoutineImpl {
        func: Routine::Tanh, dt: DatumType::F16, target: "aarch64",
        feature: Some("fp16"), tier: Tier::Native, isa_rank: 20,
        kernel: "arm64fp16_tanh_f16_8n",
        check: || check!(crate::arm64::has_fp16()),
        factory: factory!(F16, crate::arm64::arm64fp16_tanh_f16_8n),
    }
}
inventory::submit! {
    RoutineImpl {
        func: Routine::HardSwish, dt: DatumType::F32, target: "aarch64",
        feature: None, tier: Tier::Native, isa_rank: 10,
        kernel: "arm64simd_hardswish_f32_8n",
        check: || check!(true),
        factory: factory!(F32, crate::arm64::arm64simd_hardswish_f32_8n),
    }
}
inventory::submit! {
    RoutineImpl {
        func: Routine::Gelu, dt: DatumType::F32, target: "aarch64",
        feature: None, tier: Tier::Native, isa_rank: 10,
        kernel: "arm64simd_gelu_f32_4n_fused",
        check: || check!(true),
        factory: factory!(F32, crate::arm64::arm64simd_gelu_f32_4n_fused),
    }
}
inventory::submit! {
    RoutineImpl {
        func: Routine::LeakyRelu, dt: DatumType::F32, target: "aarch64",
        feature: None, tier: Tier::Native, isa_rank: 10,
        kernel: "arm64simd_leaky_relu_f32_8n",
        check: || check!(true),
        factory: factory!(F32Param, crate::arm64::arm64simd_leaky_relu_f32_8n),
    }
}
inventory::submit! {
    RoutineImpl {
        func: Routine::LeakyRelu, dt: DatumType::F16, target: "aarch64",
        feature: Some("fp16"), tier: Tier::Native, isa_rank: 20,
        kernel: "arm64fp16_leaky_relu_f16_16n",
        check: || check!(crate::arm64::has_fp16()),
        factory: factory!(F16Param, crate::arm64::arm64fp16_leaky_relu_f16_16n),
    }
}
inventory::submit! {
    RoutineImpl {
        func: Routine::MulByScalar, dt: DatumType::F32, target: "aarch64",
        feature: None, tier: Tier::Native, isa_rank: 10,
        kernel: "arm64simd_mul_by_scalar_f32_16n",
        check: || check!(true),
        factory: factory!(F32Param, crate::arm64::arm64simd_mul_by_scalar_f32_16n),
    }
}
inventory::submit! {
    RoutineImpl {
        func: Routine::MulByScalar, dt: DatumType::F16, target: "aarch64",
        feature: Some("fp16"), tier: Tier::Native, isa_rank: 20,
        kernel: "arm64fp16_mul_by_scalar_f16_32n",
        check: || check!(crate::arm64::has_fp16()),
        factory: factory!(F16Param, crate::arm64::arm64fp16_mul_by_scalar_f16_32n),
    }
}

inventory::submit! {
    RoutineImpl {
        func: Routine::Max, dt: DatumType::F32, target: "aarch64",
        feature: None, tier: Tier::Native, isa_rank: 10,
        kernel: "arm64simd_max_f32_16n",
        check: || check!(true),
        factory: factory!(F32Reduce, crate::arm64::arm64simd_max_f32_16n),
    }
}
inventory::submit! {
    RoutineImpl {
        func: Routine::Max, dt: DatumType::F16, target: "aarch64",
        feature: Some("fp16"), tier: Tier::Native, isa_rank: 20,
        kernel: "arm64fp16_max_f16_32n",
        check: || check!(crate::arm64::has_fp16()),
        factory: factory!(F16Reduce, crate::arm64::arm64fp16_max_f16_32n),
    }
}
inventory::submit! {
    RoutineImpl {
        func: Routine::Min, dt: DatumType::F32, target: "aarch64",
        feature: None, tier: Tier::Native, isa_rank: 10,
        kernel: "arm64simd_min_f32_16n",
        check: || check!(true),
        factory: factory!(F32Reduce, crate::arm64::arm64simd_min_f32_16n),
    }
}
inventory::submit! {
    RoutineImpl {
        func: Routine::Sum, dt: DatumType::F32, target: "aarch64",
        feature: None, tier: Tier::Native, isa_rank: 10,
        kernel: "arm64simd_sum_f32_16n",
        check: || check!(true),
        factory: factory!(F32Reduce, crate::arm64::arm64simd_sum_f32_16n),
    }
}
inventory::submit! {
    RoutineImpl {
        func: Routine::Sum, dt: DatumType::F16, target: "aarch64",
        feature: Some("fp16"), tier: Tier::Native, isa_rank: 20,
        kernel: "arm64fp16_sum_f16_32n",
        check: || check!(crate::arm64::has_fp16()),
        factory: factory!(F16Reduce, crate::arm64::arm64fp16_sum_f16_32n),
    }
}
inventory::submit! {
    RoutineImpl {
        func: Routine::Softmax, dt: DatumType::F32, target: "aarch64",
        feature: None, tier: Tier::Native, isa_rank: 10,
        kernel: "arm64simd_softmax2_fastcompact_f32_16n",
        check: || check!(true),
        factory: factory!(F32MapReduce, crate::arm64::arm64simd_softmax2_fastcompact_f32_16n),
    }
}
