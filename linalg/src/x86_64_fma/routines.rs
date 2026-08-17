//! x86_64 activation descriptors. Compiled for x86_64, or under
//! `registry-all-targets` on any host (with `factory: None`).

use crate::routines::{Routine, RoutineImpl, Tier};
#[cfg(target_arch = "x86_64")]
use crate::{
    frame::element_wise::ElementWiseKer,
    frame::reduce::{MapReduceKer, ReduceKer},
    routines::RoutineFactory,
};
use tract_data::prelude::DatumType;

#[cfg(target_arch = "x86_64")]
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
    (F32MapReduce, $k:path) => {
        Some(RoutineFactory::F32MapReduce(|| <$k>::red()))
    };
    (F16MapReduce, $k:path) => {
        Some(RoutineFactory::F16MapReduce(|| <$k>::red()))
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
    pub fn avx512fp16() -> bool {
        std::is_x86_feature_detected!("avx512fp16")
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
    pub fn avx512fp16() -> bool {
        false
    }
}

inventory::submit! {
    RoutineImpl {
        func: Routine::Sigmoid, dt: DatumType::F32, target: "x86_64",
        feature: Some("avx"), tier: Tier::Native, isa_rank: 10,
        kernel: "avx_sigmoid_f32",
        check: || probe::avx(),
        factory: factory!(F32, crate::x86_64_fma::avx_sigmoid_f32),
    }
}
inventory::submit! {
    RoutineImpl {
        func: Routine::Sigmoid, dt: DatumType::F32, target: "x86_64",
        feature: Some("fma"), tier: Tier::Native, isa_rank: 20,
        kernel: "fma_sigmoid_f32",
        check: || probe::fma(),
        factory: factory!(F32, crate::x86_64_fma::fma_sigmoid_f32),
    }
}
inventory::submit! {
    RoutineImpl {
        func: Routine::Sigmoid, dt: DatumType::F32, target: "x86_64",
        feature: Some("avx512f"), tier: Tier::Native, isa_rank: 30,
        kernel: "avx512_sigmoid_f32",
        check: || probe::avx512f(),
        factory: factory!(F32, crate::x86_64_fma::avx512_sigmoid_f32),
    }
}
inventory::submit! {
    RoutineImpl {
        func: Routine::Sigmoid, dt: DatumType::F16, target: "x86_64",
        feature: Some("avx512f"), tier: Tier::Via("f32"), isa_rank: 30,
        kernel: "x86_64_avx512_sigmoid_f16_16n",
        check: || probe::avx512f(),
        factory: factory!(F16, crate::x86_64_fma::act_f16::x86_64_avx512_sigmoid_f16_16n),
    }
}

// silu needs FMA — there is no avx-only silu, so a plain-avx box falls to generic.
inventory::submit! {
    RoutineImpl {
        func: Routine::Silu, dt: DatumType::F32, target: "x86_64",
        feature: Some("fma"), tier: Tier::Native, isa_rank: 20,
        kernel: "fma_silu_f32",
        check: || probe::fma(),
        factory: factory!(F32, crate::x86_64_fma::fma_silu_f32),
    }
}
inventory::submit! {
    RoutineImpl {
        func: Routine::Silu, dt: DatumType::F32, target: "x86_64",
        feature: Some("avx512f"), tier: Tier::Native, isa_rank: 30,
        kernel: "x86_64_avx512_silu_f32_16n",
        check: || probe::avx512f(),
        factory: factory!(F32, crate::x86_64_fma::act::x86_64_avx512_silu_f32_16n),
    }
}
inventory::submit! {
    RoutineImpl {
        func: Routine::Silu, dt: DatumType::F16, target: "x86_64",
        feature: Some("avx512f"), tier: Tier::Via("f32"), isa_rank: 30,
        kernel: "x86_64_avx512_silu_f16_16n",
        check: || probe::avx512f(),
        factory: factory!(F16, crate::x86_64_fma::act_f16::x86_64_avx512_silu_f16_16n),
    }
}

// tanh: avx / fma / avx512 native f32; f16 only as an avx512 via-f32 kernel.
inventory::submit! {
    RoutineImpl {
        func: Routine::Tanh, dt: DatumType::F32, target: "x86_64",
        feature: Some("avx"), tier: Tier::Native, isa_rank: 10,
        kernel: "avx_tanh_f32",
        check: || probe::avx(),
        factory: factory!(F32, crate::x86_64_fma::avx_tanh_f32),
    }
}
inventory::submit! {
    RoutineImpl {
        func: Routine::Tanh, dt: DatumType::F32, target: "x86_64",
        feature: Some("fma"), tier: Tier::Native, isa_rank: 20,
        kernel: "fma_tanh_f32",
        check: || probe::fma(),
        factory: factory!(F32, crate::x86_64_fma::fma_tanh_f32),
    }
}
inventory::submit! {
    RoutineImpl {
        func: Routine::Tanh, dt: DatumType::F32, target: "x86_64",
        feature: Some("avx512f"), tier: Tier::Native, isa_rank: 30,
        kernel: "avx512_tanh_f32",
        check: || probe::avx512f(),
        factory: factory!(F32, crate::x86_64_fma::avx512_tanh_f32),
    }
}
inventory::submit! {
    RoutineImpl {
        func: Routine::Tanh, dt: DatumType::F16, target: "x86_64",
        feature: Some("avx512f"), tier: Tier::Via("f32"), isa_rank: 30,
        kernel: "x86_64_avx512_tanh_f16_16n",
        check: || probe::avx512f(),
        factory: factory!(F16, crate::x86_64_fma::act_f16::x86_64_avx512_tanh_f16_16n),
    }
}

// erf: avx512 native f32 only.
inventory::submit! {
    RoutineImpl {
        func: Routine::Erf, dt: DatumType::F32, target: "x86_64",
        feature: Some("avx512f"), tier: Tier::Native, isa_rank: 30,
        kernel: "x86_64_avx512_erf_f32_64n",
        check: || probe::avx512f(),
        factory: factory!(F32, crate::x86_64_fma::erf::x86_64_avx512_erf_f32_64n),
    }
}

// hardswish: avx512 native f32; f16 has a via-f32 kernel and a native avx512fp16 one.
inventory::submit! {
    RoutineImpl {
        func: Routine::HardSwish, dt: DatumType::F32, target: "x86_64",
        feature: Some("avx512f"), tier: Tier::Native, isa_rank: 30,
        kernel: "x86_64_avx512_hardswish_f32_64n",
        check: || probe::avx512f(),
        factory: factory!(F32, crate::x86_64_fma::act::x86_64_avx512_hardswish_f32_64n),
    }
}
inventory::submit! {
    RoutineImpl {
        func: Routine::HardSwish, dt: DatumType::F16, target: "x86_64",
        feature: Some("avx512f"), tier: Tier::Via("f32"), isa_rank: 30,
        kernel: "x86_64_avx512_hardswish_f16_64n",
        check: || probe::avx512f(),
        factory: factory!(F16, crate::x86_64_fma::act_f16::x86_64_avx512_hardswish_f16_64n),
    }
}
inventory::submit! {
    RoutineImpl {
        func: Routine::HardSwish, dt: DatumType::F16, target: "x86_64",
        feature: Some("avx512fp16"), tier: Tier::Native, isa_rank: 40,
        kernel: "x86_64_avx512fp16_hardswish_f16_128n",
        check: || probe::avx512fp16(),
        factory: factory!(F16, crate::x86_64_fma::act_f16_fp16::x86_64_avx512fp16_hardswish_f16_128n),
    }
}

// gelu: avx512 native f32; f16 via-f32.
inventory::submit! {
    RoutineImpl {
        func: Routine::Gelu, dt: DatumType::F32, target: "x86_64",
        feature: Some("avx512f"), tier: Tier::Native, isa_rank: 30,
        kernel: "x86_64_avx512_gelu_f32_16n",
        check: || probe::avx512f(),
        factory: factory!(F32, crate::x86_64_fma::act::x86_64_avx512_gelu_f32_16n),
    }
}
inventory::submit! {
    RoutineImpl {
        func: Routine::Gelu, dt: DatumType::F16, target: "x86_64",
        feature: Some("avx512f"), tier: Tier::Via("f32"), isa_rank: 30,
        kernel: "x86_64_avx512_gelu_f16_16n",
        check: || probe::avx512f(),
        factory: factory!(F16, crate::x86_64_fma::act_f16::x86_64_avx512_gelu_f16_16n),
    }
}

// leaky-relu: avx512 native f32; f16 via-f32 (the native avx512fp16 kernel exists
// but is a measured regression, so plug() never used it — nor does the registry).
inventory::submit! {
    RoutineImpl {
        func: Routine::LeakyRelu, dt: DatumType::F32, target: "x86_64",
        feature: Some("avx512f"), tier: Tier::Native, isa_rank: 30,
        kernel: "x86_64_avx512_leaky_relu_f32_64n",
        check: || probe::avx512f(),
        factory: factory!(F32Param, crate::x86_64_fma::act::x86_64_avx512_leaky_relu_f32_64n),
    }
}
inventory::submit! {
    RoutineImpl {
        func: Routine::LeakyRelu, dt: DatumType::F16, target: "x86_64",
        feature: Some("avx512f"), tier: Tier::Via("f32"), isa_rank: 30,
        kernel: "x86_64_avx512_leaky_relu_f16_64n",
        check: || probe::avx512f(),
        factory: factory!(F16Param, crate::x86_64_fma::act_f16::x86_64_avx512_leaky_relu_f16_64n),
    }
}

// mul-by-scalar: a plain-avx f32 kernel (used from the avx tier up); no f16 SIMD.
inventory::submit! {
    RoutineImpl {
        func: Routine::MulByScalar, dt: DatumType::F32, target: "x86_64",
        feature: Some("avx"), tier: Tier::Native, isa_rank: 10,
        kernel: "x86_64_avx_f32_mul_by_scalar_32n",
        check: || probe::avx(),
        factory: factory!(F32Param, crate::x86_64_fma::by_scalar::x86_64_avx_f32_mul_by_scalar_32n),
    }
}

// max / min asm is plain AVX (despite the `fma` in the symbol names); avx512 adds a
// wider max. No x86 f16 or sum reducer — those fall to generic.
inventory::submit! {
    RoutineImpl {
        func: Routine::ReduceMax, dt: DatumType::F32, target: "x86_64",
        feature: Some("avx"), tier: Tier::Native, isa_rank: 10,
        kernel: "x86_64_fma_max_f32_32n",
        check: || probe::avx(),
        factory: factory!(F32Reduce, crate::x86_64_fma::max::x86_64_fma_max_f32_32n),
    }
}
inventory::submit! {
    RoutineImpl {
        func: Routine::ReduceMax, dt: DatumType::F32, target: "x86_64",
        feature: Some("avx512f"), tier: Tier::Native, isa_rank: 30,
        kernel: "x86_64_avx512_max_f32_64n",
        check: || probe::avx512f(),
        factory: factory!(F32Reduce, crate::x86_64_fma::max::x86_64_avx512_max_f32_64n),
    }
}
inventory::submit! {
    RoutineImpl {
        func: Routine::ReduceMin, dt: DatumType::F32, target: "x86_64",
        feature: Some("avx"), tier: Tier::Native, isa_rank: 10,
        kernel: "x86_64_fma_min_f32_32n",
        check: || probe::avx(),
        factory: factory!(F32Reduce, crate::x86_64_fma::min::x86_64_fma_min_f32_32n),
    }
}

// softmax: fma native f32, avx512 native f32 + f16.
inventory::submit! {
    RoutineImpl {
        func: Routine::Softmax, dt: DatumType::F32, target: "x86_64",
        feature: Some("fma"), tier: Tier::Native, isa_rank: 20,
        kernel: "x86_64_fma_softmax2_fastcompact_f32_32n",
        check: || probe::fma(),
        factory: factory!(F32MapReduce, crate::x86_64_fma::softmax::x86_64_fma_softmax2_fastcompact_f32_32n),
    }
}
inventory::submit! {
    RoutineImpl {
        func: Routine::Softmax, dt: DatumType::F32, target: "x86_64",
        feature: Some("avx512f"), tier: Tier::Native, isa_rank: 30,
        kernel: "x86_64_avx512_softmax2_fastcompact_f32_64n",
        check: || probe::avx512f(),
        factory: factory!(F32MapReduce, crate::x86_64_fma::softmax::x86_64_avx512_softmax2_fastcompact_f32_64n),
    }
}
inventory::submit! {
    RoutineImpl {
        func: Routine::Softmax, dt: DatumType::F16, target: "x86_64",
        feature: Some("avx512f"), tier: Tier::Native, isa_rank: 30,
        kernel: "x86_64_avx512_softmax2_fastcompact_f16_64n",
        check: || probe::avx512f(),
        factory: factory!(F16MapReduce, crate::x86_64_fma::softmax::x86_64_avx512_softmax2_fastcompact_f16_64n),
    }
}
