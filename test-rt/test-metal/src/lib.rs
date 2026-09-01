#![cfg(all(test, any(target_os = "macos", target_os = "ios")))]

use infra::device_runtime::{DeviceTestBackend, DeviceTestRuntime};
use pastey::paste;
use tract_core::internal::*;
use tract_metal::MetalGemmImplKind;

#[path = "../ggml_suite.rs"]
mod ggml_suite;
#[path = "../suite.rs"]
mod suite;

#[derive(Debug)]
struct MetalBackend {
    phase: usize,
    gemm_impl: Option<MetalGemmImplKind>,
}

impl DeviceTestBackend for MetalBackend {
    fn transform(&self, model: &mut TypedModel) -> TractResult<()> {
        tract_metal::MetalTransform { gemm_impl: self.gemm_impl }
            .transform_up_to_phase(model, self.phase)
    }

    fn with_arena(
        &self,
        plan: TypedSimplePlan,
        memory_hint: &SymbolValues,
    ) -> TractResult<TypedSimplePlan> {
        let turn_handler =
            tract_gpu::turn_handler::DeviceTurnHandler::from_plan(&plan, memory_hint)?;
        Ok(plan.with_turn_handler(turn_handler))
    }
}

macro_rules! metal_test_suite {
    ($id: ident, $phase: expr, $optimize: expr, $gemm_impl: expr, $transpose_inputs: ident, $use_arena: ident) => {
        paste! {
            mod [<$id _ $gemm_impl:lower>] {
                use super::*;

                fn runtime() -> &'static DeviceTestRuntime<MetalBackend> {
                    lazy_static::lazy_static! {
                        static ref RT: DeviceTestRuntime<MetalBackend> = DeviceTestRuntime {
                            name: stringify!([<$id _ $gemm_impl:lower>]),
                            backend: MetalBackend { phase: $phase, gemm_impl: $gemm_impl },
                            optimize: $optimize,
                            transpose_inputs: $transpose_inputs,
                            use_arena: $use_arena,
                        };
                    };
                    &RT
                }

                include!(concat!(env!("OUT_DIR"), "/tests/",  stringify!([<$gemm_impl:lower>]), ".rs"));
            }
        }
    };
}

macro_rules! metal_runtime {
    ($gemm_impl: expr) => {
        metal_test_suite!(metal_phase_2_translate, 2, false, $gemm_impl, false, false);
        metal_test_suite!(metal_phase_3_post_translate, 3, false, $gemm_impl, false, false);
        metal_test_suite!(optimized_metal, usize::MAX, true, $gemm_impl, false, false);
        metal_test_suite!(optimized_metal_transpose, usize::MAX, true, $gemm_impl, true, false);
    };
}

static MLX: Option<MetalGemmImplKind> = Some(MetalGemmImplKind::Mlx);
static MFA: Option<MetalGemmImplKind> = Some(MetalGemmImplKind::Mfa);
static GGML: Option<MetalGemmImplKind> = Some(MetalGemmImplKind::Ggml);

// Common transform
metal_test_suite!(metal_phase_0_einsum, 0, false, MLX, false, false);
metal_test_suite!(metal_phase_1_pre_translate, 1, false, MLX, false, false);

metal_runtime!(None);
metal_runtime!(MLX);
metal_runtime!(MFA);
metal_runtime!(GGML);
