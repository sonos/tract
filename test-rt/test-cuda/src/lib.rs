#![cfg(test)]

use infra::device_runtime::{DeviceTestBackend, DeviceTestRuntime};
use pastey::paste;
use tract_core::internal::*;
use tract_core::runtime::runtime_for_name;

#[path = "../suite.rs"]
mod suite;

#[derive(Debug)]
struct CudaBackend {
    phase: usize,
}

impl DeviceTestBackend for CudaBackend {
    fn transform(&self, model: &mut TypedModel) -> TractResult<()> {
        tract_cuda::CudaTransform.transform_up_to_phase(model, self.phase)
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

    fn check(&self) -> TractResult<()> {
        runtime_for_name("cuda")?.context("No cuda runtime found")?;
        Ok(())
    }
}

macro_rules! cuda_test_suite {
    ($id: ident, $phase: expr, $optimize: expr, $transpose_inputs: ident, $use_arena: ident) => {
        paste! {
            mod [<$id>] {
                use super::*;

                fn runtime() -> &'static DeviceTestRuntime<CudaBackend> {
                    lazy_static::lazy_static! {
                        static ref RT: DeviceTestRuntime<CudaBackend> = DeviceTestRuntime {
                            name: stringify!([<$id>]),
                            backend: CudaBackend { phase: $phase },
                            optimize: $optimize,
                            transpose_inputs: $transpose_inputs,
                            use_arena: $use_arena,
                        };
                    };
                    &RT
                }

                include!(concat!(env!("OUT_DIR"), "/tests/tests.rs"));
            }
        }
    };
}

//cuda_test_suite!(cuda_phase_2_translate, 2, false, , false, false);
//cuda_test_suite!(cuda_phase_3_post_translate, 3, false, , false, false);
cuda_test_suite!(optimized_cuda, usize::MAX, true, false, false);
cuda_test_suite!(optimized_cuda_with_arena, usize::MAX, true, false, true);
cuda_test_suite!(optimized_cuda_transpose, usize::MAX, true, true, false);
