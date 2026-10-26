#![cfg(all(test, not(target_arch = "wasm32")))]

use infra::device_runtime::{DeviceTestBackend, DeviceTestRuntime};
use pastey::paste;
use tract_core::internal::*;
use tract_core::runtime::runtime_for_name;
use tract_core::transform::ModelTransform;

#[path = "../suite.rs"]
mod suite;

#[derive(Debug)]
struct WgpuBackend;

impl DeviceTestBackend for WgpuBackend {
    fn transform(&self, model: &mut TypedModel) -> TractResult<()> {
        tract_wgpu::WgpuTransform.transform(model)
    }

    /// WebGPU refuses a dispatch that binds one buffer both read-only and
    /// read-write, so the plan gives each partition a buffer of its own rather
    /// than a region of one arena.
    fn with_arena(
        &self,
        plan: TypedSimplePlan,
        memory_hint: &SymbolValues,
    ) -> TractResult<TypedSimplePlan> {
        Ok(match tract_wgpu::WgpuTurnHandler::for_plan(&plan, Some(memory_hint))? {
            Some(handler) => plan.with_turn_handler(handler),
            None => plan,
        })
    }

    fn check(&self) -> TractResult<()> {
        runtime_for_name("wgpu")?.context("No wgpu runtime found")?;
        Ok(())
    }
}

macro_rules! wgpu_test_suite {
    ($id: ident, $optimize: expr, $transpose_inputs: ident, $use_arena: ident) => {
        paste! {
            mod [<$id>] {
                use super::*;

                fn runtime() -> &'static DeviceTestRuntime<WgpuBackend> {
                    lazy_static::lazy_static! {
                        static ref RT: DeviceTestRuntime<WgpuBackend> = DeviceTestRuntime {
                            name: stringify!([<$id>]),
                            backend: WgpuBackend,
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

wgpu_test_suite!(wgpu_translate, false, false, false);
wgpu_test_suite!(optimized_wgpu, true, false, false);
wgpu_test_suite!(optimized_wgpu_transpose, true, true, false);
wgpu_test_suite!(optimized_wgpu_planned, true, false, true);
