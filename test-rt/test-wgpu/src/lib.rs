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

    /// The memory pool's arena binds one buffer both read-only and read-write
    /// in a dispatch, which WebGPU refuses, so this backend never installs it
    /// and every runtime below runs with `use_arena` false.
    fn with_arena(
        &self,
        plan: TypedSimplePlan,
        _memory_hint: &SymbolValues,
    ) -> TractResult<TypedSimplePlan> {
        Ok(plan)
    }

    fn check(&self) -> TractResult<()> {
        runtime_for_name("wgpu")?.context("No wgpu runtime found")?;
        Ok(())
    }
}

macro_rules! wgpu_test_suite {
    ($id: ident, $optimize: expr, $transpose_inputs: ident) => {
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
                            use_arena: false,
                        };
                    };
                    &RT
                }

                include!(concat!(env!("OUT_DIR"), "/tests/tests.rs"));
            }
        }
    };
}

wgpu_test_suite!(wgpu_translate, false, false);
wgpu_test_suite!(optimized_wgpu, true, false);
wgpu_test_suite!(optimized_wgpu_transpose, true, true);
