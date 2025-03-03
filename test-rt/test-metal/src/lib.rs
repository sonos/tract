#![cfg(all(test, any(target_os = "macos", target_os = "ios")))]

use std::borrow::Cow;
use std::sync::Arc;
use tract_core::internal::*;

use paste::paste;
use tract_core::runtime::Runtime;
use tract_metal::MetalGemmImplKind;

#[path = "../suite.rs"]
mod suite;

#[derive(Debug)]
struct MetalTestRuntime {
    name: &'static str,
    phase: usize,
    optimize: bool,
    gemm_impl: MetalGemmImplKind,
}

impl Runtime for MetalTestRuntime {
    fn name(&self) -> Cow<str> {
        self.name.into()
    }

    fn prepare(&self, mut model: TypedModel) -> TractResult<Box<dyn Runnable>> {
        tract_metal::transform::MetalTransform { gemm_impl: Some(self.gemm_impl) }
            .transform_up_to_phase(&mut model, self.phase)?;
        if self.optimize {
            model = model.into_optimized()?;
        }
        Ok(Box::new(Arc::new(model.into_runnable()?)))
    }
}

macro_rules! metal_test_suite {
    ($id: ident, $phase: expr, $optimize: expr, $gemm_impl: ident) => {
        paste! {
            mod [<$id _ $gemm_impl:lower>] {
                use super::*;

                fn runtime() -> &'static MetalTestRuntime {
                    lazy_static::lazy_static! {
                        static ref RT: MetalTestRuntime = MetalTestRuntime { name: stringify!([<$id _ $gemm_impl:lower>]), phase: $phase, optimize: $optimize, gemm_impl: MetalGemmImplKind::$gemm_impl };
                    };
                    &RT
                }

                include!(concat!(env!("OUT_DIR"), "/tests/",  stringify!([<$gemm_impl:lower>]), ".rs"));
            }
        }
    };
}

macro_rules! metal_runtime {
    ($gemm_impl: ident) => {
        metal_test_suite!(metal_phase_2_translate, 2, false, $gemm_impl);
        metal_test_suite!(metal_phase_3_post_translate, 3, false, $gemm_impl);
        metal_test_suite!(optimized_metal, usize::MAX, true, $gemm_impl);
    };
}

// Common transform
metal_test_suite!(metal_phase_0_einsum, 0, false, Mlx);
metal_test_suite!(metal_phase_1_pre_translate, 1, false, Mlx);

metal_runtime!(Mlx);
metal_runtime!(Mfa);
metal_runtime!(Ggml);
