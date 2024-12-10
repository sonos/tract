#![cfg(all(test, any(target_os = "macos", target_os = "ios")))]

use std::borrow::Cow;
use std::sync::Arc;
use tract_core::internal::*;

use tract_core::runtime::Runtime;

#[path = "../suite.rs"]
mod suite;

#[derive(Debug)]
struct MetalTestRuntime {
    name: &'static str,
    phase: usize,
    optimize: bool,
}

impl Runtime for MetalTestRuntime {
    fn name(&self) -> Cow<str> {
        self.name.into()
    }

    fn prepare(&self, mut model: TypedModel) -> TractResult<Box<dyn Runnable>> {
        tract_metal::transform::MetalTransform::default()
            .transform_up_to_phase(&mut model, self.phase)?;
        if self.optimize {
            model = model.into_optimized()?;
        }
        Ok(Box::new(Arc::new(model.into_runnable()?)))
    }
}

macro_rules! metal_test_suite {
    ($id: ident, $phase: expr, $optimize: expr) => {
        mod $id {
            use super::*;

            fn runtime() -> &'static MetalTestRuntime {
                lazy_static::lazy_static! {
                    static ref RT: MetalTestRuntime = MetalTestRuntime { name: stringify!($id), phase: $phase, optimize: $optimize };
                };
                &RT
            }

            include!(concat!(env!("OUT_DIR"), "/tests/tests.rs"));
        }
    };
}

metal_test_suite!(metal_phase_0_einsum, 0, false);
metal_test_suite!(metal_phase_1_pre_translate, 1, false);
metal_test_suite!(metal_phase_2_translate, 2, false);
metal_test_suite!(metal_phase_3_post_translate, 3, false);
metal_test_suite!(optimized_metal, usize::MAX, true);
