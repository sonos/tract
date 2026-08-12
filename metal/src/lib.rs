mod command_buffer;
mod context;
mod encoder;
mod func_constants;
pub mod kernels;
pub mod ops;
mod rewrite_rules;
mod tensor;
mod tests;
mod transform;
pub mod tuning;
mod utils;

use tract_core::internal::*;
use tract_core::transform::ModelTransform;

use crate::func_constants::{ConstantValues, Value};
use crate::kernels::LibraryName;
pub use crate::kernels::matmul::MetalGemmImplKind;

pub use crate::context::{MetalContext, MetalStream, with_metal_stream};
pub use crate::transform::MetalTransform;
pub use crate::tuning::{
    AutotuneCache, MetalTuning, MetalTuningOverrides, autotune_cache_default_path,
    current_device_name, sanitize_device_name, set_tuning_model_key, set_tuning_overrides,
};

#[derive(Debug)]
struct MetalRuntime;

impl Runtime for MetalRuntime {
    fn name(&self) -> StaticName {
        "metal".into()
    }

    fn prepare_with_options(
        &self,
        mut model: TypedModel,
        options: &RunOptions,
    ) -> TractResult<Box<dyn Runnable>> {
        // Pick the GEMM implementation at runtime so a model can be A/B'd
        // against the alternatives without a rebuild. `ggml` matches the
        // previous default.
        let transform = std::env::var("TRACT_METAL_GEMM_IMPL")
            .unwrap_or_else(|_| "ggml".to_string())
            .parse::<MetalTransform>()?;
        transform.transform(&mut model)?;
        model = model.into_optimized()?;

        let options = RunOptions { skip_order_opt_ram: true, ..options.clone() };
        let mut runnable = TypedSimplePlan::build(model, &options)?;
        // Always plan transients through the device memory arena: without it
        // every transient is an individually allocated (wired) Metal buffer,
        // and a large-batch forward churns through gigabytes of alloc/free,
        // spiking the process footprint into the compressor and stalling the
        // next forwards on driver re-residency. Hints only tune the packing
        // order; missing symbols fall back to a representative default.
        // Escape hatch: TRACT_GPU_DISABLE_MEMORY_ARENA=1.
        if std::env::var_os("TRACT_GPU_DISABLE_MEMORY_ARENA").is_none() {
            let hints = options.memory_sizing_hints.clone().unwrap_or_default();
            let session_handler =
                tract_gpu::session_handler::DeviceSessionHandler::from_plan(&runnable, &hints)
                    .context("While sizing memory arena")?;
            runnable = runnable.with_session_handler(session_handler);
        }

        Ok(Box::new(Arc::new(runnable)))
    }

    fn check(&self) -> TractResult<()> {
        Ok(())
    }
}

register_runtime!(MetalRuntime = MetalRuntime);
