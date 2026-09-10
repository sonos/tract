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
mod utils;

use tract_core::internal::*;
use tract_core::transform::ModelTransform;

use crate::func_constants::{ConstantValues, Value};
use crate::kernels::LibraryName;
pub use crate::kernels::matmul::MetalGemmImplKind;

pub use crate::context::{MetalContext, MetalStream, with_metal_stream};
pub use crate::transform::MetalTransform;

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
        MetalTransform::default().transform(&mut model)?;
        model = model.into_optimized()?;

        let options = RunOptions { skip_order_opt_ram: true, ..options.clone() };
        let mut runnable = TypedSimplePlan::build(model, &options)?;
        // Always plan transients through the device memory arena: without it
        // every transient is an individually allocated (wired) Metal buffer,
        // and a large-batch forward churns through gigabytes of alloc/free,
        // spiking the process footprint into the compressor and stalling the
        // next forwards on driver re-residency. Hints only tune the packing
        // order; missing symbols fall back to a representative default.
        if options.enable_gpu_memory_arena.unwrap_or(true) {
            let hints = options.memory_sizing_hints.clone().unwrap_or_default();
            let turn_handler =
                tract_gpu::turn_handler::DeviceTurnHandler::from_plan(&runnable, &hints)
                    .context("While sizing memory arena")?;
            runnable = runnable.with_turn_handler(turn_handler);
        }

        Ok(Box::new(Arc::new(runnable)))
    }

    fn check(&self) -> TractResult<()> {
        Ok(())
    }
}

register_runtime!(MetalRuntime = MetalRuntime);
