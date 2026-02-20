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

pub use crate::context::{METAL_STREAM, MetalContext, MetalStream};
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
        if let Some(hints) = options.memory_sizing_hints {
            let session_handler =
                tract_gpu::session_handler::DeviceSessionHandler::from_plan(&runnable, &hints)
                    .context("While sizing memory arena. Missing hint ?")?;
            runnable = runnable.with_session_handler(session_handler);
        }

        Ok(Box::new(Arc::new(runnable)))
    }

    fn check(&self) -> TractResult<()> {
        Ok(())
    }
}

register_runtime!(MetalRuntime = MetalRuntime);
