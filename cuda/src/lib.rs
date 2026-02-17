mod context;
pub mod kernels;
pub mod ops;
mod rewrite_rules;
mod tensor;
mod transform;
pub mod utils;

pub use context::CUDA_STREAM;
use tract_core::internal::*;
use tract_core::transform::ModelTransform;
pub use transform::CudaTransform;

use crate::utils::are_culibs_present;
use crate::utils::ensure_cuda_driver_compatible;
const Q40_ROW_PADDING: usize = 512;

#[derive(Debug)]
struct CudaRuntime;

impl Runtime for CudaRuntime {
    fn name(&self) -> StaticName {
        "cuda".into()
    }

    fn prepare_with_options(
        &self,
        mut model: TypedModel,
        options: &RunOptions,
    ) -> TractResult<Box<dyn Runnable>> {
        ensure_cuda_driver_compatible()?;
        ensure!(are_culibs_present());
        CudaTransform.transform(&mut model)?;
        model.optimize()?;

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
}

register_runtime!(CudaRuntime = CudaRuntime);
