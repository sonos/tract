mod context;
pub mod kernels;
pub mod ops;
mod rewrite_rules;
mod tensor;
mod transform;
pub mod utils;

pub use context::CUDA_STREAM;
use log::warn;
use tract_core::internal::*;
use tract_core::transform::ModelTransform;
pub use transform::CudaTransform;

use crate::utils::are_culibs_present;
const Q40_ROW_PADDING: usize = 512;

#[derive(Debug)]
struct CudaRuntime;

impl Runtime for CudaRuntime {
    fn name(&self) -> StaticName {
        "cuda".into()
    }

    fn prepare(&self, mut model: TypedModel) -> TractResult<Box<dyn Runnable>> {
        ensure!(are_culibs_present());
        CudaTransform.transform(&mut model)?;
        model = model.into_optimized()?;
        let arena_hints = Option::<SymbolValues>::None;

        let mut runnable = TypedSimplePlan::new(model)?;
        if let Ok(session_handler) = tract_gpu::session_handler::DeviceSessionHandler::from_plan(
            &runnable,
            &arena_hints.unwrap_or_default(),
        ) {
            runnable = runnable.with_session_handler(session_handler);
        } else {
            warn!("Unable to compute session handler in cuda runtime");
        }

        Ok(Box::new(runnable))
    }
}

register_runtime!(CudaRuntime = CudaRuntime);
