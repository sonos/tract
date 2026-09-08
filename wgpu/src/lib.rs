//! WebGPU backend for tract, built on the `wgpu` crate.
//!
//! Native (Vulkan/Metal/DX12): blocking `request_adapter` / `PollType::Wait`.
//! Wasm (`wasm32-unknown-unknown`): enable wgpu's
//! `fragile-send-sync-non-atomic-wasm` feature (on by default here). That is
//! sound **only** because tract's browser CPU tiers compile **without**
//! atomics — `DeviceContext` / `OwnedDeviceTensor` require `Send + Sync`, and
//! wgpu's web types are `!Send`/`!Sync` otherwise.
//!
//! **Mutually exclusive** with the
//! `+atomics,+bulk-memory,+mutable-globals,+simd128` wasm-bindgen-rayon tier
//! (`linalg/MULTITHREAD_BENCHMARKS.md`). In a video call that is the correct
//! trade: inference must not take four CPU cores.
//!
//! Startup is one async (`wgpu_context_async`). Fully-covered models stay
//! synchronous: **one await at the end of `run()`**. On web, `PollType::Wait`
//! is a no-op — do not use blocking `to_host` unless Cargo feature `jspi` is
//! on (JSPI suspends across `mapAsync`; Safari 26 lacks it, Safari 27 has it).
//! Await [`to_host_async`] **once** after `run()` on the default wasm path.
//!
//! `DeviceBuffer::ptr()` is an identity of the `wgpu::Buffer` handle, not a
//! GPU address. Kernels downcast `WgpuBuffer` and bind
//! `(buffer, byte_offset as uniform)`.

mod context;
pub mod coverage;
pub mod jspi;
pub mod kernels;
pub mod ops;
mod tensor;
mod tests;
mod transform;
mod utils;

use tract_core::internal::*;
use tract_core::transform::ModelTransform;

pub use crate::context::{
    CACHE_STATS, KernelTime, WgpuContext, WgpuQueue, wgpu_context, wgpu_context_async,
    with_wgpu_queue,
};
pub use crate::coverage::{UncoveredOp, ensure_wgpu_coverage, uncovered_ops};
pub use crate::jspi::{hybrid_fallback_available, jspi_in_browser};

pub use crate::transform::WgpuTransform;

use crate::utils::get_wgpu_buffer;
use tract_gpu::tensor::DeviceTensor;

/// Single await after `run()` on web (and a blocking poll on native).
pub async fn to_host_async(tensor: &DeviceTensor) -> TractResult<Tensor> {
    let ctx = wgpu_context();
    let buffer = get_wgpu_buffer(tensor).inner.clone();
    let offset = tensor.buffer_offset::<usize>() as u64;
    let len = (tensor.len() * tensor.datum_type().size_of()) as u64;
    let bytes = ctx.download_async(&buffer, offset, len).await?;
    unsafe { Tensor::from_raw_dt(tensor.datum_type(), tensor.shape(), &bytes) }
}

#[derive(Debug)]
struct WgpuRuntime;

impl Runtime for WgpuRuntime {
    fn name(&self) -> StaticName {
        "wgpu".into()
    }

    fn prepare_with_options(
        &self,
        mut model: TypedModel,
        options: &RunOptions,
    ) -> TractResult<Box<dyn Runnable>> {
        WgpuTransform.transform(&mut model)?;
        model = model.into_optimized()?;
        if hybrid_fallback_available() {
            let bad = uncovered_ops(&model)?;
            if !bad.is_empty() {
                let list = bad
                    .iter()
                    .map(|u| format!("{} (node {:?})", u.op, u.node))
                    .collect::<Vec<_>>()
                    .join(", ");
                log::warn!(
                    "tract-wgpu hybrid fallback: running on CPU for {list}. \
                     Covered ops stay on GPU."
                );
            }
        } else {
            ensure_wgpu_coverage(&model)?;
        }

        let options = RunOptions { skip_order_opt_ram: true, ..options.clone() };
        let mut runnable = TypedSimplePlan::build(model, &options)?;
        if let Some(hints) = options.memory_sizing_hints {
            let turn_handler =
                tract_gpu::turn_handler::DeviceTurnHandler::from_plan(&runnable, &hints)
                    .context("While sizing memory arena. Missing hint ?")?;
            runnable = runnable.with_turn_handler(turn_handler);
        }

        Ok(Box::new(Arc::new(runnable)))
    }

    fn check(&self) -> TractResult<()> {
        Ok(())
    }
}

register_runtime!(WgpuRuntime = WgpuRuntime);
