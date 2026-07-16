//! Persistent-state execution of one prepared stage, plus small run helpers.

use anyhow::{Context, Result};
use tract_core::prelude::*;
use tract_core::runtime::Runnable;

/// A prepared sub-model. Each [`run`] spawns a fresh execution state, matching
/// the reference `causal_llm` loop: the KV cache is external model I/O looped by
/// [`crate::llm::StageState`], so the model itself is stateless and a persistent
/// state must NOT be reused — a reused state carries stale (device) buffers when
/// the input/cache shapes change between prefill and decode steps, which corrupts
/// output on Metal. Prepared via the backend's `Runtime`, so device placement
/// (Metal/CUDA) happens inside `prepare`.
pub struct StageRunner {
    runnable: Box<dyn Runnable>,
}

impl StageRunner {
    pub fn load(model: TypedModel, backend: &str) -> Result<Self> {
        let rt = runtime_for_name(backend)?
            .with_context(|| format!("runtime `{backend}` unavailable"))?;
        let runnable = rt.prepare(model)?;
        Ok(StageRunner { runnable })
    }

    pub fn run(&mut self, inputs: TVec<TValue>) -> Result<TVec<TValue>> {
        self.runnable.run(inputs)
    }
}

/// One-shot run on a backend with a fresh state (no persistence).
pub fn run_once(model: &TypedModel, backend: &str, inputs: TVec<TValue>) -> Result<TVec<TValue>> {
    StageRunner::load(model.clone(), backend)?.run(inputs)
}

/// Cosine similarity of two f32 tensors (flattened). Used for cross-backend
/// (f16 vs f32) tolerance checks where bit-exactness is not expected.
pub fn cosine(a: &Tensor, b: &Tensor) -> Result<f32> {
    let a = a.view();
    let a = a.as_slice::<f32>()?;
    let b = b.view();
    let b = b.as_slice::<f32>()?;
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    Ok(dot / (na * nb + 1e-12))
}
