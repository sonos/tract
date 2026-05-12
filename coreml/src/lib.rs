//! `tract-coreml`: bridge tract typed-model subgraphs to Apple Core ML for
//! Apple Neural Engine (ANE) / GPU dispatch on macOS / iOS.
//!
//! ## High-level flow
//!
//! ```text
//!   ONNX/NNEF model → tract::TypedModel
//!                    └─ CoremlTransform::transform(&mut model)
//!                       1. identify_subgraphs       — union-find translatable nodes
//!                       2. for each subgraph:
//!                          a. build_subgraph_mlpackage  — emit MIL ops (per-op translators)
//!                          b. compile + cache .mlmodelc  — Apple compileModelAtURL
//!                          c. wrap in CoremlOp           — runs MLModel.predict() at eval
//!                       3. patch source model: replace subgraph nodes with CoremlOp
//! ```
//!
//! ## Layered structure (bottom-up)
//!
//! - [`proto`] — prost-generated MIL/Model proto types (vendored from
//!   `apple/coremltools` at the commit pinned in `proto/MIL_PROTO_VERSION.md`).
//! - [`mil`] — typed builders over the generated proto types
//!   (`value`, `op`, `blob`, `program`).
//! - [`mlpackage`] — `.mlpackage` directory + `Manifest.json` writer.
//! - [`context`] — `CoremlContext`: wraps a loaded `MLModel`, runs predictions.
//! - [`tensor`] — tract `Tensor` ↔ `MLMultiArray` conversion, stride-aware
//!   (Core ML may pad inner dims for ANE; see `feedback_mlmultiarray_strides.md`
//!   in the project memory).
//! - [`coreml_op`] — `CoremlOp`: a fused tract op that owns an `MLModel` handle
//!   and dispatches inference. Bridges between tract-side (variable rank) and
//!   MLPackage-side (rank-padded to 4 by convention) via reshape at the boundary.
//! - [`compile_cache`] — content-addressed cache of compiled `.mlmodelc` directories
//!   keyed by SHA-256 of MIL bytes + weight blob bytes (canonical proto walk to
//!   handle prost HashMap iteration non-determinism).
//! - [`ops`] — per-tract-op MIL translators. ~20 modules covering Conv, BinOp,
//!   EinSum (matmul-shaped + conv-shaped), Reduce, Softmax, LayerNorm fold,
//!   RmsNorm, Reshape, MoveAxis, Pad, etc. Each translator's `analyse_*`
//!   returns `Translatable(plan)` or `Skip(reason)`; `emit_*_mil` produces
//!   the MIL ops + (optional) weight-blob entries.
//! - [`fusion`] — `identify_subgraphs` (union-find with all-or-nothing +
//!   convex-region rules) and `build_subgraph_mlpackage` (per-subgraph MIL
//!   emission). Multi-node pattern detectors for InstanceNorm + LayerNorm
//!   live in `ops/instance_norm.rs` and `ops/layer_norm.rs` and pre-scan
//!   the subgraph before per-node emission.
//! - [`transform`] — `CoremlTransform`: walks a typed model, replaces each
//!   translatable subgraph with a single `CoremlOp`.
//! - `lib` (this file) — [`CoremlRuntime`]: entry point registered via
//!   `register_runtime!` so callers reach this backend through tract's
//!   standard runtime API.
//!
//! ## In-MLPackage rank-4 convention
//!
//! Inside an MLPackage, all tensors are rank-padded to ≥ 4 (with leading 1s).
//! This matches MIL's preference for rank-4 NCHW layouts and avoids per-op
//! rank-mismatch issues. Translators that may see a tract-side rank-3 (CHW)
//! input pad to rank 4. The `CoremlOp` boundary reshapes between the
//! tract-rank and MLPackage-rank views.
//!
//! For tensors that exceed MIL's rank-5 cap (Hiera windowed-attention
//! 6D forms with declutter-inserted unit dims), translators expose
//! `*_external_shape` fields that strip leading unit dims; CoremlOp
//! squeezes/unsqueezes at the boundary.
//!
//! ## Design rationale + project history
//!
//! See the `tract-coreml-ane` project notes:
//! - `notes/phase-0-recon.md` §7 — Runtime + ModelTransform integration pattern,
//!   why we don't depend on `tract-gpu`, subgraph-fusion vs per-op dispatch.
//! - `notes/phase-1-spike.md` — offline-coremltools vs runtime MIL-build comparison.
//! - `notes/phase-N-closure.md` — per-phase gates + canary perf numbers.
//! - `notes/tract-upstream-feedback.md` — running log of issues we hit in tract
//!   that would benefit from upstream fixes.

use std::sync::Arc;

use tract_core::internal::*;
use tract_core::transform::ModelTransform;

pub mod compile_cache;
pub mod context;
pub mod coreml_op;
pub mod fusion;
pub mod mil;
pub mod mlpackage;
pub mod ops;
pub mod tensor;
pub mod transform;

pub use context::CoremlContext;
pub use coreml_op::CoremlOp;
pub use transform::CoremlTransform;

/// Runtime entry point: registered with `register_runtime!` so tract callers
/// can reach this backend via the standard `Runtime` lookup path.
#[derive(Debug)]
pub struct CoremlRuntime;

impl Runtime for CoremlRuntime {
    fn name(&self) -> StaticName {
        "coreml".into()
    }

    fn prepare_with_options(
        &self,
        mut model: TypedModel,
        options: &RunOptions,
    ) -> TractResult<Box<dyn Runnable>> {
        CoremlTransform::default().transform(&mut model)?;
        model = model.into_optimized()?;

        let runnable = TypedSimplePlan::build(model, options)?;
        Ok(Box::new(Arc::new(runnable)))
    }

    fn check(&self) -> TractResult<()> {
        // TODO(phase 2): runtime check that the platform actually has Core ML
        // (always true on macOS/iOS, never elsewhere). Return Err on others.
        Ok(())
    }
}

register_runtime!(CoremlRuntime = CoremlRuntime);

#[allow(clippy::large_enum_variant, clippy::manual_range_contains, clippy::collapsible_if)]
pub mod proto {
    pub mod core_ml {
        pub mod specification {
            include!(concat!(env!("OUT_DIR"), "/core_ml.specification.rs"));

            pub mod mil_spec {
                include!(concat!(env!("OUT_DIR"), "/core_ml.specification.mil_spec.rs"));
            }

            pub mod core_ml_models {
                include!(concat!(env!("OUT_DIR"), "/core_ml.specification.core_ml_models.rs"));
            }
        }
    }
}
