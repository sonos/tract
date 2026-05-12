//! `CoremlOp`: a fused tract op that owns an `MLModel` handle and dispatches
//! inference through Core ML.
//!
//! ## Lifecycle
//!
//! `CoremlOp` is constructed by [`crate::transform::CoremlTransform`] for each
//! identified subgraph (see [`crate::fusion`]). It carries:
//! - a shared `Arc<CoremlContext>` (the loaded `MLModel` — refcounted so that
//!   typed-model clones don't trigger reloads)
//! - positional input/output feature names matching the MIL program's I/O
//! - precomputed `output_facts` (so `TypedOp::output_facts()` answers without
//!   round-tripping to the MLModel at typed-model construction time)
//! - per-input/output `coreml_*_shapes` for boundary reshape
//!
//! ## Eval flow (per-call)
//!
//! ```text
//!   tract Tensor inputs (rank R_t per fact)
//!     → reshape to coreml_input_shapes[i] (rank R_ml per MLPackage decl)
//!     → tensor_to_mlmultiarray (zero-copy, just creates MLMultiArray view)
//!     → MLFeatureValue wrappers
//!     → CoremlContext::predict (calls MLModel.predictionFromFeatures)
//!     → MLFeatureValue outputs
//!     → mlmultiarray_to_tensor (stride-aware copy — Core ML may pad inner dims)
//!     → reshape to output_facts[i].shape (rank R_t per tract fact)
//!     → tract Tensor outputs
//! ```
//!
//! ## Why two shape vectors per slot
//!
//! tract uses CHW (rank 3) or NCHW (rank 4) depending on the op's data_format.
//! Inside an MLPackage, our convention is to rank-pad to 4 (with leading 1s)
//! so all chains stay rank-4 internally. For tensors that exceed rank 5
//! (Hiera windowed-attention 6D forms with declutter-inserted unit dims),
//! the per-op translator strips leading unit dims at the MLPackage boundary.
//!
//! This means tract's view (the fact shape) and the MLPackage's view (the
//! coreml_*_shapes) can have different ranks. CoremlOp does the metadata-only
//! reshape at the boundary via `tract_core::Tensor::into_shape` (no data copy
//! when element counts match — squeeze/unsqueeze of unit dims is cheap).
//!
//! ## Cost model
//!
//! Each CoremlOp call has ~10–40 ms cross-domain dispatch overhead on Apple
//! Silicon (varies by ANE/GPU/CPU routing). For models with many CoremlOps
//! (e.g., SAM 2 with 39), this dominates inference time. Subgraph fusion
//! (see [`crate::fusion`]) is what amortises this cost.

use std::hash::{Hash, Hasher};
use std::sync::Arc;

use anyhow::anyhow;
use objc2::rc::Retained;
use objc2::runtime::AnyObject;
use objc2_core_ml::{MLFeatureProvider, MLFeatureValue};
use objc2_foundation::NSString;

use tract_core::internal::*;

use crate::context::CoremlContext;
use crate::tensor::{mlmultiarray_to_tensor, tensor_to_mlmultiarray};

/// Fused subgraph node — wraps a loaded CoreML `MLModel` plus the I/O wiring
/// needed to translate between tract `Tensor` and Core ML's feature dict.
#[derive(Clone)]
pub struct CoremlOp {
    /// The loaded MLModel handle. `Arc` so that cargo-clones of the typed model
    /// don't trigger reloads.
    pub context: Arc<CoremlContext>,

    /// CoreML feature names corresponding to each positional input tensor.
    pub input_names: Vec<String>,

    /// CoreML feature names corresponding to each positional output tensor.
    pub output_names: Vec<String>,

    /// Output type facts (computed at transform time so we don't need to query
    /// the MLModel at fact-propagation time).
    pub output_facts: TVec<TypedFact>,

    /// Per-input shape to feed to the MLPackage. May differ from the tract
    /// input fact's shape — for example when tract uses `data_format=CHW`
    /// (rank 3, no batch axis) but MIL conv requires rank 4: the MLPackage
    /// shape prepends `N=1` and we reshape on the way in. The byte layouts
    /// are identical when the only difference is a leading `1` dim.
    pub coreml_input_shapes: Vec<Vec<usize>>,

    /// Per-input MLPackage-side dtype. Usually matches the tract input
    /// fact's dtype (F16 for floats), but can differ when tract uses a dtype
    /// MIL feature inputs don't natively support: in particular **I64 →
    /// I32**. tract typically declares Gather index inputs as I64 (matching
    /// ONNX's `INT64` semantics), but MLMultiArray only supports Int32 in
    /// that family. When tract dtype is I64 and MIL dtype is I32, eval
    /// inserts a per-element narrowing cast before constructing the
    /// MLMultiArray. Empty when no per-input dtype override is needed.
    pub coreml_input_dtypes: Vec<DatumType>,

    /// Per-output shape returned by the MLPackage. May differ from
    /// `output_facts[i].shape` for the same CHW/NCHW reason as above; we
    /// reshape on the way out.
    pub coreml_output_shapes: Vec<Vec<usize>>,
}

impl std::fmt::Debug for CoremlOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CoremlOp")
            .field("inputs", &self.input_names)
            .field("outputs", &self.output_names)
            .finish_non_exhaustive()
    }
}

// Two `CoremlOp`s are equal iff they wrap the same `MLModel` handle and have
// the same I/O wiring. `Arc::ptr_eq` is a conservative proxy for "same model"
// — it'll say two distinct loads of the same `.mlpackage` are different ops,
// which is correct: each load is its own MLModel handle. The compile cache
// (Phase 2) will be where deduplication happens.
impl PartialEq for CoremlOp {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.context, &other.context)
            && self.input_names == other.input_names
            && self.output_names == other.output_names
    }
}

impl Eq for CoremlOp {}

impl Hash for CoremlOp {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash by Arc pointer identity (matches PartialEq via ptr_eq).
        (Arc::as_ptr(&self.context) as usize).hash(state);
        self.input_names.hash(state);
        self.output_names.hash(state);
    }
}

impl Op for CoremlOp {
    fn name(&self) -> StaticName {
        "CoremlOp".into()
    }

    op_as_typed_op!();
}

impl EvalOp for CoremlOp {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        if inputs.len() != self.input_names.len() {
            bail!(
                "CoremlOp input arity mismatch: got {} tensors, expected {}",
                inputs.len(),
                self.input_names.len()
            );
        }
        if inputs.len() != self.coreml_input_shapes.len() {
            bail!(
                "CoremlOp coreml_input_shapes wiring mismatch: {} vs {}",
                inputs.len(),
                self.coreml_input_shapes.len()
            );
        }

        // Reshape inputs to the MLPackage's expected shapes (handles CHW→NCHW
        // prepending of N=1, etc.). `into_shape` is a metadata-only operation
        // when the byte count is preserved. Then narrow dtype if the
        // MLPackage expects a different one (e.g. tract I64 → MIL I32 for
        // Gather indices); MLMultiArray doesn't support Int64 natively so
        // the cast must happen before the MLMultiArray is constructed.
        let reshaped_inputs: Vec<Tensor> = inputs
            .iter()
            .enumerate()
            .map(|(i, tv)| {
                let target = &self.coreml_input_shapes[i];
                let mut t: Tensor = if tv.shape() == target.as_slice() {
                    (**tv).clone()
                } else {
                    (**tv).clone().into_shape(target)?
                };
                // Optional dtype narrowing — only kicks in when fusion
                // declared a different MIL dtype than the tract input fact.
                if let Some(&want_dt) = self.coreml_input_dtypes.get(i) {
                    if t.datum_type() != want_dt {
                        t = t.cast_to_dt(want_dt)?.into_owned();
                    }
                }
                Ok(t)
            })
            .collect::<TractResult<_>>()?;

        // Convert each (reshaped) input tensor to an MLMultiArray, then to MLFeatureValue.
        // We need to keep both the arrays and the feature values alive as
        // separate Vecs so the &AnyObject refs survive into make_input_provider.
        let arrays: Vec<_> =
            reshaped_inputs.iter().map(tensor_to_mlmultiarray).collect::<TractResult<Vec<_>>>()?;
        let feature_values: Vec<Retained<MLFeatureValue>> = arrays
            .iter()
            .map(|arr| unsafe { MLFeatureValue::featureValueWithMultiArray(arr) })
            .collect();

        let pairs: Vec<(&str, &AnyObject)> = self
            .input_names
            .iter()
            .zip(feature_values.iter())
            .map(|(name, fv)| (name.as_str(), &**fv as &AnyObject))
            .collect();
        let provider = CoremlContext::make_input_provider(&pairs)?;

        // Run prediction.
        let out_provider = self.context.predict(&provider)?;

        // Pull each output back into a tract Tensor, reshaping to the
        // tract-side fact shape (handles NCHW→CHW stripping when applicable).
        let mut outputs: TVec<TValue> = tvec![];
        for (i, name) in self.output_names.iter().enumerate() {
            let name_ns = NSString::from_str(name);
            let fv = unsafe { out_provider.featureValueForName(&name_ns) }
                .ok_or_else(|| anyhow!("CoreML output {name:?} missing from prediction"))?;
            let arr = unsafe { fv.multiArrayValue() }
                .ok_or_else(|| anyhow!("CoreML output {name:?} not a MultiArray"))?;
            let t = mlmultiarray_to_tensor(&arr)?;

            // Reshape if the tract-side fact shape differs from what the
            // MLPackage produced (e.g. CHW vs NCHW with leading 1).
            let target_shape = self.output_facts[i].shape.as_concrete().ok_or_else(|| {
                anyhow!(
                    "CoremlOp output_facts[{i}] has symbolic shape: {:?}",
                    self.output_facts[i].shape
                )
            })?;
            let final_t = if t.shape() == target_shape { t } else { t.into_shape(target_shape)? };
            outputs.push(final_t.into());
        }
        Ok(outputs)
    }
}

impl TypedOp for CoremlOp {
    fn output_facts(&self, _inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(self.output_facts.clone())
    }

    as_op!();
}
