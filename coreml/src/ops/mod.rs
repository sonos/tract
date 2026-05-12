//! Per-tract-op MIL translators. Each module here handles one tract op family
//! (Conv, MatMul, BinOp, etc.) and exposes:
//! - an `analyse_*` function returning `*Analysis::{Translatable(plan), Skip(reason)}`
//! - a `*Plan` struct carrying the data needed to emit the MIL ops
//! - an `emit_*_mil` function used by [`crate::fusion::build_subgraph_mlpackage`]
//!   to write the MIL ops + (optional) weight blob entries
//!
//! Phase 1 shipped Conv only. Phase 2 adds: `binop` (Add/Mul/Max in this commit),
//! then `cast`, `einsum` — see `notes/phase-1-mvp.md` §11.1 for the priority + rationale.

pub mod activation;
pub mod add_axis;
pub mod avgpool;
pub mod binop;
pub mod broadcast;
pub mod cast;
pub mod concat;
pub mod conv;
pub mod einsum;
pub mod einsum_outer;
pub mod gather;
pub mod general_matmul;
pub mod iff;
pub mod instance_norm;
pub mod layer_norm;
pub mod matmul;
pub mod maxpool;
pub mod move_axis;
pub mod pad;
pub mod rank;
pub mod reduce;
pub mod reshape;
pub mod resize;
pub mod rm_axis;
pub mod rms_norm;
pub mod scatter_nd;
pub mod slice;
pub mod softmax;

use std::borrow::Cow;

use anyhow::Result;
use tract_core::internal::*;
use tract_core::ops::cast::Cast;
use tract_core::ops::konst::Const;

/// Resolve `outlet` to a constant tensor if possible. Walks `Cast(Const)`
/// chains (common after `f32_to_f16` wraps F32 weights in a Cast op rather
/// than folding) by evaluating the cast eagerly. Used by every per-op
/// `analyse_*` to absorb const-fed inputs into the MLPackage.
pub fn const_tensor(model: &TypedModel, outlet: OutletId) -> Result<Option<Cow<'_, Tensor>>> {
    let node = &model.nodes[outlet.node];
    if let Some(k) = node.op_as::<Const>() {
        return Ok(Some(Cow::Borrowed(k.val())));
    }
    if let Some(cast) = node.op_as::<Cast>()
        && node.inputs.len() == 1
        && let Some(inner) = const_tensor(model, node.inputs[0])?
    {
        let casted = inner.cast_to_dt(cast.to)?.into_owned();
        return Ok(Some(Cow::Owned(casted)));
    }
    Ok(None)
}

/// Resolve a tract symbolic shape to a concrete `Vec<i64>`, returning `None`
/// if any dim is symbolic (which would make the shape unsuitable for a static
/// MLPackage).
pub fn shape_to_concrete_i64(shape: &ShapeFact) -> Option<Vec<i64>> {
    shape.dims().iter().map(|d| d.to_i64().ok()).collect()
}
