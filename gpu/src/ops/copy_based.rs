//! Translators for ops that only need the generic copy_nd dispatch.
//! These are fully backend-agnostic and can be constructed without
//! any backend-specific arguments.

use tract_core::internal::*;
use tract_core::ops::array::{MultiBroadcastTo, Slice, TypedConcat};
use tract_pulse_opl::ops::{Delay, PulsePad};
use tract_transformers::ops::dyn_kv_cache::DynKeyValueCache;

/// Try to translate a node into a copy-based GPU op.
/// Returns `Some(gpu_op)` if the node is one of the 7 copy-based ops.
pub fn try_make_copy_based_op(
    source: &TypedModel,
    node: &TypedNode,
) -> TractResult<Option<Box<dyn TypedOp>>> {
    if let Some(op) = node.op_as::<MultiBroadcastTo>() {
        return Ok(Some(Box::new(super::broadcast::GpuMultiBroadcastTo::new(op.shape.clone()))));
    }
    if let Some(op) = node.op_as::<AxisOp>() {
        let in_fact = source.node_input_facts(node.id)?[0];
        return Ok(Some(Box::new(super::change_axes::GpuAxisOp::from_tract_core_with_fact(
            op.clone(),
            in_fact,
        ))));
    }
    if let Some(op) = node.op_as::<Slice>() {
        return Ok(Some(Box::new(super::slice::GpuSlice::new(op.clone()))));
    }
    if let Some(op) = node.op_as::<TypedConcat>() {
        return Ok(Some(Box::new(super::concat::GpuConcat::new(op.axis))));
    }
    if let Some(op) = node.op_as::<DynKeyValueCache>() {
        return Ok(Some(Box::new(super::dyn_kv_cache::GpuDynKVCache::from_tract_transformers(op))));
    }
    if let Some(op) = node.op_as::<Delay>() {
        return Ok(Some(Box::new(super::pulse::GpuDelay::new(op))));
    }
    if let Some(op) = node.op_as::<PulsePad>() {
        return Ok(Some(Box::new(super::pulse::GpuPulsePad::new(op)?)));
    }
    Ok(None)
}
