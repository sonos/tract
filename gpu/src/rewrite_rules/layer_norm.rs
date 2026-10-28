use tract_core::internal::*;
use tract_core::ops::math::{Add, Mul};
use tract_core::ops::nn::LayerNorm;

use super::rms_norm::axis_vector_const;

/// Search pattern => LAYER_NORM(A) [* W] [+ B] (W, B per-axis const vectors, the learned
/// `gamma` and `beta`, at least one of them). Rewrites to the affine `LayerNorm(A, W, B)` so
/// GPU backends run norm and affine part as one kernel; a missing W is ones, a missing B zeros.
/// Only folds ops that keep the norm's datum type.
pub fn fuse_layer_norm_affine(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    node_name: &str,
    op: &LayerNorm,
) -> TractResult<Option<TypedModelPatch>> {
    rule_if!(node.inputs.len() == 1);
    let in_fact = model.node_input_facts(node.id)?[0];
    let (rank, dt) = (in_fact.rank(), in_fact.datum_type);
    rule_if_some!(dim = in_fact.shape[op.axis].to_usize().ok());
    let keeps_dt = |n: &TypedNode| n.outputs[0].fact.datum_type == dt;
    let mut last = node;
    let mut scale = None;
    if let Some(mul) = model.single_succ(last.id)?.filter(|n| keeps_dt(n))
        && let Some(w) = axis_vector_const::<Mul>(model, mul, rank, op.axis, dim)
    {
        scale = Some(w.val().clone());
        last = mul;
    }
    let mut bias = None;
    if let Some(add) = model.single_succ(last.id)?.filter(|n| keeps_dt(n))
        && let Some(b) = axis_vector_const::<Add>(model, add, rank, op.axis, dim)
    {
        bias = Some(b.val().clone());
        last = add;
    }
    rule_if!(last.id != node.id);
    let as_vector = |t: Option<Arc<Tensor>>, fill: f32| -> TractResult<Arc<Tensor>> {
        Ok(match t {
            Some(t) => t.cast_to::<f32>()?.into_owned().into_shape(&[dim])?.into_arc_tensor(),
            None => tensor1(&vec![fill; dim]).into_arc_tensor(),
        })
    };
    let mut patch = TypedModelPatch::default();
    let x = patch.tap_model(model, node.inputs[0])?;
    let scale = patch.add_const(format!("{node_name}.scale"), as_vector(scale, 1.0)?)?;
    let bias = patch.add_const(format!("{node_name}.bias"), as_vector(bias, 0.0)?)?;
    let y = patch.wire_node(format!("{node_name}.affine"), op.clone(), &[x, scale, bias])?;
    patch.shunt_outside(model, last.id.into(), y[0])?;
    Ok(Some(patch))
}
