use crate::{MetalGemmImplKind, MetalTransform};
use tract_core::internal::*;
use tract_core::ops::array::MultiBroadcastTo;
use tract_core::ops::einsum::prefix_matmul::PrefixMatMul;
use tract_gpu::rule_ensure;

/// Whether every Metal GEMM takes these batch axes as they are: both sides
/// equal, or one side all ones.
pub fn batches_need_no_broadcast(a: &[TDim], b: &[TDim]) -> bool {
    a == b || a.iter().all(|d| d.is_one()) || b.iter().all(|d| d.is_one())
}

/// Whether GGML's batch broadcast computes these batch axes: it reads B's
/// batch `i / (a_batch / b_batch)` for A's batch `i`, which is right only when
/// B matches A on a prefix of the axes and is one on the rest.
pub fn is_ggml_batch_broadcast(a: &[TDim], b: &[TDim]) -> bool {
    a.len() == b.len()
        && a != b
        && (0..=a.len())
            .any(|split| a[..split] == b[..split] && b[split..].iter().all(|d| d.is_one()))
}

/// Broadcasts the inputs of a matmul whose batch axes no Metal GEMM runs as
/// they are (see [`batches_need_no_broadcast`], and
/// [`is_ggml_batch_broadcast`] when GGML is to run it) to their common batch
/// shape, so the GEMM sees equal batches.
pub fn add_broadcast_pre_matmul(
    ctx: &MetalTransform,
    model: &TypedModel,
    node: &TypedNode,
    node_name: &str,
    op: &PrefixMatMul,
) -> TractResult<Option<TypedModelPatch>> {
    let in_facts = model.node_input_facts(node.id)?;
    let rank = in_facts[0].rank();
    rule_ensure!(rank > 2 && in_facts[1].rank() == rank);
    let a_batch = &in_facts[0].shape.dims()[..rank - 2];
    let b_batch = &in_facts[1].shape.dims()[..rank - 2];
    rule_ensure!(!batches_need_no_broadcast(a_batch, b_batch));
    let ggml = ctx.gemm_impl == Some(MetalGemmImplKind::Ggml)
        || (ctx.gemm_impl.is_none() && in_facts[0].datum_type == DatumType::F32);
    rule_ensure!(!(ggml && is_ggml_batch_broadcast(a_batch, b_batch)));

    let mut batch = tvec!();
    for (a, b) in a_batch.iter().zip(b_batch) {
        if a == b || b.is_one() {
            batch.push(a.clone());
        } else if a.is_one() {
            batch.push(b.clone());
        } else {
            return Ok(None);
        }
    }

    let mut patch = TypedModelPatch::default();
    let mut inputs = tvec!();
    for (slot, fact) in in_facts.iter().enumerate() {
        let mut wire = patch.tap_model(model, node.inputs[slot])?;
        if fact.shape.dims()[..rank - 2] != *batch {
            let shape: TVec<TDim> =
                batch.iter().chain(&fact.shape.dims()[rank - 2..]).cloned().collect();
            let brd = MultiBroadcastTo { shape: shape.into() };
            wire = patch.wire_node(format!("{node_name}.broadcast_{slot}"), brd, &[wire])?[0];
        }
        inputs.push(wire);
    }
    let mm_out = patch.wire_node(node_name, *op, &inputs)?[0];
    patch.shunt_outside(model, node.id.into(), mm_out)?;
    Ok(Some(patch))
}
