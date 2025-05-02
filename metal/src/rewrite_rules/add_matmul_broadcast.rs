use crate::{MetalGemmImplKind, MetalTransform};
use tract_core::internal::*;
use tract_core::ops::array::MultiBroadcastTo;
use tract_core::ops::einsum::prefix_matmul::PrefixMatMul;
use tract_gpu::rule_ensure;

pub fn add_broadcast_pre_matmul(
    ctx: &MetalTransform,
    model: &TypedModel,
    node: &TypedNode,
    node_name: &str,
    op: &PrefixMatMul,
) -> TractResult<Option<TypedModelPatch>> {
    let in_facts = model.node_input_facts(node.id)?;
    // GGML supports broadcast
    rule_ensure!(
        !(ctx.gemm_impl == Some(MetalGemmImplKind::Ggml)
            || (ctx.gemm_impl.is_none() && in_facts[0].datum_type == DatumType::F32))
    );

    // Detect broadcast
    let b_batch_dims = in_facts[1].shape.dims()[..in_facts[0].rank() - 2].to_vec();

    let a_rank = in_facts[0].rank();
    let mut a_batch_dims = in_facts[0].shape[..(a_rank - 2)].to_vec();

    a_batch_dims.retain(|tdim| !matches!(tdim, TDim::Sym(_)) || b_batch_dims.contains(tdim));
    let symb_in_a = a_batch_dims != in_facts[0].shape[..(a_rank - 2)].to_vec();
    let symb_in_b = b_batch_dims != in_facts[1].shape[..(a_rank - 2)].to_vec();

    let a_batch_size = a_batch_dims.iter().product::<TDim>().gcd();
    let b_batch_size = b_batch_dims.iter().product::<TDim>().gcd();

    let (activ_slot, weight_slot) = if (a_batch_size % b_batch_size == 0)
        && ((a_batch_size != b_batch_size) || symb_in_a)
    {
        (0, 1)
    } else if (b_batch_size % a_batch_size == 0) && ((a_batch_size != b_batch_size) || symb_in_b) {
        (1, 0)
    } else {
        return Ok(None);
    };

    let mut patch = TypedModelPatch::default();
    let activ = patch.tap_model(model, node.inputs[activ_slot])?;
    let weights = patch.tap_model(model, node.inputs[weight_slot])?;
    let brd_shape = ShapeFact::from_dims(
        [
            in_facts[activ_slot].shape.dims()[..a_rank - 2].to_vec(),
            in_facts[weight_slot].shape.dims()[a_rank - 2..].to_vec(),
        ]
        .concat(),
    );
    let brd = MultiBroadcastTo { shape: brd_shape };

    let brd_out = patch.wire_node(format!("{node_name}.broadcast"), brd, &[weights])?[0];

    let mm_out = patch.wire_node(node_name, op.clone(), &[activ, brd_out])?[0];

    patch.shunt_outside(model, node.id.into(), mm_out)?;

    Ok(Some(patch))
}
