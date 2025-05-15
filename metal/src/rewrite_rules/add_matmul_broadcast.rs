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
    rule_ensure!(in_facts[0].rank() > 2);
    rule_ensure!(
        !(ctx.gemm_impl == Some(MetalGemmImplKind::Ggml)
            || (ctx.gemm_impl.is_none() && in_facts[0].datum_type == DatumType::F32))
    );

    // Detect broadcast
    let a_shape = &in_facts[0].shape;
    let b_shape = &in_facts[1].shape;
    let a_rank = a_shape.rank();
    
    let a_batch = &a_shape[..a_rank - 2];
    let b_batch = &b_shape[..a_rank - 2];
    
    // Remove from batch_dim array all symbolic dimensions also present in the other batch_dim array
    // Symbolic Dimensions will be considered as 1 in gcd() so this allows identifying a 
    // symbolic broadcast factor.
    let a_batch_dims: Vec<_> = a_batch
        .iter()
        .filter(|tdim| !matches!(tdim, TDim::Sym(_)) || b_batch.contains(tdim))
        .cloned()
        .collect();
    
    let b_batch_dims: Vec<_> = b_batch
    .iter()
    .filter(|tdim| !matches!(tdim, TDim::Sym(_)) || a_batch.contains(tdim))
    .cloned()
    .collect();

    let symb_in_a = a_batch_dims != a_batch;
    let symb_in_b = b_batch_dims != b_batch;
    
    let a_batch_size = a_batch_dims.iter().product::<TDim>().gcd();
    let b_batch_size = b_batch_dims.iter().product::<TDim>().gcd();
    
    let (activ_slot, weight_slot) = if (a_batch_size % b_batch_size == 0)
        && ((a_batch_size != b_batch_size) || symb_in_a)
    {
        (0, 1)
    } else if (b_batch_size % a_batch_size == 0)
        && ((a_batch_size != b_batch_size) || symb_in_b)
    {
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

    let inputs = if activ_slot == 1 { [brd_out, activ] } else { [activ, brd_out] };
    let mm_out = patch.wire_node(node_name, op.clone(), &inputs)?[0];

    patch.shunt_outside(model, node.id.into(), mm_out)?;

    Ok(Some(patch))
}
