use crate::MetalTransform;
use tract_core::internal::*;
use tract_core::ops::einsum::prefix_matmul::PrefixMatMul;
use tract_gpu::rule_ensure;

/// Rewrite BasicMatMul { .. transpose_c: true } to BasicMatMul { .. transpose_c: false}
pub fn untranspose_matmul_output(
    _ctx: &MetalTransform,
    model: &TypedModel,
    node: &TypedNode,
    _node_name: &str,
    op: &PrefixMatMul,
) -> TractResult<Option<TypedModelPatch>> {
    rule_ensure!(op.transpose_c);

    let new_matmul = PrefixMatMul {
        transpose_a: !op.transpose_b,
        transpose_b: !op.transpose_a,
        transpose_c: false,
        ..*op
    };

    TypedModelPatch::replace_single_op(model, node, &[node.inputs[1], node.inputs[0]], new_matmul)
        .map(Some)
}
