use tract_core::internal::*;
use tract_core::ops::cnn::Conv;
use tract_gpu::rewrite_rules::conv_nhwc::conv_to_nhwc;
use tract_gpu::rule_ensure;

/// Run a 2D f16 or f32 NCHW conv as an NHWC conv between two transposes, so it reaches the
/// MLX implicit-GEMM kernel, which only takes NHWC, instead of the direct kernel.
pub fn rewrite_conv_to_nhwc(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    name: &str,
    conv: &Conv,
) -> TractResult<Option<TypedModelPatch>> {
    let dt = model.outlet_fact(node.inputs[0])?.datum_type;
    rule_ensure!(matches!(dt, DatumType::F16 | DatumType::F32));
    rule_ensure!(model.outlet_fact(node.inputs[1])?.datum_type == dt);
    conv_to_nhwc(model, node, name, conv)
}
