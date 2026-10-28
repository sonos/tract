use tract_core::internal::*;
use tract_core::ops::cnn::Conv;
use tract_gpu::rewrite_rules::conv_nhwc::conv_to_nhwc;
use tract_gpu::rule_ensure;

/// Run a 2D f16 NCHW conv as an NHWC conv between two transposes.
///
/// cuDNN's tensor-core convolutions work in NHWC; handed NCHW, they transpose the data and
/// reorder the filter on every call. `propagate_nhwc` then cancels the transposes between
/// consecutive convs. Must run after `rewrite_kernel_conv_in_oihw`, which would turn the OHWI
/// filter back.
pub fn rewrite_conv_to_nhwc(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    name: &str,
    conv: &Conv,
) -> TractResult<Option<TypedModelPatch>> {
    rule_ensure!(model.outlet_fact(node.inputs[0])?.datum_type == f16::datum_type());
    conv_to_nhwc(model, node, name, conv)
}
