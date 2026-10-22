use tract_core::internal::*;
use tract_core::ops::cnn::Conv;
use tract_gpu::rewrite_rules::conv_nhwc::conv_to_nhwc;
use tract_gpu::rule_ensure;

/// Filter-to-input element ratio above which an f16 conv runs in NHWC.
const NHWC_MIN_FILTER_TO_INPUT_RATIO: usize = 8;

/// Run a 2D f16 NCHW conv whose filter dwarfs its input as an NHWC conv between two transposes.
///
/// cuDNN's tensor-core convolutions work in NHWC; handed NCHW, they reorder the filter on
/// every call, which dominates when the filter is much larger than the activations (small
/// spatial extent, many channels). Must run after `rewrite_kernel_conv_in_oihw`, which would
/// turn the OHWI filter back.
pub fn rewrite_conv_to_nhwc(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    name: &str,
    conv: &Conv,
) -> TractResult<Option<TypedModelPatch>> {
    let input_fact = model.outlet_fact(node.inputs[0])?;
    rule_ensure!(input_fact.datum_type == f16::datum_type());
    rule_if_some!(input_shape = input_fact.shape.as_concrete());
    rule_if_some!(kernel = &model.outlet_fact(node.inputs[1])?.konst);
    rule_ensure!(
        kernel.len() >= NHWC_MIN_FILTER_TO_INPUT_RATIO * input_shape.iter().product::<usize>()
    );
    conv_to_nhwc(model, node, name, conv)
}
