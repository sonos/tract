use crate::rule_ensure;
use tract_core::internal::*;
use tract_core::ops::cnn::{Conv, KernelFormat};
use tract_core::ops::nn::DataFormat;

/// Run a 2D NCHW conv as an NHWC conv between two transposes.
///
/// GPU conv kernels (cuDNN tensor cores, MLX implicit GEMM) work in NHWC. The filter is
/// reordered to OHWI here, once, as a constant, and the input and output are transposed
/// instead. Requires a constant OIHW filter, a single group and symmetric padding. Backends
/// add their own gates (datum type, shape) before calling it.
pub fn conv_to_nhwc(
    model: &TypedModel,
    node: &TypedNode,
    name: &str,
    conv: &Conv,
) -> TractResult<Option<TypedModelPatch>> {
    rule_ensure!(conv.pool_spec.data_format == DataFormat::NCHW);
    rule_ensure!(conv.kernel_fmt == KernelFormat::OIHW);
    rule_ensure!(conv.group == 1 && conv.q_params.is_none());
    let input_fact = model.outlet_fact(node.inputs[0])?;
    rule_ensure!(input_fact.rank() == 4);
    rule_if_some!(input_shape = input_fact.shape.as_concrete());
    rule_if_some!(kernel = &model.outlet_fact(node.inputs[1])?.konst);
    let data_shape = conv.pool_spec.data_format.shape(input_shape)?;
    rule_ensure!(
        conv.pool_spec
            .computed_padding(data_shape.hw_dims())
            .iter()
            .all(|p| p.pad_before == p.pad_after)
    );

    let mut patch = TypedModelPatch::default();
    let mut inputs = patch.taps(model, &node.inputs)?;
    inputs[0] = patch.wire_node(format!("{name}.to_nhwc"), AxisOp::Move(1, 3), &[inputs[0]])?[0];
    let ohwi = (**kernel).clone().move_axis(1, 3)?;
    inputs[1] = patch.add_const(format!("{name}.kernel_ohwi"), ohwi)?;
    let mut nhwc = conv.clone();
    nhwc.pool_spec.data_format = DataFormat::NHWC;
    nhwc.kernel_fmt = KernelFormat::OHWI;
    let wire = patch.wire_node(format!("{name}.nhwc"), nhwc, &inputs)?;
    let wire = patch.wire_node(name, AxisOp::Move(3, 1), &wire)?;
    patch.shunt_outside(model, node.id.into(), wire[0])?;
    Ok(Some(patch))
}
