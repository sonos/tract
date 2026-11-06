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

/// Run a 1D f16 or f32 conv as a 2D one, so `rewrite_conv_to_nhwc` takes it to the MLX
/// implicit-GEMM kernel. Only for the convs that rule then accepts (a single group and a
/// constant kernel): any other 1D conv keeps the direct kernel's 1D path. The kernel gets its
/// unit axis as a new constant, not a wired axis op: the metal transform does not declutter
/// afterwards, so the op would stay and hide the constant from `rewrite_conv_to_nhwc`.
pub fn rewrite_conv_1d_for_nhwc(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    name: &str,
    conv: &Conv,
) -> TractResult<Option<TypedModelPatch>> {
    rule_ensure!(conv.pool_spec.rank() == 1);
    let dt = model.outlet_fact(node.inputs[0])?.datum_type;
    rule_ensure!(matches!(dt, DatumType::F16 | DatumType::F32));
    rule_ensure!(model.outlet_fact(node.inputs[1])?.datum_type == dt);
    rule_ensure!(conv.group == 1 && conv.q_params.is_none());
    rule_if_some!(kernel = &model.outlet_fact(node.inputs[1])?.konst);
    let mut new = conv.clone();
    new.pool_spec = conv.pool_spec.change_geo_axes(&AxisOp::Add(1))?;
    let mut patch = TypedModelPatch::default();
    let mut wire = patch.taps(model, &node.inputs)?;
    let pos_data = conv.pool_spec.data_format.h_axis() + 1;
    wire[0] = patch.wire_node(format!("{name}.add_dim"), AxisOp::Add(pos_data), &[wire[0]])?[0];
    let mut kernel_2d = (**kernel).clone();
    kernel_2d.insert_axis(conv.kernel_fmt.h_axis() + 1)?;
    wire[1] = patch.add_const(format!("{name}.kernel_2d"), kernel_2d)?;
    wire = patch.wire_node(name, new, &wire)?;
    wire = patch.wire_node(format!("{name}.rm_dim"), AxisOp::Rm(pos_data), &wire)?;
    patch.shunt_outside(model, node.id.into(), wire[0])?;
    Ok(Some(patch))
}
