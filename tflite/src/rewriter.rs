use tract_hir::internal::*;
use tract_hir::ops::cnn::ConvUnary;
use tract_hir::ops::nn::DataFormat;
use tract_hir::tract_core::ops::cnn::KernelFormat;

pub fn rewrite_for_tflite(model: &mut TypedModel) -> TractResult<()> {
    Rewriter::default()
        .with_rule_for::<ConvUnary>("kernel-in-ohwi", kernel_in_ohwi)
        .with_rule_for::<ConvUnary>("make_1d_2d", make_1d_2d)
        .with_rule_for::<ConvUnary>("force_n_axis", force_n_axis)
        .with_rule_for::<ConvUnary>("nchw-to-nhwc", nchw_to_nhwc)
        .rewrite(&(), model)
}

fn kernel_in_ohwi(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    name: &str,
    conv: &ConvUnary,
) -> TractResult<Option<TypedModelPatch>> {
    let rank = conv.kernel.rank();
    let kernel = match conv.kernel_fmt {
        KernelFormat::OIHW => conv.kernel.clone().into_tensor().move_axis(1, rank - 1)?,
        KernelFormat::HWIO => conv.kernel.clone().into_tensor().move_axis(rank - 1, 0)?,
        KernelFormat::OHWI => return Ok(None),
    };
    let mut new = conv.clone();
    new.kernel_fmt = KernelFormat::OHWI;
    new.kernel = kernel.into_arc_tensor();
    let mut patch = TypedModelPatch::default();
    let mut wire = tvec!(patch.tap_model(model, node.inputs[0])?);
    wire = patch.wire_node(name, new, &wire)?;
    patch.shunt_outside(model, node.id.into(), wire[0])?;
    Ok(Some(patch))
}

fn force_n_axis(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    name: &str,
    conv: &ConvUnary,
) -> TractResult<Option<TypedModelPatch>> {
    if !conv.pool_spec.data_format.has_n() {
        let mut new = conv.clone();
        new.pool_spec.data_format = conv.pool_spec.data_format.with_n();
        let mut patch = TypedModelPatch::default();
        let mut wire = tvec!(patch.tap_model(model, node.inputs[0])?);
        wire = patch.wire_node(format!("{name}.add_n"), AxisOp::Add(0), &wire)?;
        wire = patch.wire_node(name, new, &wire)?;
        wire = patch.wire_node(format!("{name}.rm_n"), AxisOp::Rm(0), &wire)?;
        patch.shunt_outside(model, node.id.into(), wire[0])?;
        return Ok(Some(patch));
    }
    Ok(None)
}

fn make_1d_2d(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    name: &str,
    conv: &ConvUnary,
) -> TractResult<Option<TypedModelPatch>> {
    if conv.pool_spec.rank() == 1 {
        let pos = conv.pool_spec.data_format.h_axis();
        let mut new = conv.clone();
        new.pool_spec.kernel_shape.insert(0, 1);
        new.pool_spec.dilations.iter_mut().for_each(|dil| dil.insert(0, 1));
        new.pool_spec.strides.iter_mut().for_each(|dil| dil.insert(0, 1));
        let mut kernel = new.kernel.clone().into_tensor();
        kernel.insert_axis(conv.kernel_fmt.h_axis())?;
        new.kernel = kernel.into_arc_tensor();
        let mut patch = TypedModelPatch::default();
        let mut wire = tvec!(patch.tap_model(model, node.inputs[0])?);
        wire = patch.wire_node(format!("{name}.add_dim"), AxisOp::Add(pos), &wire)?;
        wire = patch.wire_node(name, new, &wire)?;
        wire = patch.wire_node(format!("{name}.rm_dim"), AxisOp::Rm(pos), &wire)?;
        patch.shunt_outside(model, node.id.into(), wire[0])?;
        return Ok(Some(patch));
    }
    Ok(None)
}

fn nchw_to_nhwc(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    name: &str,
    conv: &ConvUnary,
) -> TractResult<Option<TypedModelPatch>> {
    if !conv.pool_spec.data_format.c_is_last() {
        let mut new = conv.clone();
        new.pool_spec.data_format = DataFormat::NHWC;
        let mut patch = TypedModelPatch::default();
        let mut wire = tvec!(patch.tap_model(model, node.inputs[0])?);
        wire = patch.wire_node(format!("{name}.nhwc"), AxisOp::Move(1, conv.pool_spec.rank() + 1), &wire)?;
        wire = patch.wire_node(name, new, &wire)?;
        wire = patch.wire_node(format!("{name}.nchw"), AxisOp::Move(conv.pool_spec.rank() + 1, 1), &wire)?;
        patch.shunt_outside(model, node.id.into(), wire[0])?;
        return Ok(Some(patch));
    }
    Ok(None)
}
