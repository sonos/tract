use tract_hir::internal::*;
use tract_hir::ops::cnn::ConvUnary;
use tract_hir::ops::nn::DataFormat;

pub fn rewrite_for_tflite(model: &mut TypedModel) -> TractResult<()> {
    Rewriter::default()
        .with_rule_for::<ConvUnary>("nchw-to-nhwc", Box::new(nchw_to_nhwc))
        .rewrite(&(), model)
}

fn nchw_to_nhwc(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
) -> TractResult<Option<TypedModelPatch>> {
    let conv = node.op_as::<ConvUnary>().context("wrong op")?;
    let name = &node.name;
    ensure!(conv.pool_spec.rank() == 2);
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
    if !conv.pool_spec.data_format.c_is_last() {
        let mut new = conv.clone();
        new.pool_spec.data_format = DataFormat::NHWC;
        let mut patch = TypedModelPatch::default();
        let mut wire = tvec!(patch.tap_model(model, node.inputs[0])?);
        wire = patch.wire_node(format!("{name}.nhwc"), AxisOp::Move(1, 3), &wire)?;
        wire = patch.wire_node(name, new, &wire)?;
        wire = patch.wire_node(format!("{name}.nchw"), AxisOp::Move(3, 1), &wire)?;
        patch.shunt_outside(model, node.id.into(), wire[0])?;
        return Ok(Some(patch));
    }
    Ok(None)
}
