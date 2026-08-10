use tract_core::internal::*;
use tract_core::ops::array::{Slice, TypedConcat};
use tract_core::ops::change_axes::AxisOp;

/// Search pattern => CONCAT(SLICE(x, a..b), SLICE(x, b..c), ...) on the slice
/// axis, each slice optionally routed through the SAME axis op (the qwen
/// partial-rotary export transposes both rope halves before rebuilding the
/// head): adjacent slices of one source reassembled in order are just the
/// wider SLICE(x, a..c) (+ that axis op once). Declutter's PushSliceUp
/// leaves this shape behind when a downstream boundary split a tensor that
/// later rules re-fused (`fuse_rms_norm_split_scale` re-slicing a fused
/// norm, the rotate-half concat-pair detection): collapsing it removes one
/// copy dispatch per slice plus the concat itself on GPU backends. Pure data
/// movement, bit-exact by construction.
pub fn collapse_adjacent_slice_concat(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    node_name: &str,
    op: &TypedConcat,
) -> TractResult<Option<TypedModelPatch>> {
    if std::env::var_os("TRACT_GPU_DISABLE_SLICE_CONCAT_COLLAPSE").is_some() {
        return Ok(None);
    }
    rule_if!(node.inputs.len() >= 2);

    let mut src: Option<OutletId> = None;
    let mut shared_axis_op: Option<&AxisOp> = None;
    let mut slice_axis: Option<usize> = None;
    let mut bounds: Vec<(usize, usize)> = Vec::with_capacity(node.inputs.len());
    for (ix, input) in node.inputs.iter().enumerate() {
        // Optional shared axis op between each slice and the concat; it must
        // be the same op on every branch, and each branch must be that op's
        // only consumer chain.
        let mut branch = &model.nodes()[input.node];
        if let Some(axis_op) = branch.op_as::<AxisOp>() {
            // Only shape-blind axis ops: a Reshape hardcodes the sliced
            // axis's length and would not apply to the wider slice.
            rule_if!(matches!(
                axis_op,
                AxisOp::Move(_, _) | AxisOp::Add(_) | AxisOp::Rm(_)
            ));
            rule_if!(if ix == 0 {
                shared_axis_op = Some(axis_op);
                true
            } else {
                shared_axis_op == Some(axis_op)
            });
            branch = &model.nodes()[branch.inputs[0].node];
        } else {
            rule_if!(shared_axis_op.is_none());
        }
        rule_if_some!(slice = branch.op_as::<Slice>());
        // Every branch slices the same source on the same axis, and that
        // axis must land on the concat axis after the shared axis op.
        rule_if!(slice_axis.is_none_or(|a| a == slice.axis));
        slice_axis = Some(slice.axis);
        let mapped = match shared_axis_op {
            None => Some(slice.axis),
            Some(a) => a.transform_axis(slice.axis),
        };
        rule_if!(mapped == Some(op.axis));
        rule_if_some!(start = slice.start.to_usize().ok());
        rule_if_some!(end = slice.end.to_usize().ok());
        rule_if!(src.is_none_or(|s| s == branch.inputs[0]));
        src = Some(branch.inputs[0]);
        bounds.push((start, end));
    }
    // Concat order must chain the ranges without gap or overlap.
    rule_if!(bounds.windows(2).all(|w| w[0].1 == w[1].0));

    let src = src.unwrap();
    let slice_axis = slice_axis.unwrap();
    let mut patch = TypedModelPatch::default();
    let tap = patch.tap_model(model, src)?;
    let mut out = patch.wire_node(
        format!("{node_name}.collapsed-slice"),
        Slice::new(slice_axis, bounds[0].0, bounds[bounds.len() - 1].1),
        &[tap],
    )?;
    if let Some(axis_op) = shared_axis_op {
        out = patch.wire_node(
            format!("{node_name}.collapsed-slice.axes"),
            axis_op.clone(),
            &out,
        )?;
    }
    patch.shunt_outside(model, node.id.into(), out[0])?;
    Ok(Some(patch))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(ranges: &[(usize, usize)]) -> TractResult<TypedModel> {
        let mut model = TypedModel::default();
        let input = model.add_source("input", f32::datum_type().fact([2usize, 8]))?;
        let mut slices = tvec![];
        for (ix, (start, end)) in ranges.iter().enumerate() {
            slices.push(
                model.wire_node(format!("slice.{ix}"), Slice::new(1, *start, *end), &[input])?[0],
            );
        }
        let out = model.wire_node("concat", TypedConcat { axis: 1 }, &slices)?;
        model.select_output_outlets(&out)?;
        Ok(model)
    }

    fn run_rule(model: &mut TypedModel) -> TractResult<()> {
        Rewriter::default()
            .with_rule_for("collapse_adjacent_slice_concat", collapse_adjacent_slice_concat)
            .rewrite(&(), model)
    }

    fn eval(model: &TypedModel) -> TractResult<TVec<TValue>> {
        let input =
            Tensor::from_shape(&[2, 8], &(0..16).map(|i| i as f32).collect::<Vec<_>>())?;
        SimplePlan::new(model.clone())?.run(tvec![input.into_tvalue()])
    }

    #[test]
    fn adjacent_slices_collapse() -> TractResult<()> {
        let mut model = model(&[(1, 3), (3, 6)])?;
        let reference = eval(&model)?;
        run_rule(&mut model)?;
        ensure!(!model.nodes().iter().any(|n| n.op_is::<TypedConcat>()));
        ensure!(model.nodes().iter().filter(|n| n.op_is::<Slice>()).count() == 1);
        let got = eval(&model)?;
        ensure!(reference[0] == got[0]);
        Ok(())
    }

    #[test]
    fn out_of_order_slices_do_not_collapse() -> TractResult<()> {
        // Concat order b..c, a..b is a swap, not the wider slice.
        let mut model = model(&[(3, 6), (1, 3)])?;
        run_rule(&mut model)?;
        ensure!(model.nodes().iter().any(|n| n.op_is::<TypedConcat>()));
        Ok(())
    }

    #[test]
    fn adjacent_slices_through_shared_move_axis_collapse() -> TractResult<()> {
        // The qwen rope shape: both halves transposed identically before the
        // concat. Move(1,0) sends the sliced axis 1 to 0; concat over 0.
        let mut model = TypedModel::default();
        let input = model.add_source("input", f32::datum_type().fact([2usize, 8]))?;
        let mut moved = tvec![];
        for (ix, (start, end)) in [(1usize, 3usize), (3, 6)].iter().enumerate() {
            let sliced = model.wire_node(
                format!("slice.{ix}"),
                Slice::new(1, *start, *end),
                &[input],
            )?;
            moved.push(
                model.wire_node(format!("move.{ix}"), AxisOp::Move(1, 0), &sliced)?[0],
            );
        }
        let out = model.wire_node("concat", TypedConcat { axis: 0 }, &moved)?;
        model.select_output_outlets(&out)?;

        let reference = eval(&model)?;
        run_rule(&mut model)?;
        ensure!(!model.nodes().iter().any(|n| n.op_is::<TypedConcat>()));
        ensure!(model.nodes().iter().filter(|n| n.op_is::<Slice>()).count() == 1);
        ensure!(model.nodes().iter().filter(|n| n.op_is::<AxisOp>()).count() == 1);
        let got = eval(&model)?;
        ensure!(reference[0] == got[0]);
        Ok(())
    }
}
