use tract_core::internal::*;
use tract_core::ops::change_axes::AxisOp;
use tract_transformers::ops::scaled_masked_softmax::ScaledMaskedSoftmax;

use crate::rule_ensure;

/// Drops the axes of extent one that a scaled masked softmax's scores and mask
/// share, and puts them back on the output.
///
/// The kernels address a bounded number of axes -- five -- and a batched,
/// blockified attention reaches six, of which two carry nothing. The reduction
/// is the last axis, so any other axis of extent one on both operands is exact
/// to drop. The mask's region of interest is keyed to its axis positions, so a
/// mask carrying one keeps its rank.
pub fn drop_unit_axes_of_scaled_masked_softmax(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    node_name: &str,
    op: &ScaledMaskedSoftmax,
) -> TractResult<Option<TypedModelPatch>> {
    let facts = model.node_input_facts(node.id)?;
    let (scores, mask) = (facts[0], facts[1]);
    rule_ensure!(scores.rank() == mask.rank());
    rule_ensure!(mask.uniform_tdim.is_none());
    let dropped: TVec<usize> = (0..scores.rank().saturating_sub(1))
        .filter(|&axis| scores.shape[axis].is_one() && mask.shape[axis].is_one())
        .collect();
    rule_ensure!(!dropped.is_empty());

    let mut patch = TypedModelPatch::default();
    let mut wire = patch.taps(model, &node.inputs)?;
    for (operand, side) in wire.iter_mut().zip(["scores", "mask"]) {
        for axis in dropped.iter().rev() {
            *operand = patch.wire_node(
                format!("{node_name}.{side}_rm_{axis}"),
                AxisOp::Rm(*axis),
                &[*operand],
            )?[0];
        }
    }
    let mut out = patch.wire_node(node_name, op.clone(), &wire)?[0];
    for axis in dropped.iter() {
        out = patch.wire_node(format!("{node_name}.add_{axis}"), AxisOp::Add(*axis), &[out])?[0];
    }
    patch.shunt_outside(model, node.id.into(), out)?;
    Ok(Some(patch))
}
