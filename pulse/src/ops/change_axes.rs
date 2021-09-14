use crate::internal::*;
use tract_core::ops::change_axes::AxisOp;

register_all!(AxisOp: pulsify);

fn pulsify(
    op: &AxisOp,
    _source: &TypedModel,
    node: &TypedNode,
    target: &mut PulsedModel,
    mapping: &HashMap<OutletId, OutletId>,
    _pulse: usize,
) -> TractResult<Option<TVec<OutletId>>> {
    let input = mapping[&node.inputs[0]];
    Ok(Some(target.wire_node(&*node.name, op.clone(), &[input])?))
}

impl PulsedOp for AxisOp {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let mut fact = inputs[0].clone();
        fact.shape = inputs[0].shape.clone();
        self.change_shape(&mut fact.shape, false)?;
        fact.axis = self
            .transform_axis(fact.axis)
            .ok_or_else(|| format_err!("Invalid axis for pulsification"))?;
        Ok(tvec!(fact))
    }

    as_op!();
    pulsed_op_to_typed_op!();
}
