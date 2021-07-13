use crate::internal::*;
use tract_core::ops::nn::Reduce;

register_all!(Reduce: pulsify);

fn pulsify(
    op: &Reduce,
    _source: &TypedModel,
    node: &TypedNode,
    target: &mut PulsedModel,
    mapping: &HashMap<OutletId, OutletId>,
    _pulse: usize,
) -> TractResult<Option<TVec<OutletId>>> {
    let input = mapping[&node.inputs[0]];
    let axis = target.outlet_fact(input)?.axis;
    if op.axes.contains(&axis) {
        bail!("Can not reduce over streaming axis");
    }
    Ok(Some(target.wire_node(&*node.name, op.clone(), &[input])?))
}

impl PulsedOp for Reduce {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let mut fact = inputs[0].clone();
        for &ax in &self.axes {
            fact.shape.set(ax, 1.to_dim());
        }
        Ok(tvec!(fact))
    }

    as_op!();
    pulsed_op_to_typed_op!();
}
