use crate::internal::*;
use tract_core::ops::matmul::MatMulUnary;

register_all!(MatMulUnary: pulsify);

fn pulsify(
    op: &MatMulUnary,
    _source: &TypedModel,
    node: &TypedNode,
    target: &mut PulsedModel,
    mapping: &HashMap<OutletId, OutletId>,
    _pulse: usize,
) -> TractResult<Option<TVec<OutletId>>> {
    let input = mapping[&node.inputs[0]];
    let fact = target.outlet_fact(input)?;
    ensure!(fact.axis == op.axes.b_n, "Can only pulsify QMatMulUnaryA on the n dimension");
    Ok(Some(target.wire_node(&*node.name, op.clone(), &[input])?))
}
impl PulsedOp for MatMulUnary {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let mut fact = inputs[0].clone();
        let (_m, _k, _n, c_shape) = tract_core::ops::matmul::compute_shape(
            &self.a.shape().into_iter().map(|d| d.to_dim()).collect::<TVec<_>>(),
            &inputs[0].shape,
            self.axes,
        )?;
        fact.datum_type = tract_core::ops::matmul::output_type(inputs[0].datum_type);
        fact.shape = c_shape.into();
        Ok(tvec!(fact))
    }

    as_op!();
    pulsed_op_to_typed_op!();
}
