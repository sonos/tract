use tract_pulse_opl::tract_core::ops::matmul::mir_quant_unary::QMatMulUnary;

use crate::internal::*;

register_all!(QMatMulUnary: pulsify);

fn pulsify(
    op: &QMatMulUnary,
    _source: &TypedModel,
    node: &TypedNode,
    target: &mut PulsedModel,
    mapping: &HashMap<OutletId, OutletId>,
    _pulse: usize,
) -> TractResult<Option<TVec<OutletId>>> {
    let input = mapping[&node.inputs[0]];
    let fact = target.outlet_fact(input)?;
    if fact.axis >= fact.shape.len() - op.b_trans as usize {
        bail!("Can not pulsify QMatMulUnaryA on the k dimension");
    }
    Ok(Some(target.wire_node(&*node.name, op.clone(), &[input])?))
}
impl PulsedOp for QMatMulUnary {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let mut fact = inputs[0].clone();
        let (_m, _k, _n, c_shape) = tract_core::ops::matmul::compute_shape(
            &self.a.shape().into_iter().map(|d| d.to_dim()).collect::<TVec<_>>(),
            &inputs[0].shape,
            self.a_trans,
            self.b_trans,
            self.c_trans,
        )?;
        fact.datum_type = self.output_type;
        fact.shape = c_shape.into();
        Ok(tvec!(fact))
    }

    as_op!();
    pulsed_op_to_typed_op!();
}
