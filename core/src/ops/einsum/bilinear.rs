use super::EinSum;
use crate::internal::*;

pub struct BilinearEinSum<'a> {
    pub op: &'a EinSum,
    pub a_fact: &'a TypedFact,
    pub b_fact: &'a TypedFact,
    pub k_axes: TVec<char>,
    pub m_axes: TVec<char>,
    pub n_axes: TVec<char>,
}

impl<'m> BilinearEinSum<'m> {
    pub fn from_einsum(
        model: &'m TypedModel,
        node: &'m TypedNode,
    ) -> TractResult<BilinearEinSum<'m>> {
        let Some(op) = node.op_as::<EinSum>() else { bail!("Op is not an Einsum") };
        if (op.q_params.is_none() && node.inputs.len() != 2)
            || (op.q_params.is_some() && node.inputs.len() != 9)
        {
            bail!("Wrong number of input for bilinear einsum");
        }
        let input_facts = model.node_input_facts(node.id)?;
        let [a_fact, b_fact] = [input_facts[0], input_facts[1]];
        for axis in op.axes.iter_all_axes() {
            if axis.inputs[0].len() > 1 || axis.inputs[1].len() > 1 || axis.outputs[0].len() > 1 {
                bail!("Multiple occurency of axis {} in the same interface", axis.repr);
            }
        }
        let relevant = |axis: &Axis, slot: usize| {
            axis.inputs[slot]
                .get(0)
                .map(|pos| &input_facts[slot].shape[*pos])
                .is_some_and(|d| !d.is_one())
        };
        let k_axes = op
            .axes
            .iter_all_axes()
            .filter(|axis| axis.outputs[0].len() == 0 && relevant(axis, 0) && relevant(axis, 1))
            .map(|axis| axis.repr)
            .collect();
        let m_axes = op
            .axes
            .iter_all_axes()
            .filter(|axis| axis.outputs[0].len() == 1 && relevant(axis, 0) && !relevant(axis, 0))
            .map(|axis| axis.repr)
            .collect();
        let n_axes = op
            .axes
            .iter_all_axes()
            .filter(|axis| axis.outputs[0].len() == 1 && relevant(axis, 1) && !relevant(axis, 0))
            .map(|axis| axis.repr)
            .collect();
        Ok(BilinearEinSum { op, a_fact, b_fact, k_axes, m_axes, n_axes })
    }
}
