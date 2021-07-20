use crate::internal::*;
use tract_core::ops::cnn::DeconvUnary;
use tract_pulse_opl::ops::DeconvDelay;

register_all!(DeconvUnary: pulsify);

fn pulsify(
    op: &DeconvUnary,
    _source: &TypedModel,
    node: &TypedNode,
    target: &mut PulsedModel,
    mapping: &HashMap<OutletId, OutletId>,
    _pulse: usize,
) -> TractResult<Option<TVec<OutletId>>> {
    let fact = target.outlet_fact(mapping[&node.inputs[0]])?.clone();
    let deconv = target.wire_node(
        format!("{}.deconv", node.name),
        op.clone(),
        &[mapping[&node.inputs[0]]],
    )?[0];
    let overlap = overlap(fact.axis, op);
    let delay = target.wire_node(
        &node.name,
        DeconvDelay { axis: fact.axis, overlap, delay: fact.delay, input_dim: fact.dim },
        &[deconv],
    )?[0];
    Ok(Some(tvec!(delay)))
}

fn overlap(pulse_axis: usize, op: &DeconvUnary) -> usize {
    let geo_axis = pulse_axis - op.pool_spec.data_format.h_axis();
    let axis_in_kernel = match op.kernel_format {
        tract_core::ops::cnn::KernelFormat::OIHW => 2 + geo_axis,
        tract_core::ops::cnn::KernelFormat::HWIO => geo_axis,
    };
    op.kernel.shape()[axis_in_kernel] - 1
}

impl PulsedOp for DeconvUnary {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let mut fact = inputs[0].clone();
        let overlap = overlap(fact.axis, self);
        fact.dim = fact.dim + overlap;
        let pulse_len = fact.shape[fact.axis].clone();
        fact.shape.set(fact.axis, pulse_len + overlap);
        Ok(tvec!(fact))
    }

    as_op!();
    pulsed_op_to_typed_op!();
}

impl PulsedOp for DeconvDelay {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let mut fact = inputs[0].clone();
        fact.dim = fact.dim;
        let pulse_len = fact.shape[fact.axis].clone();
        fact.shape.set(fact.axis, pulse_len - self.overlap);
        Ok(tvec!(fact))
    }

    as_op!();
    pulsed_op_to_typed_op!();
}
