use crate::internal::*;
use tract_core::ops::cnn::DeconvUnary;
use tract_pulse_opl::{
    ops::DeconvDelay,
    tract_core::ops::cnn::{PaddingSpec, PoolSpec},
};

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
    let pulse = fact.pulse();
    let geo_axis = fact.axis - op.pool_spec.data_format.h_axis();
    let stride = op.pool_spec.stride(geo_axis);
    let mut pulse_op = op.clone();
    if stride > 1 {
        /*
        let geo_rank = op.pool_spec.rank();
        let padding = PaddingSpec::Explicit(
            tvec!(0; geo_rank),
            (0..geo_rank).map(|axis| (axis == geo_axis) as usize * (stride - 1)).collect(),
            false,
        );
        let pool_spec = PoolSpec::new(
            op.pool_spec.data_format,
            op.pool_spec.kernel_shape.clone(),
            padding,
            op.pool_spec.dilations.clone(),
            op.pool_spec.strides.clone(),
            op.pool_spec.output_channel_override,
        );
        pulse_op.pool_spec = pool_spec;
        */
        pulse_op.adjustments[geo_axis] += stride - 1;
    }
    let deconv =
        target.wire_node(format!("{}.deconv", node.name), pulse_op, &[mapping[&node.inputs[0]]])?
            [0];
    let overlap = overlap(fact.axis, op);
    let deconv_input_dim = (fact.dim - 1) * stride + 1;
    let delay = target.wire_node(
        &node.name,
        DeconvDelay {
            axis: fact.axis,
            overlap,
            delay: fact.delay,
            input_dim: deconv_input_dim,
            stride,
            pulse,
        },
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
    (op.kernel.shape()[axis_in_kernel] - 1) * op.pool_spec.dilation(geo_axis)
}

impl PulsedOp for DeconvUnary {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let mut fact = inputs[0].clone();
        let overlap = overlap(fact.axis, self);
        let geo_axis = fact.axis - self.pool_spec.data_format.h_axis();
        let stride = self.pool_spec.stride(geo_axis);
        fact.dim = (fact.dim - 1) * stride + 1 + overlap;
        let pulse_len = fact.shape[fact.axis].clone() * stride;
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
