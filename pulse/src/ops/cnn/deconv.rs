use crate::internal::*;
use tract_core::num_traits::Zero;
use tract_core::ops::cnn::DeconvUnary;
use tract_core::ops::cnn::PaddingSpec;
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
    let c_axis = op.pool_spec.data_format.shape(&fact.shape)?.c_axis();
    if c_axis == fact.axis {
        bail!("Pulsification on C axis is not supported");
    }
    let pulse = fact.pulse();
    let geo_axis = fact.axis - op.pool_spec.data_format.h_axis();
    let stride = op.pool_spec.stride(geo_axis);
    let mut pulse_op = op.clone();
    pulse_op.adjustments[geo_axis] = stride - 1;
    pulse_op.pool_spec.padding = PaddingSpec::Valid;
    let deconv =
        target.wire_node(format!("{}.deconv", node.name), pulse_op, &[mapping[&node.inputs[0]]])?
            [0];
    let overlap = overlap(fact.axis, op);
    let deconv_input_dim = (fact.dim.clone() - 1) * stride + 1;
    let output_shape = tract_core::ops::cnn::deconv::output_shape(
        &op.pool_spec,
        &fact.streaming_shape(),
        &op.adjustments,
    )?;
    let kernel_spatial_shape = match op.kernel_format {
        tract_core::ops::cnn::KernelFormat::OIHW => &op.kernel.shape()[2..],
        tract_core::ops::cnn::KernelFormat::HWIO => &op.kernel.shape()[..op.kernel.rank() - 2],
    };
    let shape = op.pool_spec.data_format.shape(fact.streaming_shape())?;
    let paddings = op.pool_spec.padding.compute_for_deconv(
        &shape.hw_dims(),
        kernel_spatial_shape,
        &op.pool_spec.dilations(),
        &op.pool_spec.strides(),
        &op.adjustments,
    )?;
    let mut wire = target.wire_node(
        &node.name,
        DeconvDelay {
            axis: fact.axis,
            overlap,
            delay: paddings[geo_axis].pad_before.to_usize()? + fact.delay,
            deconv_input_dim,
            stride,
            pulse,
            deconv_output_dim: output_shape[fact.axis].clone(),
        },
        &[deconv],
    )?;

    for (geo_axis, padding) in paddings.iter().enumerate() {
        if !padding.pad_before.is_zero() || !padding.pad_after.is_zero() {
            let axis = geo_axis + shape.h_axis();
            if axis == fact.axis {
                continue;
            };
            let op = crate::model::PulseWrappingOp(Box::new(tract_core::ops::array::Slice::new(
                axis,
                padding.pad_before.clone(),
                padding.deconvoluted.clone() + &padding.pad_before,
            )));
            wire = target.wire_node(format!("{}.padding.{}", node.name, geo_axis), op, &wire)?;
        }
    }
    Ok(Some(wire))
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
        let mut output_shape = tract_core::ops::cnn::deconv::output_shape(
            &self.pool_spec,
            &fact.streaming_shape(),
            &self.adjustments,
        )?;
        fact.dim = output_shape[fact.axis].clone();
        let pulse_len = fact.shape[fact.axis].clone() * stride;
        output_shape[fact.axis] = pulse_len + overlap;
        fact.shape = output_shape.into();
        if let Some(c) = self.pool_spec.output_channel_override {
            let c_axis = self.pool_spec.data_format.shape(&fact.shape)?.c_axis();
            fact.shape.set(c_axis, c.to_dim())
        }
        Ok(tvec!(fact))
    }

    as_op!();
    pulsed_op_to_typed_op!();
}

impl PulsedOp for DeconvDelay {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let mut fact = inputs[0].clone();
        fact.dim = self.deconv_output_dim.clone();
        let pulse_len = fact.shape[fact.axis].clone();
        fact.shape.set(fact.axis, pulse_len - self.overlap);
        fact.delay = self.delay;
        Ok(tvec!(fact))
    }

    as_op!();
    pulsed_op_to_typed_op!();
}
