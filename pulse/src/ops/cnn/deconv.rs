use crate::internal::*;
use tract_core::num_traits::Zero;
use tract_core::ops::cnn::Deconv;
use tract_core::ops::cnn::PaddingSpec;
use tract_pulse_opl::ops::DeconvDelay;
use tract_pulse_opl::ops::PulseMask;

register_all!(Deconv: pulsify);

fn pulsify(
    op: &Deconv,
    source: &TypedModel,
    node: &TypedNode,
    target: &mut PulsedModel,
    mapping: &HashMap<OutletId, OutletId>,
    _symbol: &Symbol,
    _pulse: &TDim,
) -> TractResult<Option<TVec<OutletId>>> {
    let fact = target.outlet_fact(mapping[&node.inputs[0]])?.clone();
    let pulse = fact.pulse().unwrap();
    let stream = fact.stream.as_ref().unwrap();
    let c_axis = op.pool_spec.data_format.shape(&fact.shape)?.c_axis();
    if c_axis == stream.axis {
        bail!("Pulsification on C axis is not supported");
    }
    // general case for invariants will manage
    rule_if!(
        op.axes_mapping(&source.node_input_facts(node.id)?, &source.node_output_facts(node.id)?)?
            .axis((InOut::In(0), stream.axis))?
            .outputs[0]
            .len()
            != 1
    );
    let geo_axis = stream.axis - op.pool_spec.data_format.h_axis();
    let stride = op.pool_spec.stride(geo_axis);
    let mut pulse_op = op.clone();
    pulse_op.adjustments[geo_axis] = stride - 1;
    pulse_op.pool_spec.padding = PaddingSpec::Valid;
    let mut wire = tvec![mapping[&node.inputs[0]]];
    let mask = PulseMask {
        axis: stream.axis,
        begin: stream.delay,
        end: stream.dim.clone() + stream.delay,
        value: Tensor::zero_scalar_dt(fact.datum_type)?,
    };
    wire = target.wire_node(format!("{}.mask", node.name), mask, &wire)?;
    wire.push(mapping[&node.inputs[1]]);
    // Feed a zero bias to the per-pulse Deconv. With kernel > stride the
    // per-pulse output has overlap slots that the bulk Deconv would never
    // emit; DeconvDelay's overlap-add then double-counts the bias on
    // those slots. Adding the original bias back after DeconvDelay
    // guarantees it's added exactly once per output position.
    let source_bias_fact = source.outlet_fact(node.inputs[2])?.clone();
    let bias_tensor = source_bias_fact
        .konst
        .clone()
        .context("Deconv bias must be a constant for pulsification")?;
    let zero_bias = Tensor::zero_dt(bias_tensor.datum_type(), bias_tensor.shape())?;
    let zero_bias = target.add_const(format!("{}.zero_bias", node.name), zero_bias)?;
    wire.push(zero_bias);
    wire = target.wire_node(format!("{}.deconv", node.name), pulse_op, &wire)?;
    let overlap = overlap(stream.axis, op);
    let deconv_input_dim = (stream.dim.clone() - 1) * stride + 1;
    let output_shape = tract_core::ops::cnn::deconv::output_shape(
        &op.pool_spec,
        &fact.streaming_shape(),
        &op.adjustments,
    )?;
    let kernel_spatial_shape = &op.pool_spec.kernel_shape;
    let shape = op.pool_spec.data_format.shape(fact.streaming_shape())?;
    let paddings = op.pool_spec.padding.compute_for_deconv(
        shape.hw_dims(),
        kernel_spatial_shape,
        &op.pool_spec.dilations(),
        &op.pool_spec.strides(),
        &op.adjustments,
    )?;
    wire = target.wire_node(
        &node.name,
        DeconvDelay {
            axis: stream.axis,
            overlap,
            delay: paddings[geo_axis].pad_before.to_usize()? + stream.delay,
            deconv_input_dim,
            stride,
            pulse: pulse.to_owned(),
            deconv_output_dim: output_shape[stream.axis].clone(),
        },
        &wire,
    )?;

    for (geo_axis, padding) in paddings.iter().enumerate() {
        if !padding.pad_before.is_zero() || !padding.pad_after.is_zero() {
            let axis = geo_axis + shape.h_axis();
            if axis == stream.axis {
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

    // Add the original bias to the now-merged output. See the zero-bias
    // comment above pulse_op wiring for why this can't be done inside
    // per-pulse Deconv.
    let out_shape = target.outlet_fact(wire[0])?.shape.clone();
    let out_rank = out_shape.rank();
    let c_axis = op.pool_spec.data_format.shape(out_shape.to_tvec())?.c_axis();
    let mut reshaped_bias_shape: TVec<usize> = tvec![1; out_rank];
    reshaped_bias_shape[c_axis] = op.pool_spec.output_channels;
    // Broadcast the bias to ``(1, ..., C, ..., 1)`` so the post-Deconv
    // Add is a clean elementwise op. Bulk Deconv accepts both scalar and
    // rank-1 ``(C,)`` biases (handled by ``wire_reshape_bias_for_bin``);
    // we mirror that here at build time.
    let bias_const = if bias_tensor.rank() == 0 {
        bias_tensor.broadcast_scalar_to_shape(&reshaped_bias_shape)?.into_arc_tensor()
    } else if bias_tensor.shape() == [op.pool_spec.output_channels] {
        bias_tensor.clone().into_tensor().into_shape(&reshaped_bias_shape)?.into_arc_tensor()
    } else if bias_tensor.shape() == &*reshaped_bias_shape {
        bias_tensor.clone()
    } else {
        bail!(
            "Unexpected Deconv bias shape {:?} for {} output channels",
            bias_tensor.shape(),
            op.pool_spec.output_channels
        );
    };
    let bias = target.add_const(format!("{}.bias", node.name), bias_const)?;
    wire = target.wire_node(
        format!("{}.add_bias", node.name),
        crate::model::PulseWrappingOp(Box::new(tract_core::ops::math::add())),
        &[wire[0], bias],
    )?;

    Ok(Some(wire))
}

fn overlap(pulse_axis: usize, op: &Deconv) -> usize {
    let geo_axis = pulse_axis - op.pool_spec.data_format.h_axis();
    (op.pool_spec.kernel_shape[geo_axis] - 1) * op.pool_spec.dilation(geo_axis)
}

impl PulsedOp for Deconv {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let mut fact = inputs[0].clone();
        let stream = fact.stream.as_mut().unwrap();
        let overlap = overlap(stream.axis, self);
        let geo_axis = stream.axis - self.pool_spec.data_format.h_axis();
        let stride = self.pool_spec.stride(geo_axis);
        let mut output_shape = tract_core::ops::cnn::deconv::output_shape(
            &self.pool_spec,
            &inputs[0].streaming_shape(),
            &self.adjustments,
        )?;
        stream.dim = output_shape[stream.axis].clone();
        let pulse_len = fact.shape[stream.axis].clone() * stride;
        output_shape[stream.axis] = pulse_len + overlap;
        let c_axis = self.pool_spec.data_format.shape(&output_shape)?.c_axis();
        output_shape[c_axis] = self.pool_spec.output_channels.into();
        fact.shape = output_shape.into();
        Ok(tvec!(fact))
    }

    as_op!();
    pulsed_op_to_typed_op!();
}

impl PulsedOp for DeconvDelay {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let mut fact = inputs[0].clone();
        let stream = fact.stream.as_mut().unwrap();
        stream.dim = self.deconv_output_dim.clone();
        let pulse_len = fact.shape[stream.axis].clone();
        fact.shape.set(stream.axis, pulse_len - self.overlap);
        stream.delay = self.delay;
        Ok(tvec!(fact))
    }

    as_op!();
    pulsed_op_to_typed_op!();
}
