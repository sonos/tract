use crate::internal::*;
use tract_core::num_traits::Zero;
use tract_core::ops::cnn::{MaxPool, PaddingSpec, PoolSpec, SumPool};

register_all!(MaxPool: pulsify_max_pool, SumPool: pulsify_sum_pool);

fn pulsify_max_pool(
    op: &MaxPool,
    source: &TypedModel,
    node: &TypedNode,
    target: &mut PulsedModel,
    mapping: &HashMap<OutletId, OutletId>,
    _pulse: usize,
) -> TractResult<TVec<OutletId>> {
    fn min_value<D: Datum + tract_core::num_traits::Bounded>() -> Tensor {
        tensor0(D::min_value())
    }
    let fact = target.outlet_fact(mapping[&node.inputs[0]])?;
    let min = dispatch_numbers!(min_value(fact.datum_type)());
    let (wire, pool_spec) = pulsify(&op.pool_spec, source, node, target, mapping, Some(min))?;
    target.wire_node(&node.name, MaxPool { pool_spec, ..op.clone() }, &[wire])
}

fn pulsify_sum_pool(
    op: &SumPool,
    source: &TypedModel,
    node: &TypedNode,
    target: &mut PulsedModel,
    mapping: &HashMap<OutletId, OutletId>,
    _pulse: usize,
) -> TractResult<TVec<OutletId>> {
    let (wire, pool_spec) = pulsify(&op.pool_spec, source, node, target, mapping, None)?;
    target.wire_node(&node.name, SumPool { pool_spec, ..op.clone() }, &[wire])
}

impl PulsedOp for SumPool {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        pulsed_output_facts(&self.pool_spec, inputs)
    }

    as_op!();
    pulsed_op_to_typed_op!();
}

impl PulsedOp for MaxPool {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let mut facts = pulsed_output_facts(&self.pool_spec, inputs)?;
        if let Some(idt) = self.with_index_outputs {
            facts.push(facts[0].clone());
            facts[1].datum_type = idt;
        }
        Ok(facts)
    }

    as_op!();
    pulsed_op_to_typed_op!();
}

pub fn pulsed_output_facts(
    spec: &PoolSpec,
    inputs: &[&PulsedFact],
) -> TractResult<TVec<PulsedFact>> {
    let ishape = spec.data_format.shape(&inputs[0].shape)?;
    let computed = spec.padding.compute(
        ishape.hw_dims(),
        &*spec.kernel_shape,
        &spec.dilations(),
        &spec.strides(),
    );
    let spatial_dims = computed.into_iter().map(|d| d.convoluted).collect::<TVec<TDim>>();
    let oshape = spec.data_format.from_n_c_hw(
        ishape.n().cloned().unwrap_or(1.to_dim()),
        spec.output_channel_override.map(|d| d.to_dim()).unwrap_or_else(|| ishape.c().clone()),
        spatial_dims,
    )?;
    let mut fact = inputs[0].clone();
    let input_shape = spec.data_format.shape(&*fact.shape)?;
    let geo_axis = fact.axis - input_shape.h_axis();
    let dilation = spec.dilations.as_ref().map(|d| d[geo_axis]).unwrap_or(1);
    let kernel_len = (spec.kernel_shape[geo_axis] - 1) * dilation;
    let stride = spec.strides.as_ref().and_then(|v| v.get(geo_axis).cloned()).unwrap_or(1);
    fact.delay /= stride;
    fact.dim = (fact.dim.clone() - kernel_len.to_dim()).div_ceil(stride as _);
    fact.shape = oshape.shape;
    Ok(tvec!(fact))
}

pub fn pulsify(
    spec: &PoolSpec,
    _source: &TypedModel,
    node: &TypedNode,
    target: &mut PulsedModel,
    mapping: &HashMap<OutletId, OutletId>,
    padding_value: Option<Tensor>,
) -> TractResult<(OutletId, PoolSpec)> {
    let mut wire = mapping[&node.inputs[0]];
    let mut fact: PulsedFact = target.outlet_fact(wire)?.clone();
    let input_shape = spec.data_format.shape(fact.shape.clone())?;
    if Some(fact.axis) == input_shape.n_axis() {
        return Ok((wire, spec.clone()));
    }
    if fact.axis == input_shape.c_axis() {
        bail!("Can not pulsify cnn pooling ops along the input channel axis");
    }

    let geo_axis = fact.axis - input_shape.h_axis();
    let stride = spec.strides.as_ref().and_then(|v| v.get(geo_axis).cloned()).unwrap_or(1);
    let pulse = fact.pulse();
    if pulse % stride != 0 {
        bail!("Pulsificaton requires pulse to be a stride multiple")
    }

    let computed_padding = spec.padding.compute_one(
        geo_axis,
        &fact.dim,
        spec.kernel_shape[geo_axis],
        spec.dilation(geo_axis),
        spec.stride(geo_axis),
    );
    let has_padding =
        !computed_padding.pad_before.is_zero() || !computed_padding.pad_after.is_zero();

    if has_padding {
        use tract_core::ops::array::PadMode;
        let value = if let Some(tensor) = padding_value {
            tensor.into_arc_tensor()
        } else {
            bail!("No padding value for streaming pool operation");
        };
        let before = computed_padding.pad_before.to_usize()?;
        let extra_delay = before.saturating_sub(fact.delay);
        if extra_delay > 0 {
            wire = target.wire_node(
                format!("{}.delay-for-pad", node.name),
                tract_pulse_opl::ops::Delay::new(fact.axis, &(&fact).into(), extra_delay, 0),
                &[wire],
            )?[0];
            fact = target.outlet_fact(wire)?.clone();
        }
        let op = tract_pulse_opl::ops::PulsePad {
            axis: fact.axis,
            pulse,
            before,
            after: computed_padding.pad_after.clone(),
            begin_input: fact.delay,
            end_input: fact.delay.to_dim() + &fact.dim,
            mode: PadMode::Constant(value),
        };
        wire = target.wire_node(format!("{}.pad", node.name), op, &[wire])?[0];
        fact = target.outlet_fact(wire)?.clone();
    }

    let dilation = spec.dilations.as_ref().map(|d| d[geo_axis]).unwrap_or(1);
    let kernel_len = (spec.kernel_shape[geo_axis] - 1) * dilation;
    let overlap = (kernel_len + 1).saturating_sub(stride);
    let misalignment = fact.delay % pulse;

    if overlap > 0 || misalignment > 0 {
        let align_to = (overlap + fact.delay).div_ceil(stride) * stride;
        let delay = align_to - overlap - fact.delay;
        wire = target.wire_node(
            format!("{}.delay", node.name),
            tract_pulse_opl::ops::Delay::new(fact.axis, &(&fact).into(), delay, overlap),
            &[wire],
        )?[0];
    }

    if has_padding {
        let mut bef = tvec!();
        let mut aft = tvec!();
        for ix in 0..input_shape.hw_rank() {
            if ix == geo_axis {
                bef.push(0);
                aft.push(0);
            } else {
                let c = spec.padding.compute_one(
                    ix,
                    &input_shape.hw_dims()[ix],
                    spec.kernel_shape[ix],
                    spec.dilations()[ix],
                    spec.strides()[ix],
                );
                bef.push(c.pad_before.to_usize()?);
                aft.push(c.pad_after.to_usize()?);
            };
        }
        Ok((wire, PoolSpec { padding: PaddingSpec::Explicit(bef, aft, true), ..spec.clone() }))
    } else {
        Ok((wire, spec.clone()))
    }
}
