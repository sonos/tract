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
    _symbol: &Symbol,
    _pulse: &TDim,
) -> TractResult<Option<TVec<OutletId>>> {
    fn min_value<D: Datum + tract_core::num_traits::Bounded>() -> Tensor {
        tensor0(D::min_value())
    }
    let fact = target.outlet_fact(mapping[&node.inputs[0]])?;
    let min = dispatch_numbers!(min_value(fact.datum_type)());
    if let Some((wire, pool_spec)) =
        pulsify_pooled_input(&op.pool_spec, source, node, target, mapping, Some(min))?
    {
        Ok(Some(target.wire_node(&node.name, MaxPool { pool_spec, ..op.clone() }, &[wire])?))
    } else {
        Ok(None)
    }
}

fn pulsify_sum_pool(
    op: &SumPool,
    source: &TypedModel,
    node: &TypedNode,
    target: &mut PulsedModel,
    mapping: &HashMap<OutletId, OutletId>,
    _symbol: &Symbol,
    _pulse: &TDim,
) -> TractResult<Option<TVec<OutletId>>> {
    if let Some((wire, pool_spec)) =
        pulsify_pooled_input(&op.pool_spec, source, node, target, mapping, None)?
    {
        Ok(Some(target.wire_node(&node.name, SumPool { pool_spec, ..op.clone() }, &[wire])?))
    } else {
        Ok(None)
    }
}

impl PulsedOp for SumPool {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        pulsed_output_facts(&self.pool_spec, inputs, inputs[0].datum_type)
    }

    as_op!();
    pulsed_op_to_typed_op!();
}

impl PulsedOp for MaxPool {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let mut facts = pulsed_output_facts(&self.pool_spec, inputs, inputs[0].datum_type)?;
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
    output_dt: DatumType,
) -> TractResult<TVec<PulsedFact>> {
    let ishape = spec.data_format.shape(&inputs[0].shape)?;
    let computed = spec.padding.compute(
        ishape.hw_dims(),
        &spec.kernel_shape,
        &spec.dilations(),
        &spec.strides(),
    );
    let spatial_dims = computed.into_iter().map(|d| d.convoluted).collect::<TVec<TDim>>();
    let oshape = spec.data_format.from_n_c_hw(
        ishape.n().cloned().unwrap_or_else(|| 1.to_dim()),
        spec.output_channels.into(),
        spatial_dims,
    )?;
    let mut fact = inputs[0].clone();
    let stream = fact.stream.as_mut().unwrap();
    let input_shape = spec.data_format.shape(&*fact.shape)?;
    let geo_axis = stream.axis - input_shape.h_axis();
    let dilation = spec.dilations.as_ref().map(|d| d[geo_axis]).unwrap_or(1);
    let kernel_len = (spec.kernel_shape[geo_axis] - 1) * dilation;
    let stride = spec.strides.as_ref().and_then(|v| v.get(geo_axis).cloned()).unwrap_or(1);
    stream.delay /= stride;
    stream.dim = (stream.dim.clone() - kernel_len.to_dim()).div_ceil(stride as _);
    fact.shape = oshape.shape.into();
    fact.datum_type = output_dt;
    Ok(tvec!(fact))
}

pub fn pulsify_pooled_input(
    spec: &PoolSpec,
    _source: &TypedModel,
    node: &TypedNode,
    target: &mut PulsedModel,
    mapping: &HashMap<OutletId, OutletId>,
    padding_value: Option<Tensor>,
) -> TractResult<Option<(OutletId, PoolSpec)>> {
    let mut wire = mapping[&node.inputs[0]];
    let input_fact: PulsedFact = target.outlet_fact(wire)?.clone();
    let input_stream = input_fact.stream.as_ref().unwrap();
    let input_shape = spec.data_format.shape(input_fact.shape.clone())?;
    if Some(input_stream.axis) == input_shape.n_axis() {
        return Ok(None);
    }
    if input_stream.axis == input_shape.c_axis() {
        bail!("Can not pulsify cnn pooling ops along the input channel axis");
    }

    let geo_axis = input_stream.axis - input_shape.h_axis();
    let stride = spec.strides.as_ref().and_then(|v| v.get(geo_axis).cloned()).unwrap_or(1);
    let pulse = input_fact.pulse().unwrap();
    if !(pulse.to_owned() % (stride as i64)).is_zero() {
        bail!("Pulsification requires pulse ({}) to be a stride ({}) multiple", pulse, stride)
    }

    let dilation = spec.dilations.as_ref().map(|d| d[geo_axis]).unwrap_or(1);
    let kernel_len = (spec.kernel_shape[geo_axis] - 1) * dilation;
    let overlap = (kernel_len + 1).saturating_sub(stride);

    let computed_padding = spec.padding.compute_one(
        geo_axis,
        &input_stream.dim,
        spec.kernel_shape[geo_axis],
        spec.dilation(geo_axis),
        spec.stride(geo_axis),
    );

    let before = computed_padding.pad_before.to_usize()?;
    let early = input_stream.delay as isize + overlap as isize - before as isize;
    let mut extra_delay = if early < 0 { (-early) as usize } else { 0 };
    let delayed_input = input_stream.delay + overlap + extra_delay - before;
    let misalignment = delayed_input % stride;
    if misalignment > 0 {
        extra_delay += stride - misalignment;
    }

    if overlap > 0 || extra_delay > 0 {
        wire = target.wire_node(
            format!("{}.delay", node.name),
            tract_pulse_opl::ops::Delay::new_typed(
                &(&input_fact).into(),
                input_stream.axis,
                extra_delay,
                overlap,
            ),
            &[wire],
        )?[0];
    }

    let has_padding =
        !computed_padding.pad_before.is_zero() || !computed_padding.pad_after.is_zero();

    if has_padding {
        use tract_core::ops::array::PadMode;
        let value = if let Some(tensor) = padding_value {
            tensor.into_arc_tensor()
        } else {
            bail!("No padding value for streaming pool operation");
        };
        let op = tract_pulse_opl::ops::PulsePad {
            axis: input_stream.axis,
            before,
            after: computed_padding.pad_after,
            begin_input: input_stream.delay + extra_delay + overlap,
            end_input: input_stream.dim.clone()
                + input_stream.delay
                + extra_delay
                + overlap.to_dim(),
            mode: PadMode::Constant(value),
            overlap,
        };
        wire = target.wire_node(format!("{}.pulse-pad", node.name), op, &[wire])?[0];
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
        Ok(Some((
            wire,
            PoolSpec { padding: PaddingSpec::ExplicitOnnxPool(bef, aft, false), ..spec.clone() },
        )))
    } else {
        Ok(Some((wire, spec.clone())))
    }
}

#[cfg(test)]
mod test {
    use tract_pulse_opl::tract_core::ops::cnn::{Conv, PoolSpec};
    use tract_pulse_opl::tract_nnef::internal::*;

    use crate::model::{PulsedModel, PulsedModelExt};

    #[test]
    fn left_padded_conv_wo_delay() -> TractResult<()> {
        let mut model = TypedModel::default();
        let stream_sym = model.symbols.sym("S");
        let stream_dim = stream_sym.to_dim();
        let source = model.add_source("source", f32::fact(dims!(1, stream_dim)))?;
        let kernel = model.add_const("kernel", rctensor3(&[[[1f32, 2f32]]]))?;
        let bias = model.add_const("bias", rctensor0(0f32))?;
        let conv = model.wire_node(
            "conv",
            Conv {
                pool_spec: PoolSpec {
                    data_format: tract_core::ops::nn::DataFormat::CHW,
                    dilations: None,
                    strides: None,
                    kernel_shape: tvec![2],
                    padding: tract_core::ops::cnn::PaddingSpec::ExplicitOnnxPool(
                        tvec![1],
                        tvec![0],
                        false,
                    ),
                    input_channels: 1,
                    output_channels: 1,
                },
                kernel_fmt: tract_core::ops::cnn::KernelFormat::OIHW,
                group: 1,
                q_params: None,
            },
            &[source, kernel, bias],
        )?;
        model.set_output_outlets(&conv)?;
        let pulsed = PulsedModel::new(&model, stream_sym, &1.to_dim())?;
        let output_fact = pulsed.output_fact(0)?;
        assert_eq!(output_fact.stream.as_ref().unwrap().delay, 0);
        Ok(())
    }
}
