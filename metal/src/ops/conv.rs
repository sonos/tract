use crate::kernels::conv::metal_conv_dispatch;
use tract_core::internal::*;
use tract_core::ops::OpStateFreeze;
use tract_core::ops::cnn::Conv;
use tract_gpu::ops::change_axes::GpuAxisOp;
use tract_gpu::tensor::DeviceTensorExt;

pub fn wire_metal_conv(
    source: &TypedModel,
    node: &TypedNode,
    target: &mut TypedModel,
    inputs: &[OutletId],
    op: &Conv,
) -> TractResult<TVec<OutletId>> {
    let facts = source.node_input_facts(node.id)?;
    let data_shape = op.pool_spec.data_format.shape(&facts[0].shape)?;
    let prefix = &node.name;
    let bias = &facts[2];
    let need_bias = !(bias.konst.is_some() && bias.konst.as_ref().unwrap().is_all_zero()?);
    let conv_name = format!("{prefix}.conv");
    let mut conv_wire = target.wire_node(
        if need_bias { &conv_name } else { &node.name },
        MetalConv { op: op.clone() },
        &inputs[0..2],
    )?[0];
    if need_bias {
        let mut needed_shape = tvec![1.to_dim(); node.outputs[0].fact.rank()];
        needed_shape[data_shape.c_axis()] = op.pool_spec.output_channels.to_dim();
        let reshaped = target.wire_node(
            format!("{prefix}.bias_reshaped"),
            GpuAxisOp::new(AxisOp::Reshape(0, bias.shape.to_tvec(), needed_shape)),
            &[inputs[2]],
        )?[0];
        conv_wire = target.wire_node(
            prefix,
            crate::kernels::bin_ops::metal_bin_op(Box::new(tract_core::ops::math::Add)),
            &[conv_wire, reshaped],
        )?[0];
    }
    Ok(tvec!(conv_wire))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetalConv {
    pub op: Conv,
}

impl Op for MetalConv {
    fn name(&self) -> StaticName {
        "MetalConv".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        self.op.info()
    }

    op_as_typed_op!();
}

impl EvalOp for MetalConv {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        node_id: usize,
        session: &TurnState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let inputs =
            inputs.iter().map(|it| it.to_device_tensor()).collect::<TractResult<TVec<_>>>()?;
        let output_shape = self.op.pool_spec.output_shape(inputs[0].shape())?;
        let output = tract_gpu::session_handler::make_tensor_for_node(
            session,
            node_id,
            inputs[0].datum_type(),
            &output_shape.shape,
        )?;

        if output.len() > 0 {
            crate::with_metal_stream(|stream| {
                metal_conv_dispatch(
                    stream,
                    &self.op,
                    inputs[0],
                    inputs[1],
                    inputs.get(2).cloned(),
                    &output,
                )
            })?;
        }
        Ok(tvec!(output.into_tensor().into_tvalue()))
    }
}

impl TypedOp for MetalConv {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        tract_gpu::utils::facts_to_device_facts(inputs, |facts| {
            let zero = facts[0].datum_type.scalar_fact();
            let mut facts: TVec<&TypedFact> = facts.into();
            if facts.len() == 2 {
                facts.push(&zero);
            }
            self.op.output_facts(&facts)
        })
        .with_context(|| "Error while computing facts for MetalConv")
    }
}
