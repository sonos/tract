use crate::kernels::bin_ops::wgpu_bin_op;
use crate::kernels::deconv::wgpu_deconv_dispatch;
use crate::kernels::shaders::ChainStep;
use tract_core::internal::*;
use tract_core::ops::cnn::Deconv;
use tract_gpu::ops::change_axes::GpuAxisOp;
use tract_gpu::tensor::DeviceTensorExt;

pub fn wire_wgpu_deconv(
    source: &TypedModel,
    node: &TypedNode,
    target: &mut TypedModel,
    inputs: &[OutletId],
    op: &Deconv,
) -> TractResult<TVec<OutletId>> {
    let facts = source.node_input_facts(node.id)?;
    let prefix = &node.name;
    let has_bias = facts.len() > 2;
    let need_bias = has_bias
        && !(facts[2].konst.is_some() && facts[2].konst.as_ref().unwrap().is_all_zero()?);
    let name = format!("{prefix}.deconv");
    let mut wire = target.wire_node(
        if need_bias { &name } else { &node.name },
        WgpuDeconv { op: op.clone(), epilogue: vec![] },
        &inputs[0..2],
    )?[0];
    if need_bias {
        let data_shape = op.pool_spec.data_format.shape(&facts[0].shape)?;
        let mut needed_shape = tvec![1.to_dim(); node.outputs[0].fact.rank()];
        needed_shape[data_shape.c_axis()] = facts[2].shape.volume();
        let reshaped = target.wire_node(
            format!("{prefix}.bias_reshaped"),
            GpuAxisOp::new(AxisOp::Reshape(0, facts[2].shape.to_tvec(), needed_shape)),
            &[inputs[2]],
        )?[0];
        wire = target.wire_node(
            prefix,
            wgpu_bin_op(Box::new(tract_core::ops::math::Add)),
            &[wire, reshaped],
        )?[0];
    }
    Ok(tvec!(wire))
}

/// Inputs past the data and the weights belong to the epilogue, one per
/// binary step, as for [`crate::ops::conv::WgpuConv`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WgpuDeconv {
    pub op: Deconv,
    pub epilogue: Vec<ChainStep>,
}

impl Op for WgpuDeconv {
    fn name(&self) -> StaticName {
        "WgpuConvTranspose".into()
    }
    fn info(&self) -> TractResult<Vec<String>> {
        let mut info = self.op.info()?;
        if !self.epilogue.is_empty() {
            info.push(format!("epilogue: {:?}", self.epilogue));
        }
        Ok(info)
    }
    op_as_typed_op!();
}

impl EvalOp for WgpuDeconv {
    op_out_of_plan!();

    fn eval(&self, ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let inputs =
            inputs.iter().map(|it| it.to_device_tensor()).collect::<TractResult<TVec<_>>>()?;
        let output_shape = tract_core::ops::cnn::deconv::output_shape(
            &self.op.pool_spec,
            inputs[0].shape(),
            &self.op.adjustments,
        )?;
        let output = tract_gpu::turn_handler::make_tensor_for_node(
            ctx,
            inputs[0].datum_type(),
            &output_shape,
        )?;
        if output.len() > 0 {
            wgpu_deconv_dispatch(
                &self.op,
                &self.epilogue,
                inputs[0],
                inputs[1],
                &inputs[2..],
                &output,
            )?;
        }
        Ok(tvec!(output.into_tensor().into_tvalue()))
    }
}

impl TypedOp for WgpuDeconv {
    as_op!();
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        tract_gpu::utils::facts_to_device_facts(inputs, |facts| {
            let zero = facts[0].datum_type.scalar_fact();
            let mut facts: TVec<&TypedFact> = facts[..2].into();
            facts.push(&zero);
            self.op.output_facts(&facts)
        })
        .with_context(|| "Error while computing facts for WgpuDeconv")
    }
}
