use crate::kernels::bin_ops::wgpu_bin_op;
use crate::kernels::conv::{DirectTile, measured_packed_tile, wgpu_conv_dispatch};
use tract_core::internal::*;
use tract_core::ops::cnn::{Conv, KernelFormat};
use tract_core::ops::konst::Const;
use tract_core::ops::nn::DataFormat;
use tract_gpu::ops::change_axes::GpuAxisOp;
use tract_gpu::sync::{DeviceSyncKind, sync_inputs_if_required};
use tract_gpu::tensor::DeviceTensorExt;

use crate::transform::Mapping;

/// Translates a float, non-quantized 2-D `Conv` through [`wire_wgpu_conv`].
pub fn translate_conv(
    source: &TypedModel,
    node: &TypedNode,
    target: &mut TypedModel,
    mapping: &Mapping,
) -> TractResult<Option<TVec<OutletId>>> {
    let input_facts = source.node_input_facts(node.id)?;
    rule_if_some!(conv = node.op_as::<Conv>());
    rule_if!(
        input_facts.iter().all(|f| crate::utils::is_supported_fact(f))
            && matches!(input_facts[0].datum_type, DatumType::F16 | DatumType::F32)
            && conv.pool_spec.kernel_shape.len() == 2
            && conv.q_params.is_none()
    );
    let device_inputs = sync_inputs_if_required(target, node, mapping, DeviceSyncKind::ToDevice)?;
    wire_wgpu_conv(source, node, target, &device_inputs, conv).map(Some)
}

pub fn wire_wgpu_conv(
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
    let f16_accumulation = crate::kernels::TRACT_WGPU_ALLOW_F16_ACCUMULATION.get();
    let (packed_weights, packed_tile) =
        packed_direct_weights(facts[0], facts[1], op, f16_accumulation)?.unzip();
    let mut conv_wire = target.wire_node(
        if need_bias { &conv_name } else { &node.name },
        WgpuConv {
            op: op.clone(),
            epilogue: vec![],
            f16_accumulation,
            packed_weights,
            packed_tile,
        },
        &inputs[0..2],
    )?[0];
    if need_bias {
        let mut needed_shape = tvec![1.to_dim(); node.outputs[0].fact.rank()];
        needed_shape[data_shape.c_axis()] = facts[2].shape.volume();
        let reshaped = target.wire_node(
            format!("{prefix}.bias_reshaped"),
            GpuAxisOp::new(AxisOp::Reshape(0, bias.shape.to_tvec(), needed_shape)),
            &[inputs[2]],
        )?[0];
        conv_wire = target.wire_node(
            prefix,
            wgpu_bin_op(Box::new(tract_core::ops::math::Add)),
            &[conv_wire, reshaped],
        )?[0];
    }
    Ok(tvec!(conv_wire))
}

/// Packs one measured RVM direct 3×3 for the vec4 weight kernel. Each filter
/// row gains a zero lane. Absent means the scalar direct kernel.
fn packed_direct_weights(
    input: &TypedFact,
    weight: &TypedFact,
    op: &Conv,
    f16_accumulation: bool,
) -> TractResult<Option<(Const, DirectTile)>> {
    if !f16_accumulation || op.group != 1 || op.kernel_fmt != KernelFormat::OIHW {
        return Ok(None);
    }
    let spec = &op.pool_spec;
    if spec.data_format != DataFormat::NCHW
        || spec.kernel_shape[..] != [3, 3]
        || spec.strides()[..] != [1, 1]
        || spec.dilations()[..] != [1, 1]
    {
        return Ok(None);
    }
    let Some([1, ci, h, w]) = input.shape.as_concrete() else {
        return Ok(None);
    };
    let Some(tile) = measured_packed_tile(*ci, spec.output_channels, *h, *w) else {
        return Ok(None);
    };
    let padding = spec.computed_padding(&[*h, *w]);
    if padding[0].pad_before != 1 || padding[1].pad_before != 1 {
        return Ok(None);
    }
    let Some(weights) = weight.konst.as_ref() else {
        return Ok(None);
    };
    if weights.datum_type() != DatumType::F16
        || weights.shape() != [spec.output_channels, *ci, 3, 3]
        || !weights.has_plain_ram_storage()
    {
        return Ok(None);
    }
    if !crate::kernels::matmul::apple_webgpu(crate::context::wgpu_context().device()) {
        return Ok(None);
    }
    let taps = weights.try_as_plain_ram()?.as_slice::<f16>()?;
    let padded: Vec<f16> =
        taps.chunks_exact(3).flat_map(|row| [row[0], row[1], row[2], f16::ZERO]).collect();
    let shape = [spec.output_channels, *ci, 3, 4];
    ensure!(padded.len() == shape.iter().product::<usize>());
    let host = Const::new(Tensor::from_shape(&shape, &padded)?.into_arc_tensor())?;
    Ok(Some((crate::transform::convert_const(&host)?, tile)))
}

/// A convolution with any elementwise ops that followed it applied to each
/// result before it is stored. Inputs past the data and the weights belong to
/// the epilogue, one per binary step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WgpuConv {
    pub op: Conv,
    pub epilogue: Vec<crate::kernels::shaders::ChainStep>,
    /// f16 data and weights accumulate in f16 rather than f32.
    pub f16_accumulation: bool,
    pub(crate) packed_weights: Option<Const>,
    pub(crate) packed_tile: Option<DirectTile>,
}

impl Op for WgpuConv {
    fn name(&self) -> StaticName {
        "WgpuConv".into()
    }
    fn info(&self) -> TractResult<Vec<String>> {
        let mut info = self.op.info()?;
        if self.f16_accumulation {
            info.push("f16 accumulation".to_string());
        }
        if self.packed_weights.is_some() {
            info.push("packed direct weights".to_string());
        }
        Ok(info)
    }
    op_as_typed_op!();
}

impl EvalOp for WgpuConv {
    op_out_of_plan!();

    fn eval(&self, ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let inputs =
            inputs.iter().map(|it| it.to_device_tensor()).collect::<TractResult<TVec<_>>>()?;
        let output_shape = self.op.pool_spec.output_shape(inputs[0].shape())?;
        let output = tract_gpu::turn_handler::make_tensor_for_node(
            ctx,
            inputs[0].datum_type(),
            &output_shape.shape,
        )?;
        if output.len() > 0 {
            let packed =
                self.packed_weights.as_ref().map(|p| p.val().to_device_tensor()).transpose()?;
            wgpu_conv_dispatch(
                &self.op,
                &self.epilogue,
                inputs[0],
                inputs[1],
                &inputs[2..],
                &output,
                self.f16_accumulation,
                packed,
                self.packed_tile,
            )?;
        }
        Ok(tvec!(output.into_tensor().into_tvalue()))
    }
}

impl TypedOp for WgpuConv {
    as_op!();
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        tract_gpu::utils::facts_to_device_facts(inputs, |facts| {
            let zero = facts[0].datum_type.scalar_fact();
            let mut facts: TVec<&TypedFact> = facts[0..2.min(facts.len())].into();
            if facts.len() == 2 {
                facts.push(&zero);
            }
            self.op.output_facts(&facts)
        })
        .with_context(|| "Error while computing facts for WgpuConv")
    }
}
