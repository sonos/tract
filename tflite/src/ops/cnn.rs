use super::wire_fused_activation;
use crate::registry::{DeserOp, Registry};
use crate::ser::{BuiltinOp, SubgraphBuilder};
use crate::tflite::{
    ActivationFunctionType, BuiltinOperator, BuiltinOptions, Conv2DOptions, Conv2DOptionsArgs,
    DepthwiseConv2DOptions, DepthwiseConv2DOptionsArgs, PadOptions, PadOptionsArgs, Padding,
};
use tract_hir::internal::*;
use tract_hir::ops::array::{Pad, PadMode};
use tract_hir::ops::cnn::{ConvUnary, PaddingSpec};
use tract_hir::ops::nn::DataFormat;
use tract_hir::ops::quant::quantize_linear_i8;
use tract_hir::prelude::tract_itertools::Itertools;
use tract_hir::tract_core::ops as core;
use tract_hir::tract_core::ops::cnn::KernelFormat;

pub fn register_all(reg: &mut Registry) {
    reg.reg_to_tract(BuiltinOperator::AVERAGE_POOL_2D, average_pool_2d);
    reg.reg_to_tract(BuiltinOperator::CONV_2D, de_conv2d);
    reg.reg_to_tflite::<ConvUnary>(ser_conv);
    reg.reg_to_tract(BuiltinOperator::DEPTHWISE_CONV_2D, de_dw_conv2d);
    reg.reg_to_tflite::<Pad>(ser_pad);
}

fn average_pool_2d(op: &mut DeserOp) -> TractResult<TVec<OutletId>> {
    let options = builtin!(op, builtin_options_as_pool_2_doptions);
    let strides = tvec!(options.stride_h() as usize, options.stride_w() as usize);
    let kernel_shape = tvec!(options.filter_height() as usize, options.filter_width() as usize);
    let padding = match options.padding() {
        Padding::SAME => PaddingSpec::SameUpper,
        Padding::VALID => PaddingSpec::Valid,
        _ => todo!(),
    };
    let pool_spec = core::cnn::PoolSpec {
        data_format: DataFormat::NHWC,
        kernel_shape,
        padding,
        strides: Some(strides),
        dilations: None,
        output_channel_override: None,
    };
    let pool = core::cnn::SumPool { pool_spec, normalize: true, count_include_pad: false };
    let wires = op.ctx.target.wire_node(op.prefix, pool, &op.inputs[0..1])?;
    wire_fused_activation(op, &wires, &options.fused_activation_function())
}

fn ser_conv(
    builder: &mut SubgraphBuilder,
    model: &TypedModel,
    node: &TypedNode,
) -> TractResult<()> {
    let conv = node.op_as::<ConvUnary>().unwrap();
    ensure!(conv.pool_spec.data_format == DataFormat::NHWC);
    ensure!(model.node_input_facts(node.id)?[0].rank() == 4);
    ensure!(conv.kernel_fmt == KernelFormat::OHWI);
    ensure!(conv.group == 1 || conv.group.to_dim() == model.node_input_facts(node.id)?[0].shape[3]);
    ensure!(
        conv.pool_spec.padding == PaddingSpec::Valid
            || conv.pool_spec.padding == PaddingSpec::SameUpper
    );
    let node_name = &node.name;
    let mut inputs = node.inputs.iter().map(|o| builder.outlets_to_tensors[o]).collect_vec();
    let outputs = (0..node.outputs.len())
        .map(|o| builder.outlets_to_tensors[&OutletId::new(node.id, o)])
        .collect_vec();
    inputs.push(builder.write_fact(&format!("{node_name}.weights"), &conv.kernel)?);
    inputs.push(builder.write_fact(
        &format!("{node_name}.bias"),
        &conv.bias.clone().unwrap_or_else(|| {
            rctensor1(&vec![0f32; conv.pool_spec.output_channel_override.unwrap()])
        }),
    )?);
    let padding =
        if conv.pool_spec.padding == PaddingSpec::Valid { Padding::VALID } else { Padding::SAME };
    if conv.group == 1 {
        let options = Conv2DOptions::create(
            builder.fb(),
            &Conv2DOptionsArgs {
                padding,
                stride_h: conv.pool_spec.stride(0) as _,
                stride_w: conv.pool_spec.stride(1) as _,
                dilation_h_factor: conv.pool_spec.dilation(0) as _,
                dilation_w_factor: conv.pool_spec.dilation(1) as _,
                fused_activation_function: ActivationFunctionType::NONE,
            },
        );
        builder.write_op_with_options(
            &inputs,
            &outputs,
            BuiltinOp::new(3, 2, BuiltinOperator::CONV_2D, BuiltinOptions::Conv2DOptions),
            options.as_union_value(),
        )
    } else {
        let depth_multiplier =
            (conv.pool_spec.output_channel_override.unwrap() / conv.group) as i32;
        let options = DepthwiseConv2DOptions::create(
            builder.fb(),
            &DepthwiseConv2DOptionsArgs {
                padding,
                depth_multiplier,
                stride_h: conv.pool_spec.stride(0) as _,
                stride_w: conv.pool_spec.stride(1) as _,
                dilation_h_factor: conv.pool_spec.dilation(0) as _,
                dilation_w_factor: conv.pool_spec.dilation(1) as _,
                fused_activation_function: ActivationFunctionType::NONE,
            },
        );
        builder.write_op_with_options(
            &inputs,
            &outputs,
            BuiltinOp::new(
                4,
                2,
                BuiltinOperator::DEPTHWISE_CONV_2D,
                BuiltinOptions::DepthwiseConv2DOptions,
            ),
            options.as_union_value(),
        )
    }
}

fn quantization_suport(
    op: &mut DeserOp,
    input: &TypedFact,
    kernel: &Tensor,
    inputs: &mut TVec<OutletId>,
) -> TractResult<Option<DatumType>> {
    if op.output_facts[0].datum_type.is_quantized() {
        let p = &op.prefix;
        let kqp = kernel.datum_type().qparams().unwrap();
        let iqp = input.datum_type.qparams().unwrap();
        let oqp = op.output_facts[0].datum_type;
        inputs.push(op.ctx.target.add_const(format!("{p}.k0"), rctensor0(kqp.zp_scale().0))?);
        inputs.push(op.ctx.target.add_const(format!("{p}.kscale"), rctensor0(kqp.zp_scale().1))?);
        inputs.push(op.ctx.target.add_const(format!("{p}.i0"), rctensor0(iqp.zp_scale().0))?);
        inputs.push(op.ctx.target.add_const(format!("{p}.iscale"), rctensor0(iqp.zp_scale().1))?);
        inputs.push(op.ctx.target.add_const(format!("{p}.c0"), rctensor0(oqp.zp_scale().0))?);
        inputs.push(op.ctx.target.add_const(format!("{p}.cscale"), rctensor0(oqp.zp_scale().1))?);
        Ok(Some(oqp))
    } else {
        Ok(None)
    }
}

fn de_conv2d(op: &mut DeserOp) -> TractResult<TVec<OutletId>> {
    let (input, kernel, bias) = args_3!(op.facts()?);
    let kernel = kernel.konst.unwrap();
    let bias = bias.konst.unwrap();
    let kernel_full_shape: TVec<usize> = kernel.shape().into();
    let kernel_shape: TVec<usize> = KernelFormat::OHWI.spatial_shape(&kernel_full_shape).into();
    let options = builtin!(op, builtin_options_as_conv_2_doptions);
    let padding = match options.padding() {
        Padding::SAME => PaddingSpec::SameUpper,
        Padding::VALID => PaddingSpec::Valid,
        _ => todo!(),
    };
    let strides = tvec!(options.stride_h() as usize, options.stride_w() as usize);
    let dilations =
        tvec!(options.dilation_h_factor() as usize, options.dilation_w_factor() as usize);
    let co = KernelFormat::OHWI.o(&kernel_full_shape);
    let pool_spec = core::cnn::PoolSpec {
        data_format: tract_hir::ops::nn::DataFormat::NHWC,
        kernel_shape,
        padding,
        strides: Some(strides),
        dilations: Some(dilations),
        output_channel_override: Some(*co),
    };
    let mut inputs = tvec!(op.inputs[0]);
    let q_params = quantization_suport(op, &input, &kernel, &mut inputs)?;
    let conv = core::cnn::ConvUnary {
        pool_spec,
        kernel_fmt: KernelFormat::OHWI,
        kernel,
        group: 1,
        bias: Some(bias),
        q_params,
    };
    let wires = op.ctx.target.wire_node(op.prefix, conv, &inputs)?;
    wire_fused_activation(op, &wires, &options.fused_activation_function())
}

fn de_dw_conv2d(op: &mut DeserOp) -> TractResult<TVec<OutletId>> {
    let (input, kernel, bias) = args_3!(op.facts()?);
    let bias = bias.konst.unwrap();
    let kernel = kernel.konst.unwrap();
    let kernel_full_shape: TVec<usize> = kernel.shape().into();
    let kernel_shape: TVec<usize> = KernelFormat::OHWI.spatial_shape(&kernel_full_shape).into();
    let options = builtin!(op, builtin_options_as_depthwise_conv_2_doptions);
    let padding = match options.padding() {
        Padding::SAME => PaddingSpec::SameUpper,
        Padding::VALID => PaddingSpec::Valid,
        _ => todo!(),
    };
    let strides = tvec!(options.stride_h() as usize, options.stride_w() as usize);
    let dilations =
        tvec!(options.dilation_h_factor() as usize, options.dilation_w_factor() as usize);
    let co = *KernelFormat::OHWI.i(&kernel_full_shape);
    let pool_spec = core::cnn::PoolSpec {
        data_format: tract_hir::ops::nn::DataFormat::NHWC,
        kernel_shape,
        padding,
        strides: Some(strides),
        dilations: Some(dilations),
        output_channel_override: Some(co),
    };
    let mut inputs = tvec!(op.inputs[0]);
    let q_params = quantization_suport(op, &input, &kernel, &mut inputs)?;
    let conv = core::cnn::ConvUnary {
        pool_spec,
        kernel_fmt: KernelFormat::OHWI,
        kernel,
        group: co,
        bias: Some(bias),
        q_params,
    };
    let wires = op.ctx.target.wire_node(op.prefix, conv, &inputs)?;
    wire_fused_activation(op, &wires, &options.fused_activation_function())
}

fn ser_pad(
    builder: &mut SubgraphBuilder,
    _model: &TypedModel,
    node: &TypedNode,
) -> TractResult<()> {
    let pad = node.op_as::<Pad>().unwrap();
    let node_name = &node.name;
    let mut inputs = tvec!(builder.outlets_to_tensors[&node.inputs[0]]);
    let outputs = (0..node.outputs.len())
        .map(|o| builder.outlets_to_tensors[&OutletId::new(node.id, o)])
        .collect_vec();
    let paddings = tract_ndarray::Array2::<i32>::from_shape_fn((pad.pads.len(), 2), |(d, side)| {
        (if side == 0 { pad.pads[d].0 } else { pad.pads[d].1 }) as i32
    });
    inputs.push(builder.write_fact(format!("{node_name}.paddings"), paddings.into_tensor())?);
    let PadMode::Constant(pad_value) = &pad.mode else {
        bail!("Only constant padding is supported by tflite");
    };
    inputs.push(builder.write_fact(format!("{node_name}.pad_value"), pad_value)?);
    let options = PadOptions::create(builder.fb(), &PadOptionsArgs {});
    builder.write_op_with_options(
        &inputs,
        &outputs,
        BuiltinOp::new(60, 1, BuiltinOperator::PADV2, BuiltinOptions::PadV2Options),
        options.as_union_value(),
    )?;
    Ok(())
}
