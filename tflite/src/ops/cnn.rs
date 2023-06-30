use super::wire_fused_activation;
use crate::registry::{DeserOp, Registry};
use crate::ser::{BuiltinOp, SubgraphBuilder};
use crate::tflite::{
    ActivationFunctionType, BuiltinOperator, BuiltinOptions, Conv2DOptions, Conv2DOptionsArgs,
    Padding,
};
use tract_hir::internal::*;
use tract_hir::ops::cnn::{ConvUnary, PaddingSpec};
use tract_hir::ops::nn::DataFormat;
use tract_hir::prelude::tract_itertools::Itertools;
use tract_hir::tract_core::ops as core;
use tract_hir::tract_core::ops::cnn::KernelFormat;

pub fn register_all(reg: &mut Registry) {
    reg.to_tract.insert(BuiltinOperator::AVERAGE_POOL_2D, average_pool_2d);
    reg.to_tract.insert(BuiltinOperator::CONV_2D, conv2d);
    reg.reg_to_tflite::<ConvUnary>(ser_conv);
    reg.to_tract.insert(BuiltinOperator::DEPTHWISE_CONV_2D, dw_conv2d);
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
    ensure!(conv.group == 1);
    let node_name = &node.name;
    let mut inputs = node.inputs.iter().map(|o| builder.outlets_to_tensors[o]).collect_vec();
    let outputs = (0..node.outputs.len())
        .map(|o| builder.outlets_to_tensors[&OutletId::new(node.id, o)])
        .collect_vec();
    let ohwi = match conv.kernel_fmt {
        KernelFormat::OHWI => conv.kernel.clone(),
        KernelFormat::HWIO => conv.kernel.clone().into_tensor().move_axis(3, 0)?.into_arc_tensor(),
        KernelFormat::OIHW => conv.kernel.clone().into_tensor().move_axis(1, 3)?.into_arc_tensor(),
    };
    inputs.push(builder.write_fact(&format!("{node_name}.weights"), &ohwi.into())?);
    inputs.push(
        builder.write_fact(
            &format!("{node_name}.bias"),
            &conv
                .bias
                .clone()
                .unwrap_or_else(|| {
                    rctensor1(&vec![0f32; conv.pool_spec.output_channel_override.unwrap()])
                })
                .into(),
        )?,
    );
    let padding = &conv.pool_spec.padding;
    let padding_h = (conv.pool_spec.kernel_shape[0] - 1) * conv.pool_spec.dilation(0);
    let padding_w = (conv.pool_spec.kernel_shape[1] - 1) * conv.pool_spec.dilation(1);
    let padding = if padding.valid_dim(0, true) && padding.valid_dim(1, true) {
        Padding::VALID
    } else if padding
        == &PaddingSpec::Explicit(
            tvec!(padding_h / 2, padding_w / 2),
            tvec!(padding_h.divceil(2), padding_w.divceil(2)),
            false,
        )
    {
        Padding::SAME
    } else if padding == &PaddingSpec::SameUpper {
        Padding::SAME
    } else {
        todo!();
    };
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
    )?;
    Ok(())
}

fn conv2d(op: &mut DeserOp) -> TractResult<TVec<OutletId>> {
    let (_input, kernel, bias) = args_3!(op.facts()?);
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
    let conv = core::cnn::ConvUnary {
        pool_spec,
        kernel_fmt: KernelFormat::OHWI,
        kernel,
        group: 1,
        bias: Some(bias),
        q_params: None,
    };
    let wires = op.ctx.target.wire_node(op.prefix, conv, &op.inputs[0..1])?;
    wire_fused_activation(op, &wires, &options.fused_activation_function())
}

fn dw_conv2d(op: &mut DeserOp) -> TractResult<TVec<OutletId>> {
    let (_input, kernel, bias) = args_3!(op.facts()?);
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
    let conv = core::cnn::ConvUnary {
        pool_spec,
        kernel_fmt: KernelFormat::OHWI,
        kernel,
        group: co,
        bias: Some(bias),
        q_params: None,
    };
    let wires = op.ctx.target.wire_node(op.prefix, conv, &op.inputs[0..1])?;
    wire_fused_activation(op, &wires, &options.fused_activation_function())
}
