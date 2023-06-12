use tract_hir::internal::*;
use tract_hir::ops::binary::wire_cast;
use tract_hir::ops::cnn::PaddingSpec;
use tract_hir::ops::logic::wire_with_rank_broadcast;
use tract_hir::tract_core::ops as core;
use tract_hir::tract_core::ops::cnn::KernelFormat;

use crate::registry::Registry;
use crate::tflite::{ActivationFunctionType, BuiltinOperator, Model, Operator, Padding, SubGraph};

pub fn register_all(reg: &mut Registry) {
    reg.binary_ops.push((BuiltinOperator::ADD, core::math::add()));
    reg.binary_ops.push((BuiltinOperator::SUB, core::math::sub()));
    reg.binary_ops.push((BuiltinOperator::MUL, core::math::mul()));
    reg.binary_ops.push((BuiltinOperator::DIV, core::math::div()));

    reg.to_tract.insert(BuiltinOperator::RESHAPE, reshape);

    reg.to_tract.insert(BuiltinOperator::AVERAGE_POOL_2D, average_pool_2d);
    reg.to_tract.insert(BuiltinOperator::CONV_2D, conv2d);
    reg.to_tract.insert(BuiltinOperator::DEPTHWISE_CONV_2D, dw_conv2d);

    reg.to_tract.insert(BuiltinOperator::MEAN, reduce_mean);

    reg.to_tract.insert(BuiltinOperator::RELU, relu);
    reg.to_tract.insert(BuiltinOperator::RELU6, relu6);
    reg.element_wise_ops.push((BuiltinOperator::HARD_SWISH, Box::new(core::nn::HardSwish {})));
}

fn reshape(
    _model: &Model,
    _subgraph: &SubGraph,
    prefix: &str,
    _flat: &Operator,
    target: &mut TypedModel,
    inputs: &[OutletId],
) -> TractResult<TVec<OutletId>> {
    let input_shape: TVec<TDim> = target.outlet_fact(inputs[0])?.shape.to_tvec();
    let shape = target.outlet_fact(inputs[1])?.konst.clone().unwrap();
    let shape = shape.cast_to::<TDim>()?;
    let shape = shape.as_slice::<TDim>()?;
    let mut wire = tvec!(inputs[0]);
    for (ix, op) in to_axis_ops_with_tf_rules(&input_shape, shape)?.into_iter().enumerate() {
        wire = target.wire_node(format!("{prefix}.{ix}"), op, &wire)?;
    }
    Ok(wire)
}

fn average_pool_2d(
    model: &Model,
    subgraph: &SubGraph,
    prefix: &str,
    flat: &Operator,
    target: &mut TypedModel,
    inputs: &[OutletId],
) -> TractResult<TVec<OutletId>> {
    let options = flat.builtin_options_as_pool_2_doptions().unwrap();
    let strides = tvec!(options.stride_h() as usize, options.stride_w() as usize);
    let kernel_shape = tvec!(options.filter_height() as usize, options.filter_width() as usize);
    let padding = match options.padding() {
        Padding::SAME => PaddingSpec::SameUpper,
        Padding::VALID => PaddingSpec::Valid,
        _ => todo!(),
    };
    let pool_spec = core::cnn::PoolSpec {
        data_format: tract_hir::ops::nn::DataFormat::NHWC,
        kernel_shape,
        padding,
        strides: Some(strides),
        dilations: None,
        output_channel_override: None,
    };
    let op = core::cnn::SumPool { pool_spec, normalize: true, count_include_pad: false };
    let wires = target.wire_node(prefix, op, &inputs[0..1])?;
    wire_fused_activation(
        model,
        subgraph,
        prefix,
        flat,
        target,
        &wires,
        &options.fused_activation_function(),
    )
}

fn conv2d(
    model: &Model,
    subgraph: &SubGraph,
    prefix: &str,
    flat: &Operator,
    target: &mut TypedModel,
    inputs: &[OutletId],
) -> TractResult<TVec<OutletId>> {
    let mut facts =
        inputs.iter().map(|o| target.outlet_fact(*o)).collect::<TractResult<TVec<_>>>()?;
    let (_input, kernel, bias) = args_3!(facts);
    let bias = bias.konst.clone().unwrap();
    let kernel = kernel.konst.clone().unwrap();
    let kernel_full_shape: TVec<usize> = kernel.shape().into();
    let kernel_shape: TVec<usize> = KernelFormat::OHWI.spatial_shape(&kernel_full_shape).into();
    let options = flat.builtin_options_as_conv_2_doptions().unwrap();
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
    let op = core::cnn::ConvUnary {
        pool_spec,
        kernel_fmt: KernelFormat::OHWI,
        kernel,
        group: 1,
        bias: Some(bias),
        q_params: None,
    };
    let wires = target.wire_node(prefix, op, &inputs[0..1])?;
    wire_fused_activation(
        model,
        subgraph,
        prefix,
        flat,
        target,
        &wires,
        &options.fused_activation_function(),
    )
}

fn dw_conv2d(
    model: &Model,
    subgraph: &SubGraph,
    prefix: &str,
    flat: &Operator,
    target: &mut TypedModel,
    inputs: &[OutletId],
) -> TractResult<TVec<OutletId>> {
    let mut facts =
        inputs.iter().map(|o| target.outlet_fact(*o)).collect::<TractResult<TVec<_>>>()?;
    let (_input, kernel, bias) = args_3!(facts);
    let bias = bias.konst.clone().unwrap();
    let kernel = kernel.konst.clone().unwrap();
    let kernel_full_shape: TVec<usize> = kernel.shape().into();
    let kernel_shape: TVec<usize> = KernelFormat::OHWI.spatial_shape(&kernel_full_shape).into();
    let options = flat.builtin_options_as_depthwise_conv_2_doptions().unwrap();
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
    let op = core::cnn::ConvUnary {
        pool_spec,
        kernel_fmt: KernelFormat::OHWI,
        kernel,
        group: co,
        bias: Some(bias),
        q_params: None,
    };
    let wires = target.wire_node(prefix, op, &inputs[0..1])?;
    wire_fused_activation(
        model,
        subgraph,
        prefix,
        flat,
        target,
        &wires,
        &options.fused_activation_function(),
    )
}

fn wire_fused_activation(
    model: &Model,
    subgraph: &SubGraph,
    prefix: &str,
    flat: &Operator,
    target: &mut TypedModel,
    inputs: &[OutletId],
    activation: &ActivationFunctionType,
) -> TractResult<TVec<OutletId>> {
    match *activation {
        ActivationFunctionType::NONE => Ok(inputs.into()),
        ActivationFunctionType::RELU => relu(model, subgraph, prefix, flat, target, inputs),
        ActivationFunctionType::RELU6 => relu6(model, subgraph, prefix, flat, target, inputs),
        af => bail!("Unsupported fused activation type: {af:?}"),
    }
}

fn reduce_mean(
    _model: &Model,
    _subgraph: &SubGraph,
    prefix: &str,
    flat: &Operator,
    target: &mut TypedModel,
    inputs: &[OutletId],
) -> TractResult<TVec<OutletId>> {
    let mut facts =
        inputs.iter().map(|o| target.outlet_fact(*o).cloned()).collect::<TractResult<TVec<_>>>()?;
    let (input, axes) = args_2!(facts);
    let options = flat.builtin_options_as_reducer_options().unwrap();
    ensure!(options.keep_dims());
    let axes: TVec<usize> =
        axes.konst.as_ref().unwrap().as_slice::<i32>()?.iter().map(|d| *d as usize).collect();
    let norm: TDim = axes.iter().map(|d| &input.shape[*d]).product();
    let wire = target.wire_node(
        format!("{prefix}.sum"),
        core::nn::Reduce::new(axes, core::nn::Reducer::Sum),
        &[inputs[0]],
    )?;
    let norm = target.add_const("{prefix}.card", tensor0(norm))?;
    let wires = wire_cast(prefix, target, &[wire[0], norm], input.datum_type)?;
    wire_with_rank_broadcast(prefix, target, core::math::div(), &wires)
}

fn relu(
    _model: &Model,
    _subgraph: &SubGraph,
    prefix: &str,
    _flat: &Operator,
    target: &mut TypedModel,
    inputs: &[OutletId],
) -> TractResult<TVec<OutletId>> {
    let input = inputs[0];
    let zero = target.add_const(format!("{prefix}.zero"), tensor0(0f32))?;
    let wires = wire_cast(prefix, target, &[input, zero], target.outlet_fact(input)?.datum_type)?;
    wire_with_rank_broadcast(&format!("{prefix}.relu"), target, core::math::max(), &wires)
}

fn relu6(
    model: &Model,
    subgraph: &SubGraph,
    prefix: &str,
    flat: &Operator,
    target: &mut TypedModel,
    inputs: &[OutletId],
) -> TractResult<TVec<OutletId>> {
    let input = relu(model, subgraph, prefix, flat, target, inputs)?[0];
    let six = target.add_const(format!("{prefix}.six"), tensor0(6f32))?;
    let wires = wire_cast(prefix, target, &[input, six], target.outlet_fact(input)?.datum_type)?;
    wire_with_rank_broadcast(&format!("{prefix}.relu6"), target, core::math::min(), &wires)
}
