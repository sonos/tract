use tract_hir::internal::*;
use tract_hir::ops::binary::wire_cast;
use tract_hir::ops::cnn::PaddingSpec;
use tract_hir::ops::logic::wire_with_rank_broadcast;
use tract_hir::tract_core::ops as core;
use tract_hir::tract_core::ops::cnn::KernelFormat;

use crate::registry::{DeserContext, DeserOp, Registry};
use crate::tflite::{ActivationFunctionType, BuiltinOperator, Padding};

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
    reg.to_tract.insert(BuiltinOperator::SOFTMAX, softmax);

    reg.to_tract.insert(BuiltinOperator::RELU, relu);
    reg.to_tract.insert(BuiltinOperator::RELU6, relu6);
    reg.element_wise_ops.push((BuiltinOperator::HARD_SWISH, Box::new(core::nn::HardSwish {})));
}

macro_rules! builtin {
    ($op: expr, $id:ident) => {
        $op.flat.$id().with_context(|| {
            format!(
                "Wrong option type {:?} for operator {:?}",
                $op.flat.builtin_options_type(),
                $op.flat
            )
        })?
    };
}

fn reshape(op: &mut DeserOp) -> TractResult<TVec<OutletId>> {
    let input_shape: TVec<TDim> = op.ctx.target.outlet_fact(op.inputs[0])?.shape.to_tvec();
    let shape = op.ctx.target.outlet_fact(op.inputs[1])?.konst.clone().unwrap();
    let shape = shape.cast_to::<TDim>()?;
    let shape = shape.as_slice::<TDim>()?;
    let mut wire = tvec!(op.inputs[0]);
    let prefix = op.prefix;
    for (ix, axis_op) in to_axis_ops_with_tf_rules(&input_shape, shape)?.into_iter().enumerate() {
        wire = op.ctx.target.wire_node(format!("{prefix}.{ix}"), axis_op, &wire)?;
    }
    Ok(wire)
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
        data_format: tract_hir::ops::nn::DataFormat::NHWC,
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

fn wire_fused_activation(
    op: &mut DeserOp,
    wires: &[OutletId],
    activation: &ActivationFunctionType,
) -> TractResult<TVec<OutletId>> {
    let prefix = format!("{}.fused", op.prefix);
    let mut op = DeserOp {
        ctx: DeserContext { model: op.ctx.model, subgraph: op.ctx.subgraph, target: op.ctx.target },
        prefix: &prefix,
        flat: op.flat,
        inputs: wires,
    };
    match *activation {
        ActivationFunctionType::NONE => Ok(wires.into()),
        ActivationFunctionType::RELU => relu(&mut op),
        ActivationFunctionType::RELU6 => relu6(&mut op),
        af => bail!("Unsupported fused activation type: {af:?}"),
    }
}

fn reduce_mean(op: &mut DeserOp) -> TractResult<TVec<OutletId>> {
    let (input, axes) = args_2!(op.facts()?);
    let options = builtin!(op, builtin_options_as_reducer_options);
    ensure!(options.keep_dims());
    let axes: TVec<usize> =
        axes.konst.as_ref().unwrap().as_slice::<i32>()?.iter().map(|d| *d as usize).collect();
    let norm: TDim = axes.iter().map(|d| &input.shape[*d]).product();
    let wire = op.ctx.target.wire_node(
        op.prefix.to_string() + ".sum",
        core::nn::Reduce::new(axes, core::nn::Reducer::Sum),
        &[op.inputs[0]],
    )?;
    let norm = op.ctx.target.add_const("{prefix}.card", tensor0(norm))?;
    let wires = wire_cast(op.prefix, op.ctx.target, &[wire[0], norm], input.datum_type)?;
    wire_with_rank_broadcast(op.prefix, op.ctx.target, core::math::div(), &wires)
}

fn softmax(op: &mut DeserOp) -> TractResult<TVec<OutletId>> {
    let input = args_1!(op.facts()?);
    let options = builtin!(op, builtin_options_as_softmax_options);
    ensure!(options.beta() == 1.0);
    let softmax = core::nn::Softmax { axes: tvec!(input.rank() - 1), output_dt: input.datum_type };
    op.ctx.target.wire_node(op.prefix, softmax, op.inputs)
}

fn relu(op: &mut DeserOp) -> TractResult<TVec<OutletId>> {
    let input = op.inputs[0];
    let zero = op.ctx.target.add_const(format!("{}.zero", op.prefix), tensor0(0f32))?;
    let wires = wire_cast(
        op.prefix,
        op.ctx.target,
        &[input, zero],
        op.ctx.target.outlet_fact(input)?.datum_type,
    )?;
    wire_with_rank_broadcast(
        &format!("{}.relu", op.prefix),
        op.ctx.target,
        core::math::max(),
        &wires,
    )
}

fn relu6(op: &mut DeserOp) -> TractResult<TVec<OutletId>> {
    let input = relu(op)?[0];
    let six = op.ctx.target.add_const(format!("{}.six", op.prefix), tensor0(6f32))?;
    let wires = wire_cast(
        op.prefix,
        op.ctx.target,
        &[input, six],
        op.ctx.target.outlet_fact(input)?.datum_type,
    )?;
    wire_with_rank_broadcast(
        &format!("{}.relu6", op.prefix),
        op.ctx.target,
        core::math::min(),
        &wires,
    )
}
