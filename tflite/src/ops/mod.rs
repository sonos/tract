use tract_hir::internal::*;
use tract_hir::ops::cnn::PaddingSpec;
use tract_hir::tract_core::ops as core;
use tract_hir::tract_core::ops::cnn::KernelFormat;

use crate::registry::Registry;
use crate::tflite::{ActivationFunctionType, BuiltinOperator, Model, Operator, Padding, SubGraph};

pub fn register_all(reg: &mut Registry) {
    reg.binary_ops.push((BuiltinOperator::ADD, core::math::add()));
    reg.binary_ops.push((BuiltinOperator::SUB, core::math::sub()));
    reg.binary_ops.push((BuiltinOperator::MUL, core::math::mul()));
    reg.binary_ops.push((BuiltinOperator::DIV, core::math::div()));

    reg.to_tract.insert(BuiltinOperator::CONV_2D, conv2d);
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
    let kernel = kernel.konst.clone().unwrap();
    let bias = bias.konst.clone().unwrap();
    let kernel_shape = KernelFormat::OIHW.spatial_shape(&kernel.shape()).into();
    let options = flat.builtin_options_as_conv_2_doptions().unwrap();
    assert!(options.fused_activation_function() == ActivationFunctionType::NONE);
    let padding = match options.padding() {
        Padding::SAME => PaddingSpec::SameUpper,
        Padding::VALID => PaddingSpec::Valid,
        _ => todo!(),
    };
    let strides = tvec!(options.stride_h() as usize, options.stride_w() as usize);
    let dilations =
        tvec!(options.dilation_h_factor() as usize, options.dilation_w_factor() as usize);
    let co = KernelFormat::OIHW.o(&kernel.shape());
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
        kernel_fmt: KernelFormat::OIHW,
        kernel,
        group: 1,
        bias: Some(bias),
        q_params: None,
    };
    target.wire_node(prefix, op, &inputs[0..1])
}
