use tract_core::internal::*;
use tract_core::ops as tractops;
use tract_core::ops::cnn::{KernelFormat, PaddingSpec};
use tract_core::ops::nn::DataFormat;

use crate::model::{ OnnxOpRegister, ParsingContext };
use crate::pb::NodeProto;
use crate::pb_helpers::OptionExt;

mod dropout;

macro_rules! reduce {
    ($id:ident) => {
        |_cts, node| {
            let axes = node.get_attr_opt_vec("axes")?;
            //                .map(|axes| axes.iter().map(|&i| i as i64).collect());
            let keep_dims = node.get_attr_opt("keepdims")?.unwrap_or(1i64) == 1;
            Ok((Box::new(tractops::nn::Reduce::new(axes, keep_dims, tractops::nn::Reducer::$id)), vec!()))
        }
    };
}

pub fn register_all_ops(reg: &mut OnnxOpRegister) {
    reg.insert("ArgMax", arg_max_min);
    reg.insert("ArgMin", arg_max_min);
    reg.insert("AveragePool", average_pool);
    reg.insert("BatchNormalization", batch_normalization);
    reg.insert("Conv", conv);
    reg.insert("Dropout", dropout::dropout);
    reg.insert("Elu", elu);
    reg.insert("GlobalAveragePool", |_, _| Ok((Box::new(tractops::nn::GlobalAvgPool::default()),vec!())));
    reg.insert("GlobalLpPool", global_lp_pool);
    reg.insert("GlobalMaxPool", |_, _| Ok((Box::new(tractops::nn::GlobalMaxPool::default()),vec!())));
    reg.insert("Hardmax", layer_hard_max);
    reg.insert("HardSigmoid", hard_sigmoid);
    reg.insert("LeakyRelu", leaky_relu);
    reg.insert("LogSoftmax", layer_log_soft_max);
    reg.insert("LRN", lrn);
    reg.insert("MaxPool", max_pool);
    reg.insert("ParametricSoftplus", parametric_softplus);
    reg.insert("PRelu", |_, _| Ok((Box::new(Prelu::default()),vec!())));
    reg.insert("ReduceL1", reduce!(L1));
    reg.insert("ReduceL2", reduce!(L2));
    reg.insert("ReduceLogSum", reduce!(LogSum));
    reg.insert("ReduceLogSumExp", reduce!(LogSumExp));
    reg.insert("ReduceMax", reduce!(Max));
    reg.insert("ReduceMean", reduce!(Mean));
    reg.insert("ReduceMin", reduce!(Min));
    reg.insert("ReduceProd", reduce!(Prod));
    reg.insert("ReduceSum", reduce!(Sum));
    reg.insert("ReduceSumSquare", reduce!(SumSquare));
    reg.insert("Relu", |_, _| Ok((Box::new(tractops::nn::Relu::default()),vec!())));
    reg.insert("ScaledTanh", scaled_tanh);
    reg.insert("Shrink", shrink);
    reg.insert("ThresholdedRelu", thresholded_relu);
    reg.insert("Selu", selu);
    reg.insert("Sigmoid", |_, _| Ok((Box::new(tractops::nn::Sigmoid::default()),vec!())));
    reg.insert("Softmax", layer_soft_max);
    reg.insert("Softplus", |_, _| Ok((Box::new(tractops::nn::Softplus::default()),vec!())));
    reg.insert("Softsign", |_, _| Ok((Box::new(tractops::nn::Softsign::default()),vec!())));
}

fn pad(node: &NodeProto) -> TractResult<PaddingSpec> {
    if let Some(pads) = node.get_attr_opt_tvec("pads")? {
        let len = pads.len();
        return Ok(PaddingSpec::Explicit(
            pads.iter().cloned().take(len / 2).collect(),
            pads.iter().cloned().skip(len / 2).collect(),
        ));
    }
    Ok(node
        .get_attr_opt("auto_pad")?
        .and_try(|s| {
            node.check_value(
                "auto_pad",
                match s {
                    "NOTSET" => Ok(PaddingSpec::Valid),
                    "VALID" => Ok(PaddingSpec::Valid),
                    "SAME_UPPER" => Ok(PaddingSpec::SameUpper),
                    "SAME_LOWER" => Ok(PaddingSpec::SameLower),
                    _ => Err(s),
                },
            )
        })?
        .unwrap_or(PaddingSpec::Valid))
}

fn dilations(node: &NodeProto) -> TractResult<Option<TVec<usize>>> {
    node.get_attr_opt_tvec("dilations")
}

fn strides(node: &NodeProto) -> TractResult<Option<TVec<usize>>> {
    node.get_attr_opt_tvec("strides")
}

pub fn arg_max_min(_ctx: &ParsingContext, node: &NodeProto) -> TractResult<(Box<dyn InferenceOp>,Vec<String>)> {
    let max = node.get_op_type() == "ArgMax";
    let axis = node.get_attr_opt("axis")?.unwrap_or(0);
    let keepdims = node.get_attr_opt("keepdims")?.unwrap_or(true);
    Ok((Box::new(tractops::nn::ArgMaxMin::new(max, axis, keepdims)),vec!()))
}

pub fn batch_normalization(_ctx: &ParsingContext, node: &NodeProto) -> TractResult<(Box<dyn InferenceOp>,Vec<String>)> {
    let epsilon = node.get_attr_opt("epsilon")?.unwrap_or(1e-5);
    let spatial = node.get_attr_opt("spatial")?.unwrap_or(0);
    assert_eq!(spatial, 0);
    Ok((Box::new(tractops::nn::BatchNorm::new(DataFormat::NCHW, epsilon, spatial != 0)),vec!()))
}

pub fn conv(_ctx: &ParsingContext, node: &NodeProto) -> TractResult<(Box<dyn InferenceOp>,Vec<String>)> {
    let kernel_shape = node.get_attr_opt_tvec("kernel_shape")?;
    let group = node.get_attr_opt("group")?.unwrap_or(1);
    Ok((Box::new(tractops::cnn::Conv::new(
        DataFormat::NCHW,
        KernelFormat::OIHW,
        dilations(node)?,
        kernel_shape,
        pad(node)?,
        strides(node)?,
        group,
    )),vec!()))
}

pub fn average_pool(_ctx: &ParsingContext, node: &NodeProto) -> TractResult<(Box<dyn InferenceOp>,Vec<String>)> {
    let kernel_shape = node.get_attr_tvec("kernel_shape")?;
    let pad = pad(node)?;
    let strides = strides(node)?;
    let count_include_pad = node.get_attr_opt("count_include_pad")?.unwrap_or(false);
    Ok((Box::new(tractops::cnn::AvgPool::new(
        tractops::cnn::PoolSpec::new(DataFormat::NCHW, kernel_shape, pad, strides),
        count_include_pad,
    )),vec!()))
}

pub fn elu(_ctx: &ParsingContext, node: &NodeProto) -> TractResult<(Box<dyn InferenceOp>,Vec<String>)> {
    let alpha = node.get_attr_opt("alpha")?.unwrap_or(1.);
    Ok((Box::new(tractops::nn::Elu::new(alpha)),vec!()))
}

pub fn global_lp_pool(_ctx: &ParsingContext, node: &NodeProto) -> TractResult<(Box<dyn InferenceOp>,Vec<String>)> {
    let p: usize = node.get_attr_opt("p")?.unwrap_or(2);
    Ok((Box::new(tractops::nn::GlobalLpPool::new(p)),vec!()))
}

pub fn hard_sigmoid(_ctx: &ParsingContext, node: &NodeProto) -> TractResult<(Box<dyn InferenceOp>,Vec<String>)> {
    let alpha = node.get_attr_opt("alpha")?.unwrap_or(0.2);
    let beta = node.get_attr_opt("beta")?.unwrap_or(0.5);
    Ok((Box::new(tractops::nn::Hardsigmoid::new(alpha, beta)),vec!()))
}

pub fn layer_hard_max(_ctx: &ParsingContext, node: &NodeProto) -> TractResult<(Box<dyn InferenceOp>,Vec<String>)> {
    let axis = node.get_attr_opt("axis")?.unwrap_or(1);
    Ok((Box::new(tractops::nn::LayerHardmax::new(axis)),vec!()))
}

pub fn layer_log_soft_max(_ctx: &ParsingContext, node: &NodeProto) -> TractResult<(Box<dyn InferenceOp>,Vec<String>)> {
    let axis = node.get_attr_opt("axis")?.unwrap_or(1);
    Ok((Box::new(tractops::nn::LayerLogSoftmax::new(axis)),vec!()))
}

pub fn layer_soft_max(_ctx: &ParsingContext, node: &NodeProto) -> TractResult<(Box<dyn InferenceOp>,Vec<String>)> {
    let axis = node.get_attr_opt("axis")?.unwrap_or(1);
    Ok((Box::new(tractops::nn::LayerSoftmax::new(axis)),vec!()))
}

pub fn leaky_relu(_ctx: &ParsingContext, node: &NodeProto) -> TractResult<(Box<dyn InferenceOp>,Vec<String>)> {
    let alpha = node.get_attr_opt("alpha")?.unwrap_or(0.01);
    Ok((Box::new(tractops::nn::LeakyRelu::new(alpha)),vec!()))
}

pub fn lrn(_ctx: &ParsingContext, node: &NodeProto) -> TractResult<(Box<dyn InferenceOp>,Vec<String>)> {
    let alpha = node.get_attr_opt("alpha")?.unwrap_or(0.0001);
    let beta = node.get_attr_opt("beta")?.unwrap_or(0.75);
    let bias = node.get_attr_opt("bias")?.unwrap_or(1.);
    let size = node.get_attr("size")?;
    Ok((Box::new(tractops::nn::Lrn::new(alpha, beta, bias, size)),vec!()))
}

pub fn max_pool(_ctx: &ParsingContext, node: &NodeProto) -> TractResult<(Box<dyn InferenceOp>,Vec<String>)> {
    let kernel_shape = node.get_attr_tvec("kernel_shape")?;
    let pad = pad(node)?;
    let strides = strides(node)?;
    Ok((Box::new(tractops::cnn::MaxPool::new(
        tractops::cnn::PoolSpec::new(DataFormat::NCHW, kernel_shape, pad, strides),
        if node.get_output().len() == 2 { Some(DatumType::I64) } else { None },
    )),vec!()))
}

pub fn parametric_softplus(_ctx: &ParsingContext, node: &NodeProto) -> TractResult<(Box<dyn InferenceOp>,Vec<String>)> {
    let alpha = node.get_attr("alpha")?;
    let beta = node.get_attr("beta")?;
    Ok((Box::new(tractops::nn::ParametricSoftplus::new(alpha, beta)),vec!()))
}

element_bin!(Prelu, match
    f16 => f16 { |a:f16, b:f16| {
        use num_traits::Zero;
        if a < f16::zero() { a*b } else { b }
    } },
    f32 => f32 { |a, b| if a < 0.0 { a*b } else { a } },
    f64 => f64 { |a, b| if a < 0.0 { a*b } else { a } }
);

pub fn scaled_tanh(_ctx: &ParsingContext, node: &NodeProto) -> TractResult<(Box<dyn InferenceOp>,Vec<String>)> {
    let alpha = node.get_attr("alpha")?;
    let beta = node.get_attr("beta")?;
    Ok((Box::new(tractops::nn::ScaledTanh::new(alpha, beta)),vec!()))
}

element_map_with_params!(Shrink, [f16, f32, f64], { bias: f32, lambd: f32 },
    fn eval_one<T>(s: &Shrink, x:T) -> T
    where T: Datum+::num_traits::Float, f32: ::num_traits::AsPrimitive<T>
    {
        use num_traits::AsPrimitive;
        if x < -s.lambd.as_() { x + s.bias.as_() } else if x > s.lambd.as_() { x - s.bias.as_() } else { T::zero() }
    }
);

pub fn shrink(_ctx: &ParsingContext, node: &NodeProto) -> TractResult<(Box<dyn InferenceOp>,Vec<String>)> {
    let bias = node.get_attr_opt("bias")?.unwrap_or(0.0);
    let lambd = node.get_attr_opt("lambd")?.unwrap_or(0.5);
    Ok((Box::new(Shrink::new(bias, lambd)),vec!()))
}

pub fn selu(_ctx: &ParsingContext, node: &NodeProto) -> TractResult<(Box<dyn InferenceOp>,Vec<String>)> {
    let alpha = node.get_attr_opt("alpha")?.unwrap_or(1.67326);
    let gamma = node.get_attr_opt("gamma")?.unwrap_or(1.0507);
    Ok((Box::new(tractops::nn::Selu::new(alpha, gamma)), vec!()))
}

pub fn thresholded_relu(_ctx: &ParsingContext, node: &NodeProto) -> TractResult<(Box<dyn InferenceOp>,Vec<String>)> {
    let alpha = node.get_attr_opt("alpha")?.unwrap_or(1.);
    Ok((Box::new(tractops::nn::ThresholdedRelu::new(alpha)),vec!()))
}
