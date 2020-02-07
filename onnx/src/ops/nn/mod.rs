use tract_core::internal::*;
use tract_core::infer::*;
use tract_core::ops as tractops;
use tract_core::ops::cnn::PaddingSpec;
use tract_core::ops::nn::DataFormat;

use crate::model::{OnnxOpRegister, ParsingContext};
use crate::pb::NodeProto;
use crate::pb_helpers::OptionExt;

use num_traits::AsPrimitive;
use tractops::nn::Reducer;

mod batch_norm;
mod dropout;

fn reduce(node: &NodeProto, reducer: Reducer) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let axes = node.get_attr_opt_vec("axes")?;
    let keep_dims = node.get_attr_opt("keepdims")?.unwrap_or(1i64) == 1;
    Ok((Box::new(tractops::nn::Reduce::new(axes, keep_dims, reducer)), vec![]))
}

pub fn register_all_ops(reg: &mut OnnxOpRegister) {
    reg.insert("ArgMax", arg_max_min);
    reg.insert("ArgMin", arg_max_min);
    reg.insert("AveragePool", average_pool);
    reg.insert("BatchNormalization", batch_normalization);
    reg.insert("Conv", conv);
    reg.insert("ConvInteger", conv_integer);
    reg.insert("Dropout", dropout::dropout);
    reg.insert("Elu", elu);
    reg.insert("GlobalAveragePool", |_, _| {
        Ok((Box::new(tractops::nn::GlobalAvgPool::default()), vec![]))
    });
    reg.insert("GlobalLpPool", global_lp_pool);
    reg.insert("GlobalMaxPool", |_, _| {
        Ok((Box::new(tractops::nn::GlobalMaxPool::default()), vec![]))
    });
    reg.insert("Hardmax", layer_hard_max);
    reg.insert("HardSigmoid", hard_sigmoid);
    reg.insert("LeakyRelu", leaky_relu);
    reg.insert("LogSoftmax", layer_log_soft_max);
    reg.insert("LRN", lrn);
    reg.insert("MaxPool", max_pool);
    reg.insert("ParametricSoftplus", parametric_softplus);
    reg.insert("QLinearConv", conv_qlinear);
    reg.insert("PRelu", |_, _| Ok((Box::new(prelu::bin()), vec![])));
    reg.insert("ReduceL1", |_, node| reduce(node, Reducer::L1));
    reg.insert("ReduceL2", |_, node| reduce(node, Reducer::L2));
    reg.insert("ReduceLogSum", |_, node| reduce(node, Reducer::LogSum));
    reg.insert("ReduceLogSumExp", |_, node| reduce(node, Reducer::LogSumExp));
    reg.insert("ReduceMax", |_, node| reduce(node, Reducer::Max));
    reg.insert("ReduceMean", |_, node| reduce(node, Reducer::Mean));
    reg.insert("ReduceMin", |_, node| reduce(node, Reducer::Min));
    reg.insert("ReduceProd", |_, node| reduce(node, Reducer::Prod));
    reg.insert("ReduceSum", |_, node| reduce(node, Reducer::Sum));
    reg.insert("ReduceSumSquare", |_, node| reduce(node, Reducer::SumSquare));
    reg.insert("Relu", |_, _| Ok((Box::new(tractops::math::scalar_max((0.0).into())), vec![])));
    reg.insert("ScaledTanh", scaled_tanh);
    reg.insert("Shrink", shrink);
    reg.insert("ThresholdedRelu", thresholded_relu);
    reg.insert("Selu", selu);
    reg.insert("Sigmoid", |_, _| Ok((Box::new(tractops::nn::sigmoid()), vec![])));
    reg.insert("Softmax", layer_soft_max);
    reg.insert("Softplus", |_, _| Ok((Box::new(tractops::nn::softplus()), vec![])));
    reg.insert("Softsign", |_, _| Ok((Box::new(tractops::nn::softsign()), vec![])));
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

pub fn arg_max_min(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let max = node.op_type == "ArgMax";
    let axis = node.get_attr_opt("axis")?.unwrap_or(0);
    let keepdims = node.get_attr_opt("keepdims")?.unwrap_or(true);
    Ok((Box::new(tractops::nn::ArgMaxMin::new(max, axis, keepdims)), vec![]))
}

pub fn batch_normalization(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let epsilon = node.get_attr_opt("epsilon")?.unwrap_or(1e-5);
    let spatial = node.get_attr_opt("spatial")?.unwrap_or(0);
    assert_eq!(spatial, 0);
    Ok((Box::new(batch_norm::BatchNorm::new(DataFormat::NCHW, epsilon, spatial != 0)), vec![]))
}

fn common_conv(node: &NodeProto) -> TractResult<tractops::cnn::Conv> {
    let mut op = tractops::cnn::Conv::default().padding(pad(node)?);
    if let Some(kernel_shape) = node.get_attr_opt_tvec("kernel_shape")? {
        op = op.kernel_shape(kernel_shape);
    }
    if let Some(group) = node.get_attr_opt("group")? {
        op = op.group(group);
    }
    if let Some(v) = dilations(node)? {
        op = op.dilations(v);
    }
    if let Some(v) = strides(node)? {
        op = op.strides(v);
    }
    Ok(op)
}

pub fn conv(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let mut op = common_conv(node)?;
    if node.input.len() == 3 {
        op = op.bias_input(2);
    }
    Ok((Box::new(op), vec![]))
}

pub fn conv_integer(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let mut op = common_conv(node)?;
    let mut options = crate::model::optional_inputs(node).skip(2);
    if let Some(i) = options.next().unwrap() {
        op = op.x_zero_point_input(i);
    }
    if let Some(i) = options.next().unwrap() {
        op = op.k_zero_point_input(i);
    }
    op.override_output_datum_type = Some(i32::datum_type());
    Ok((Box::new(op), vec![]))
}

pub fn conv_qlinear(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let mut op = common_conv(node)?;
    op.x_scale_input = Some(1);
    op.x_zero_point_input = Some(2);
    op.k_input = Some(3);
    op.k_scale_input = Some(4);
    op.k_zero_point_input = Some(5);
    op.y_scale_input = Some(6);
    op.y_zero_point_input = Some(7);
    if node.input.len() == 9 {
        op.bias_input = Some(8);
    }
    op.override_bias_datum_type = Some(i32::datum_type());
    Ok((Box::new(op), vec![]))
}

pub fn average_pool(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let kernel_shape = node.get_attr_tvec("kernel_shape")?;
    let pad = pad(node)?;
    let strides = strides(node)?;
    let count_include_pad = node.get_attr_opt("count_include_pad")?.unwrap_or(false);
    Ok((
        Box::new(tractops::cnn::AvgPool::new(
            tractops::cnn::PoolSpec::new(DataFormat::NCHW, kernel_shape, pad, None, strides, None),
            count_include_pad,
        )),
        vec![],
    ))
}

pub fn elu(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let alpha = node.get_attr_opt("alpha")?.unwrap_or(1.);
    Ok((Box::new(tractops::nn::elu(alpha)), vec![]))
}

pub fn global_lp_pool(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let p: usize = node.get_attr_opt("p")?.unwrap_or(2);
    Ok((Box::new(tractops::nn::GlobalLpPool::new(p)), vec![]))
}

pub fn hard_sigmoid(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let alpha = node.get_attr_opt("alpha")?.unwrap_or(0.2);
    let beta = node.get_attr_opt("beta")?.unwrap_or(0.5);
    Ok((Box::new(tractops::nn::hard_sigmoid(alpha, beta)), vec![]))
}

pub fn layer_hard_max(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let axis = node.get_attr_opt("axis")?.unwrap_or(1);
    Ok((Box::new(tractops::nn::LayerHardmax::new(axis)), vec![]))
}

pub fn layer_log_soft_max(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let axis = node.get_attr_opt("axis")?.unwrap_or(1);
    Ok((Box::new(tractops::nn::LayerLogSoftmax::new(axis)), vec![]))
}

pub fn layer_soft_max(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let axis = node.get_attr_opt("axis")?.unwrap_or(1);
    Ok((Box::new(tractops::nn::LayerSoftmax::new(axis)), vec![]))
}

pub fn leaky_relu(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let alpha = node.get_attr_opt("alpha")?.unwrap_or(0.01);
    Ok((Box::new(tractops::nn::leaky_relu(alpha)), vec![]))
}

pub fn lrn(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let alpha = node.get_attr_opt("alpha")?.unwrap_or(0.0001);
    let beta = node.get_attr_opt("beta")?.unwrap_or(0.75);
    let bias = node.get_attr_opt("bias")?.unwrap_or(1.);
    let size = node.get_attr("size")?;
    Ok((Box::new(tractops::nn::Lrn::new(alpha, beta, bias, size)), vec![]))
}

pub fn max_pool(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let kernel_shape = node.get_attr_tvec("kernel_shape")?;
    let pad = pad(node)?;
    let strides = strides(node)?;
    Ok((
        Box::new(tractops::cnn::MaxPool::new(
            tractops::cnn::PoolSpec::new(DataFormat::NCHW, kernel_shape, pad, None, strides, None),
            if node.output.len() == 2 { Some(DatumType::I64) } else { None },
        )),
        vec![],
    ))
}

pub fn parametric_softplus(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let alpha = node.get_attr("alpha")?;
    let beta = node.get_attr("beta")?;
    Ok((Box::new(tractops::nn::parametric_softplus(alpha, beta)), vec![]))
}

bin_to_super_type!(prelu, Prelu, declutter_bin: prelu_to_prelu_unary,
  [f16,f32,f64] => |c, &a, &b| *c = if a < 0f32.into() { a * b } else { a });

element_wise!(prelu_unary, PreluUnary { b: f32 },
    [f32] => |op, xs| {
        xs.iter_mut().for_each(|x| *x = if *x < 0.0 { *x * op.b } else { *x });
        Ok(())
    },
    [f64] => |op, xs| {
        xs.iter_mut().for_each(|x| *x = if *x < 0.0 { *x * op.b as f64 } else { *x });
        Ok(())
    }
);

fn prelu_to_prelu_unary(
    _prelu: &Prelu,
    model: &TypedModel,
    node: &TypedNode,
) -> TractResult<Option<TypedModelPatch>> {
    if let Some(ref b) = model.outlet_fact(node.inputs[1])?.konst {
        let b = b.cast_to::<f32>()?;
        let op = prelu_unary(*b.to_scalar::<f32>()?);
        Ok(Some(TypedModelPatch::replace_single_op(model, node, &node.inputs[0..1], op)?))
    } else {
        Ok(None)
    }
}

pub fn scaled_tanh(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let alpha = node.get_attr("alpha")?;
    let beta = node.get_attr("beta")?;
    Ok((Box::new(tractops::nn::scaled_tanh(alpha, beta)), vec![]))
}

element_wise!(shrink_op, Shrink { bias: f32, lambd: f32 },
    [f16,f32,f64] => |s, xs| {
        xs.iter_mut().for_each(|x| *x = shrink_value(*x, s));
        Ok(())
});

fn shrink_value<T>(x: T, s: &Shrink) -> T
where
    T: Datum + ::num_traits::Float,
    f32: ::num_traits::AsPrimitive<T>,
{
    if x < -s.lambd.as_() {
        x + s.bias.as_()
    } else if x > s.lambd.as_() {
        x - s.bias.as_()
    } else {
        T::zero()
    }
}

pub fn shrink(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let bias = node.get_attr_opt("bias")?.unwrap_or(0.0);
    let lambd = node.get_attr_opt("lambd")?.unwrap_or(0.5);
    Ok((Box::new(shrink_op(bias, lambd)), vec![]))
}

pub fn selu(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let alpha = node.get_attr_opt("alpha")?.unwrap_or(1.67326);
    let gamma = node.get_attr_opt("gamma")?.unwrap_or(1.0507);
    Ok((Box::new(tractops::nn::selu(alpha, gamma)), vec![]))
}

pub fn thresholded_relu(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let alpha = node.get_attr_opt("alpha")?.unwrap_or(1.);
    Ok((Box::new(tractops::nn::threshold_relu(alpha)), vec![]))
}
