use tract_hir::internal::*;
use tract_hir::ops;
use tract_hir::ops::{cnn, nn};

use crate::model::{OnnxOpRegister, ParsingContext};
use crate::pb::NodeProto;
use crate::pb_helpers::OptionExt;

mod batch_norm;
mod dropout;
mod instance_norm;
mod lrn;

pub fn arg_max_min(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let axis = node.get_attr_opt("axis")?.unwrap_or(0);
    let keepdims = node.get_attr_opt("keepdims")?.unwrap_or(true);
    let take_last = node.get_attr_opt("select_last_index")?.unwrap_or(false);
    let red = if node.op_type == "ArgMax" { nn::Reducer::ArgMax(take_last) } else { nn::Reducer::ArgMin(take_last) };
    Ok((expand(nn::Reduce::new(Some(vec![axis]), keepdims, red)), vec![]))
}

fn reduce(
    node: &NodeProto,
    reducer: nn::Reducer,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let axes = node.get_attr_opt_vec("axes")?;
    let keep_dims = node.get_attr_opt("keepdims")?.unwrap_or(1i64) == 1;
    Ok((expand(ops::nn::Reduce::new(axes, keep_dims, reducer)), vec![]))
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
    reg.insert("GlobalAveragePool", |_, _| Ok((expand(ops::nn::GlobalAvgPool), vec![])));
    reg.insert("GlobalLpPool", global_lp_pool);
    reg.insert("GlobalMaxPool", |_, _| Ok((expand(ops::nn::GlobalMaxPool), vec![])));
    reg.insert("Hardmax", layer_hard_max);
    reg.insert("HardSigmoid", hard_sigmoid);
    reg.insert("InstanceNormalization", instance_norm::instance_normalization);
    reg.insert("LeakyRelu", leaky_relu);
    reg.insert("LogSoftmax", layer_log_soft_max);
    reg.insert("LRN", lrn::lrn);
    reg.insert("MaxPool", max_pool);
    reg.insert("ParametricSoftplus", parametric_softplus);
    reg.insert("QLinearConv", conv_qlinear);
    reg.insert("PRelu", |_, _| Ok((expand(Prelu), vec![])));
    reg.insert("ReduceL1", |_, node| reduce(node, nn::Reducer::L1));
    reg.insert("ReduceL2", |_, node| reduce(node, nn::Reducer::L2));
    reg.insert("ReduceLogSum", |_, node| reduce(node, nn::Reducer::LogSum));
    reg.insert("ReduceLogSumExp", |_, node| reduce(node, nn::Reducer::LogSumExp));
    reg.insert("ReduceMax", |_, node| reduce(node, nn::Reducer::Max));
    reg.insert("ReduceMean", |_, node| reduce(node, nn::Reducer::Mean));
    reg.insert("ReduceMin", |_, node| reduce(node, nn::Reducer::Min));
    reg.insert("ReduceProd", |_, node| reduce(node, nn::Reducer::Prod));
    reg.insert("ReduceSum", |_, node| reduce(node, nn::Reducer::Sum));
    reg.insert("ReduceSumSquare", |_, node| reduce(node, nn::Reducer::SumSquare));
    reg.insert("Relu", |_, _| Ok((expand(ops::activations::Clip::new(Some(0.0), None)), vec![])));
    reg.insert("ScaledTanh", scaled_tanh);
    reg.insert("Shrink", shrink);
    reg.insert("ThresholdedRelu", thresholded_relu);
    reg.insert("Selu", selu);
    reg.insert("Sigmoid", |_, _| Ok((Box::new(ops::nn::sigmoid()), vec![])));
    reg.insert("Softmax", layer_soft_max);
    reg.insert("Softplus", |_, _| Ok((expand(ops::activations::Softplus), vec![])));
    reg.insert("Softsign", |_, _| Ok((expand(ops::activations::Softsign), vec![])));
}

fn pad(node: &NodeProto) -> TractResult<cnn::PaddingSpec> {
    let ceil_mode = node.get_attr_opt::<isize>("ceil_mode")?.unwrap_or(0) == 1;
    let default = match node.get_attr_opt_vec::<isize>("kernel_shape")? {
        Some(shape) => {
            cnn::PaddingSpec::Explicit(tvec!(0; shape.len()), tvec!(0; shape.len()), ceil_mode)
        }
        None => cnn::PaddingSpec::Valid,
    };
    if let Some(pads) = node.get_attr_opt_tvec("pads")? {
        let len = pads.len();
        return Ok(cnn::PaddingSpec::Explicit(
            pads.iter().cloned().take(len / 2).collect(),
            pads.iter().cloned().skip(len / 2).collect(),
            ceil_mode,
        ));
    }
    Ok(node
        .get_attr_opt("auto_pad")?
        .and_try(|s| {
            node.check_value(
                "auto_pad",
                match s {
                    "NOTSET" => Ok(default.clone()),
                    "VALID" => Ok(cnn::PaddingSpec::Valid),
                    "SAME_UPPER" => Ok(cnn::PaddingSpec::SameUpper),
                    "SAME_LOWER" => Ok(cnn::PaddingSpec::SameLower),
                    _ => Err(s),
                },
            )
        })?
        .unwrap_or(default))
}

fn dilations(node: &NodeProto) -> TractResult<Option<TVec<usize>>> {
    node.get_attr_opt_tvec("dilations")
}

fn strides(node: &NodeProto) -> TractResult<Option<TVec<usize>>> {
    node.get_attr_opt_tvec("strides")
}

pub fn batch_normalization(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let epsilon = node.get_attr_opt("epsilon")?.unwrap_or(1e-5);
    let spatial = node.get_attr_opt("spatial")?.unwrap_or(1);
    if spatial != 1 {
        bail!("BatchNormalization: attribute 'spatial' is not supported (deprecated by ONNX operator set 9)")
    }
    Ok((expand(batch_norm::BatchNorm::new(nn::DataFormat::NCHW, epsilon, spatial != 0)), vec![]))
}

fn common_conv(node: &NodeProto) -> TractResult<cnn::Conv> {
    let mut op = ops::cnn::Conv::default().padding(pad(node)?);
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
    Ok((expand(op), vec![]))
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
    Ok((expand(op), vec![]))
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
    Ok((expand(op), vec![]))
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
        Box::new(cnn::SumPool::new(
            cnn::PoolSpec::new(nn::DataFormat::NCHW, kernel_shape, pad, None, strides, None),
            count_include_pad,
            true,
        )),
        vec![],
    ))
}

pub fn elu(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let alpha = node.get_attr_opt("alpha")?.unwrap_or(1.);
    Ok((expand(ops::activations::Elu(alpha)), vec![]))
}

pub fn global_lp_pool(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let p: usize = node.get_attr_opt("p")?.unwrap_or(2);
    Ok((expand(ops::nn::GlobalLpPool::new(p)), vec![]))
}

pub fn hard_sigmoid(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let alpha = node.get_attr_opt("alpha")?.unwrap_or(0.2);
    let beta = node.get_attr_opt("beta")?.unwrap_or(0.5);
    Ok((expand(ops::activations::HardSigmoid(alpha, beta)), vec![]))
}

pub fn layer_hard_max(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let axis = node.get_attr_opt("axis")?.unwrap_or(1);
    Ok((Box::new(ops::nn::LayerHardmax::new(axis)), vec![]))
}

pub fn layer_log_soft_max(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let axis = node.get_attr_opt("axis")?.unwrap_or(1);
    Ok((expand(ops::nn::LayerLogSoftmax::new(axis)), vec![]))
}

pub fn layer_soft_max(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let axis = node.get_attr_opt("axis")?.unwrap_or(1);
    Ok((expand(ops::nn::LayerSoftmax::new(axis)), vec![]))
}

pub fn leaky_relu(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let alpha = node.get_attr_opt("alpha")?.unwrap_or(0.01);
    Ok((expand(ops::activations::LeakyRelu(alpha)), vec![]))
}

pub fn max_pool(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let kernel_shape = node.get_attr_tvec("kernel_shape")?;
    let pad = pad(node)?;
    let strides = strides(node)?;
    Ok((
        Box::new(cnn::MaxPool::new(
            cnn::PoolSpec::new(nn::DataFormat::NCHW, kernel_shape, pad, None, strides, None),
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
    Ok((expand(ops::activations::ParametricSoftplus(alpha, beta)), vec![]))
}

#[derive(Debug, Clone, Hash)]
struct Prelu;
tract_data::impl_dyn_hash!(Prelu);

impl Expansion for Prelu {
    fn name(&self) -> Cow<str> {
        "Prelu".into()
    }

    op_onnx!();

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(inputs, 2)?;
        check_output_arity(outputs, 1)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].shape, &outputs[0].shape)?;
        Ok(())
    }

    fn wire(
        &self,
        name: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let a = inputs[0];
        let mut b = inputs[1];
        let rank = model.outlet_fact(a)?.rank();
        while model.outlet_fact(b)?.rank() < rank {
            b = model.wire_node(
                format!("{}.add-axis-{}", name, model.outlet_fact(b)?.rank()),
                AxisOp::Add(0),
                &[b],
            )?[0];
        }
        let mut zero = tensor0(0.0).cast_to_dt(model.outlet_fact(a)?.datum_type)?.into_owned();
        while zero.rank() < rank {
            zero.insert_axis(0)?;
        }
        let ab = model.wire_node(
            format!("{}.mul", name),
            tract_hir::ops::math::mul::bin_typed(),
            &[a, b],
        )?[0];
        let test = model.wire_node(
            name.to_string() + ".test",
            tract_hir::ops::logic::greater::unary(zero.into_arc_tensor()),
            &[a],
        )?;
        model.wire_node(name.to_string() + ".iff", tract_hir::ops::logic::Iff, &[test[0], ab, a])
    }
}

pub fn scaled_tanh(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let alpha = node.get_attr("alpha")?;
    let beta = node.get_attr("beta")?;
    Ok((expand(ops::activations::ScaledTanh(alpha, beta)), vec![]))
}

pub fn shrink(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let bias = node.get_attr_opt("bias")?.unwrap_or(0.0);
    let lambd = node.get_attr_opt("lambd")?.unwrap_or(0.5);
    Ok((expand(ops::activations::Shrink(bias, lambd)), vec![]))
}

pub fn selu(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let alpha = node.get_attr_opt("alpha")?.unwrap_or(1.67326);
    let gamma = node.get_attr_opt("gamma")?.unwrap_or(1.0507);
    Ok((expand(ops::activations::Selu(alpha, gamma)), vec![]))
}

pub fn thresholded_relu(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let alpha = node.get_attr_opt("alpha")?.unwrap_or(1.);
    Ok((expand(ops::activations::ThresholdRelu(alpha)), vec![]))
}
