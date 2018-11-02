use tract_core::ops as tfdops;
use tract_core::ops::nn::{DataFormat, PaddingSpec};
use tract_core::ops::prelude::*;

use ops::OpRegister;
use pb::NodeProto;

macro_rules! reduce {
    ($id:ident) => {
        |node| {
            let axes = node
                .get_attr_opt_ints("axes")?
                .map(|axes| axes.iter().map(|&i| i as usize).collect());
            let keep_dims = node.get_attr_opt_int("keepdims")?.unwrap_or(1i64) == 1;
            Ok(Box::new(tfdops::nn::Reduce::new(
                axes,
                keep_dims,
                tfdops::nn::Reducer::$id,
            )))
        }
    };
}

pub fn register_all_ops(reg: &mut OpRegister) {
    reg.insert("ArgMax", arg_max_min);
    reg.insert("ArgMin", arg_max_min);
    reg.insert("AveragePool", average_pool);
    reg.insert("BatchNormalization", batch_normalization);
    reg.insert("Conv", conv);
    reg.insert("Dropout", |_| {
        Ok(Box::new(tfdops::identity::Identity::default()))
    });
    reg.insert("Elu", elu);
    reg.insert("GlobalAveragePool", |_| {
        Ok(Box::new(tfdops::nn::GlobalAvgPool::default()))
    });
    reg.insert("GlobalLpPool", global_lp_pool);
    reg.insert("GlobalMaxPool", |_| {
        Ok(Box::new(tfdops::nn::GlobalMaxPool::default()))
    });
    reg.insert("Hardmax", layer_hard_max);
    reg.insert("HardSigmoid", hard_sigmoid);
    reg.insert("LeakyRelu", leaky_relu);
    reg.insert("LogSoftmax", layer_log_soft_max);
    reg.insert("LRN", lrn);
    reg.insert("MaxPool", max_pool);
    reg.insert("ParametricSoftplus", parametric_softplus);
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
    reg.insert("Relu", |_| Ok(Box::new(tfdops::nn::Relu::default())));
    reg.insert("ScaledTanh", scaled_tanh);
    reg.insert("ThresholdedRelu", thresholded_relu);
    reg.insert("Selu", selu);
    reg.insert("Sigmoid", |_| Ok(Box::new(tfdops::nn::Sigmoid::default())));
    reg.insert("Softmax", layer_soft_max);
    reg.insert("Softplus", |_| {
        Ok(Box::new(tfdops::nn::Softplus::default()))
    });
    reg.insert("Softsign", |_| {
        Ok(Box::new(tfdops::nn::Softsign::default()))
    });
}

fn pad(node: &NodeProto) -> TfdResult<PaddingSpec> {
    if let Some(pads) = node.get_attr_opt_ints("pads")? {
        let len = pads.len();
        return Ok(PaddingSpec::Explicit(
            pads.iter().take(len / 2).map(|&i| i as usize).collect(),
            pads.iter().skip(len / 2).map(|&i| i as usize).collect(),
        ));
    }
    match node.get_attr_opt_str("auto_pad")?.unwrap_or("NOTSET") {
        "NOTSET" => Ok(PaddingSpec::Valid),
        "VALID" => Ok(PaddingSpec::Valid),
        "SAME_UPPER" => Ok(PaddingSpec::SameUpper),
        "SAME_LOWER" => Ok(PaddingSpec::SameLower),
        e => bail!("Unexpected auto_pad value {}", e),
    }
}

fn dilations(node: &NodeProto) -> TfdResult<Option<Vec<usize>>> {
    Ok(node
        .get_attr_opt_ints("dilations")?
        .map(|i| i.iter().map(|&i| i as usize).collect()))
}

fn strides(node: &NodeProto) -> TfdResult<Option<Vec<usize>>> {
    Ok(node
        .get_attr_opt_ints("strides")?
        .map(|i| i.iter().map(|&i| i as usize).collect()))
}

pub fn arg_max_min(node: &NodeProto) -> TfdResult<Box<Op>> {
    let max = node.get_op_type() == "ArgMax";
    let axis = node
        .get_attr_opt_int("axis")?
        .map(|i| i as usize)
        .unwrap_or(0);
    let keepdims = node.get_attr_opt_int("keepdims")?.unwrap_or(1i64) == 1;
    Ok(Box::new(tfdops::nn::ArgMaxMin::new(max, axis, keepdims)))
}

pub fn batch_normalization(node: &NodeProto) -> TfdResult<Box<Op>> {
    let epsilon = node.get_attr_opt_float("epsilon")?.unwrap_or(1e-5);
    let spatial = node.get_attr_opt_int("spatial")?.unwrap_or(0);
    assert_eq!(spatial, 0);
    Ok(Box::new(tfdops::nn::BatchNorm::new(
        DataFormat::NCHW,
        epsilon,
        spatial != 0,
    )))
}

pub fn conv(node: &NodeProto) -> TfdResult<Box<Op>> {
    let kernel_shape = node
        .get_attr_opt_ints("kernel_shape")?
        .map(|i| i.iter().map(|&i| i as usize).collect());
    let group = node.get_attr_opt_int("group")?.unwrap_or(1);
    Ok(Box::new(tfdops::nn::Conv::new(
        DataFormat::NCHW,
        false,
        dilations(node)?,
        kernel_shape,
        pad(node)?,
        strides(node)?,
        group as usize,
    )))
}

pub fn average_pool(node: &NodeProto) -> TfdResult<Box<Op>> {
    let kernel_shape: Vec<usize> = node
        .get_attr_ints("kernel_shape")?
        .iter()
        .map(|&i| i as usize)
        .collect();
    let pad = pad(node)?;
    let strides = strides(node)?;
    let count_include_pad = node.get_attr_opt_int("count_include_pad")?.unwrap_or(0) != 0;
    Ok(Box::new(tfdops::nn::AvgPool::new(
        DataFormat::NCHW,
        kernel_shape,
        pad,
        strides,
        count_include_pad,
    )))
}

pub fn elu(node: &NodeProto) -> TfdResult<Box<Op>> {
    let alpha = node.get_attr_opt_float("alpha")?.unwrap_or(1.0);
    Ok(Box::new(tfdops::nn::Elu::new(alpha)))
}

pub fn global_lp_pool(node: &NodeProto) -> TfdResult<Box<Op>> {
    let p: usize = node.get_attr_opt_int("p")?.map(|i| i as usize).unwrap_or(2);
    Ok(Box::new(tfdops::nn::GlobalLpPool::new(p)))
}

pub fn hard_sigmoid(node: &NodeProto) -> TfdResult<Box<Op>> {
    let alpha = node.get_attr_opt_float("alpha")?.unwrap_or(0.2);
    let beta = node.get_attr_opt_float("beta")?.unwrap_or(0.5);
    Ok(Box::new(tfdops::nn::Hardsigmoid::new(alpha, beta)))
}

pub fn layer_hard_max(node: &NodeProto) -> TfdResult<Box<Op>> {
    let axis: isize = node
        .get_attr_opt_int("axis")?
        .map(|i| i as isize)
        .unwrap_or(1);
    Ok(Box::new(tfdops::nn::LayerHardmax::new(axis)))
}

pub fn layer_log_soft_max(node: &NodeProto) -> TfdResult<Box<Op>> {
    let axis: isize = node
        .get_attr_opt_int("axis")?
        .map(|i| i as isize)
        .unwrap_or(1);
    Ok(Box::new(tfdops::nn::LayerLogSoftmax::new(axis)))
}

pub fn layer_soft_max(node: &NodeProto) -> TfdResult<Box<Op>> {
    let axis: isize = node
        .get_attr_opt_int("axis")?
        .map(|i| i as isize)
        .unwrap_or(1);
    Ok(Box::new(tfdops::nn::LayerSoftmax::new(axis)))
}

pub fn leaky_relu(node: &NodeProto) -> TfdResult<Box<Op>> {
    let alpha = node.get_attr_opt_float("alpha")?.unwrap_or(0.01);
    Ok(Box::new(tfdops::nn::LeakyRelu::new(alpha)))
}

pub fn lrn(node: &NodeProto) -> TfdResult<Box<Op>> {
    let alpha = node.get_attr_opt_float("alpha")?.unwrap_or(0.0001);
    let beta = node.get_attr_opt_float("beta")?.unwrap_or(0.75);
    let bias = node.get_attr_opt_float("bias")?.unwrap_or(1.0);
    let size: usize = node.get_attr_int("size")? as usize;
    Ok(Box::new(tfdops::nn::Lrn::new(alpha, beta, bias, size)))
}

pub fn max_pool(node: &NodeProto) -> TfdResult<Box<Op>> {
    let kernel_shape: Vec<usize> = node
        .get_attr_ints("kernel_shape")?
        .iter()
        .map(|&i| i as usize)
        .collect();
    let pad = pad(node)?;
    let strides = strides(node)?;
    Ok(Box::new(tfdops::nn::MaxPool::new(
        DataFormat::NCHW,
        kernel_shape,
        pad,
        strides,
        node.get_output().len() == 2,
    )))
}

pub fn parametric_softplus(node: &NodeProto) -> TfdResult<Box<Op>> {
    let alpha = node.get_attr_float("alpha")?;
    let beta = node.get_attr_float("beta")?;
    Ok(Box::new(tfdops::nn::ParametricSoftplus::new(alpha, beta)))
}

pub fn scaled_tanh(node: &NodeProto) -> TfdResult<Box<Op>> {
    let alpha = node.get_attr_float("alpha")?;
    let beta = node.get_attr_float("beta")?;
    Ok(Box::new(tfdops::nn::ScaledTanh::new(alpha, beta)))
}

pub fn selu(node: &NodeProto) -> TfdResult<Box<Op>> {
    let alpha = node.get_attr_opt_float("alpha")?.unwrap_or(1.67326);
    let gamma = node.get_attr_opt_float("gamma")?.unwrap_or(1.0507);
    Ok(Box::new(tfdops::nn::Selu::new(alpha, gamma)))
}

pub fn thresholded_relu(node: &NodeProto) -> TfdResult<Box<Op>> {
    let alpha = node.get_attr_opt_float("alpha")?.unwrap_or(1.0);
    Ok(Box::new(tfdops::nn::ThresholdedRelu::new(alpha)))
}

