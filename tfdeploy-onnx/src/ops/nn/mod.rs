use tfdeploy::ops as tfdops;
use tfdeploy::ops::nn::PaddingSpec;
use tfdeploy::ops::prelude::*;

use ops::OpRegister;
use pb::NodeProto;

pub fn register_all_ops(reg: &mut OpRegister) {
    reg.insert("AveragePool", average_pool);
    reg.insert("Conv", conv);
    reg.insert("Dropout", |_| Ok(Box::new(tfdops::identity::Identity::default())));
    reg.insert("GlobalAveragePool", |_| Ok(Box::new(tfdops::nn::GlobalAvgPool::default())));
    reg.insert("GlobalLpPool", global_lp_pool);
    reg.insert("GlobalMaxPool", |_| Ok(Box::new(tfdops::nn::GlobalMaxPool::default())));
    reg.insert("Hardmax", layer_hard_max);
    reg.insert("LogSoftmax", layer_log_soft_max);
    reg.insert("Softmax", layer_soft_max);
    reg.insert("MaxPool", max_pool);
    reg.insert("Relu", |_| Ok(Box::new(tfdops::nn::Relu::default())));
    reg.insert("Sigmoid", |_| Ok(Box::new(tfdops::nn::Sigmoid::default())));
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

pub fn conv(node: &NodeProto) -> TfdResult<Box<Op>> {
    let kernel_shape = node
        .get_attr_opt_ints("kernel_shape")?
        .map(|i| i.iter().map(|&i| i as usize).collect());
    Ok(Box::new(tfdops::nn::Conv::new(
        false,
        false,
        dilations(node)?,
        kernel_shape,
        pad(node)?,
        strides(node)?,
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
        false,
        kernel_shape,
        pad,
        strides,
        count_include_pad,
    )))
}

pub fn global_lp_pool(node: &NodeProto) -> TfdResult<Box<Op>> {
    let p: usize = node
        .get_attr_opt_int("p")?
        .map(|i| i as usize)
        .unwrap_or(2);
    Ok(Box::new(tfdops::nn::GlobalLpPool::new(p)))
}

pub fn layer_hard_max(node: &NodeProto) -> TfdResult<Box<Op>> {
    let axis: usize = node
        .get_attr_opt_int("axis")?
        .map(|i| i as usize)
        .unwrap_or(1);
    Ok(Box::new(tfdops::nn::LayerHardmax::new(axis)))
}

pub fn layer_log_soft_max(node: &NodeProto) -> TfdResult<Box<Op>> {
    let axis: usize = node
        .get_attr_opt_int("axis")?
        .map(|i| i as usize)
        .unwrap_or(1);
    Ok(Box::new(tfdops::nn::LayerLogSoftmax::new(axis)))
}

pub fn layer_soft_max(node: &NodeProto) -> TfdResult<Box<Op>> {
    let axis: usize = node
        .get_attr_opt_int("axis")?
        .map(|i| i as usize)
        .unwrap_or(1);
    Ok(Box::new(tfdops::nn::LayerSoftmax::new(axis)))
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
        false,
        kernel_shape,
        pad,
        strides,
    )))
}

