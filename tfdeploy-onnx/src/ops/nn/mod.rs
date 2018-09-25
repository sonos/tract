use tfdeploy::ops::prelude::*;
use tfdeploy::ops as tfdops;

use pb::NodeProto;
use ops::OpRegister;

pub fn register_all_ops(reg: &mut OpRegister) {
    reg.insert("Conv", conv);
    reg.insert("Relu", |_| Ok(Box::new(tfdops::nn::Relu::default())));
    reg.insert("Sigmoid", |_| Ok(Box::new(tfdops::nn::Sigmoid::default())));
}

pub fn conv(node: &NodeProto) -> TfdResult<Box<Op>> {
    let dilations = node.get_attr_opt_ints("dilations")?
        .map(|i| i.iter().map(|&i| i as usize).collect());
    let kernel_shape = node.get_attr_opt_ints("kernel_shape")?
        .map(|i| i.iter().map(|&i| i as usize).collect());
    let pads = node.get_attr_opt_ints("pads")?
        .map(|i| i.iter().map(|&i| i as usize).collect());
    let strides = node.get_attr_opt_ints("strides")?
        .map(|i| i.iter().map(|&i| i as usize).collect());
    Ok(Box::new(tfdops::nn::Conv::new(false, false, dilations, kernel_shape, pads, strides)))
}
