use tfdeploy::ops as tfdops;

use ops::OpRegister;

pub fn register_all_ops(reg: &mut OpRegister) {
    reg.insert("Relu", |_| Ok(Box::new(tfdops::nn::Relu::default())));
    reg.insert("Sigmoid", |_| Ok(Box::new(tfdops::nn::Sigmoid::default())));
}

