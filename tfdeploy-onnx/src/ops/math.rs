use tfdeploy::ops as tfdops;

use tfdeploy::ops::prelude::*;
use pb::NodeProto;
use ops::OpRegister;

pub fn register_all_ops(reg: &mut OpRegister) {
    reg.insert("Add", |_| Ok(Box::new(tfdops::math::Add::default())));
    reg.insert("Sub", |_| Ok(Box::new(tfdops::math::Sub::default())));
    reg.insert("Mul", |_| Ok(Box::new(tfdops::math::Mul::default())));
    reg.insert("Div", |_| Ok(Box::new(tfdops::math::Div::default())));

    reg.insert("Sum", |_| Ok(Box::new(tfdops::math::AddN::default())));
    reg.insert("Max", |_| Ok(Box::new(tfdops::math::MaxN::default())));
    reg.insert("Min", |_| Ok(Box::new(tfdops::math::MinN::default())));
    reg.insert("Mean", |_| Ok(Box::new(tfdops::math::MeanN::default())));

    reg.insert("Abs", |_| Ok(Box::new(tfdops::math::Abs::default())));
    reg.insert("Ceil", |_| Ok(Box::new(tfdops::math::Ceil::default())));
    reg.insert("Floor", |_| Ok(Box::new(tfdops::math::Floor::default())));
    reg.insert("Clip", clip);

    reg.insert("Cos", |_| Ok(Box::new(tfdops::math::Cos::default())));
    reg.insert("Sin", |_| Ok(Box::new(tfdops::math::Sin::default())));
    reg.insert("Tan", |_| Ok(Box::new(tfdops::math::Tan::default())));
    reg.insert("Acos", |_| Ok(Box::new(tfdops::math::Acos::default())));
    reg.insert("Asin", |_| Ok(Box::new(tfdops::math::Asin::default())));
    reg.insert("Atan", |_| Ok(Box::new(tfdops::math::Atan::default())));

    reg.insert("Exp", |_| Ok(Box::new(tfdops::math::Exp::default())));
    reg.insert("Log", |_| Ok(Box::new(tfdops::math::Ln::default())));
    reg.insert("Sqrt", |_| Ok(Box::new(tfdops::math::Sqrt::default())));
    reg.insert("Rsqrt", |_| Ok(Box::new(tfdops::math::Rsqrt::default())));

    reg.insert("Neg", |_| Ok(Box::new(tfdops::math::Neg::default())));
    reg.insert("Reciprocal", |_| Ok(Box::new(tfdops::math::Recip::default())));

    reg.insert("Pow", |_| Ok(Box::new(tfdops::math::Pow::default())));

    reg.insert("Tanh", |_| Ok(Box::new(tfdops::math::Tanh::default())));

    reg.insert("Gemm", gemm);
}

pub fn clip(node: &NodeProto) -> TfdResult<Box<Op>> {
    let min = node.get_attr_opt_float("min")?.unwrap_or(::std::f32::MIN);
    let max = node.get_attr_opt_float("max")?.unwrap_or(::std::f32::MAX);
    Ok(Box::new(tfdops::math::Clip::new(min, max)))
}

pub fn gemm(node: &NodeProto) -> TfdResult<Box<Op>> {
    let alpha = node.get_attr_opt_float("alpha")?.unwrap_or(1.0);
    let beta = node.get_attr_opt_float("beta")?.unwrap_or(1.0);
    let trans_a = node.get_attr_opt_int("transA")?.map(|a| a != 0).unwrap_or(false);
    let trans_b = node.get_attr_opt_int("transB")?.map(|a| a != 0).unwrap_or(false);
    Ok(Box::new(tfdops::math::Gemm::new(alpha, beta, trans_a, trans_b)))
}
