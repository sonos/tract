use tract_core::ops as tractops;

use crate::model::{ OnnxOpRegister, ParsingContext };
use crate::pb::*;
use tract_core::internal::*;

pub fn register_all_ops(reg: &mut OnnxOpRegister) {
    reg.insert("Add", |_, _| Ok(Box::new(tractops::math::Add::default())));
    reg.insert("Sub", |_, _| Ok(Box::new(tractops::math::Sub::default())));
    reg.insert("Mul", |_, _| Ok(Box::new(tractops::math::Mul::default())));
    reg.insert("Div", |_, _| Ok(Box::new(tractops::math::Div::default())));

    reg.insert("Sum", |_, _| Ok(Box::new(tractops::math::AddN::default())));
    reg.insert("Max", |_, _| Ok(Box::new(tractops::math::MaxN::default())));
    reg.insert("Min", |_, _| Ok(Box::new(tractops::math::MinN::default())));
    reg.insert("Mean", |_, _| Ok(Box::new(tractops::math::MeanN::default())));

    reg.insert("Abs", |_, _| Ok(Box::new(tractops::math::Abs::default())));
    reg.insert("Ceil", |_, _| Ok(Box::new(tractops::math::Ceil::default())));
    reg.insert("Floor", |_, _| Ok(Box::new(tractops::math::Floor::default())));
    reg.insert("Clip", clip);

    reg.insert("Cos", |_, _| Ok(Box::new(tractops::math::Cos::default())));
    reg.insert("Sin", |_, _| Ok(Box::new(tractops::math::Sin::default())));
    reg.insert("Tan", |_, _| Ok(Box::new(tractops::math::Tan::default())));
    reg.insert("Acos", |_, _| Ok(Box::new(tractops::math::Acos::default())));
    reg.insert("Asin", |_, _| Ok(Box::new(tractops::math::Asin::default())));
    reg.insert("Atan", |_, _| Ok(Box::new(tractops::math::Atan::default())));

    reg.insert("Cosh", |_, _| Ok(Box::new(tractops::math::Cosh::default())));
    reg.insert("Sinh", |_, _| Ok(Box::new(tractops::math::Sinh::default())));
    reg.insert("Tanh", |_, _| Ok(Box::new(tractops::nn::Tanh::default())));
    reg.insert("Acosh", |_, _| Ok(Box::new(tractops::math::Acosh::default())));
    reg.insert("Asinh", |_, _| Ok(Box::new(tractops::math::Asinh::default())));
    reg.insert("Atanh", |_, _| Ok(Box::new(tractops::math::Atanh::default())));

    reg.insert("Exp", |_, _| Ok(Box::new(tractops::math::Exp::default())));
    reg.insert("Log", |_, _| Ok(Box::new(tractops::math::Ln::default())));
    reg.insert("Sqrt", |_, _| Ok(Box::new(tractops::math::Sqrt::default())));
    reg.insert("Rsqrt", |_, _| Ok(Box::new(tractops::math::Rsqrt::default())));

    reg.insert("IsNaN", |_, _| Ok(Box::new(tractops::math::IsNan::default())));
    reg.insert("Neg", |_, _| Ok(Box::new(tractops::math::Neg::default())));
    reg.insert("Sign", |_, _| Ok(Box::new(tractops::math::Sign::default())));
    reg.insert("Reciprocal", |_, _| Ok(Box::new(tractops::math::Recip::default())));

    reg.insert("Pow", |_, _| Ok(Box::new(tractops::math::Pow::default())));

    reg.insert("MatMul", |_, _| Ok(Box::new(tractops::math::MatMul::default())));
    reg.insert("Gemm", gemm);
}

pub fn clip(_ctx: &ParsingContext, node: &NodeProto) -> TractResult<Box<InferenceOp>> {
    let min = node.get_attr_opt("min")?.unwrap_or(::std::f32::MIN);
    let max = node.get_attr_opt("max")?.unwrap_or(::std::f32::MAX);
    Ok(Box::new(tractops::math::Clip::new(min, max)))
}

pub fn gemm(_ctx: &ParsingContext, node: &NodeProto) -> TractResult<Box<InferenceOp>> {
    let alpha = node.get_attr_opt("alpha")?.unwrap_or(1.);
    let beta = node.get_attr_opt("beta")?.unwrap_or(1.);
    let trans_a = node.get_attr_opt("transA")?.unwrap_or(false);
    let trans_b = node.get_attr_opt("transB")?.unwrap_or(false);
    Ok(Box::new(tractops::math::Gemm::new(alpha, beta, trans_a, trans_b, true)))
}
