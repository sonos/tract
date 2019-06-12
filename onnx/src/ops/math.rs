use tract_core::ops as tractops;

use crate::model::{OnnxOpRegister, ParsingContext};
use crate::pb::*;
use tract_core::internal::*;

pub fn register_all_ops(reg: &mut OnnxOpRegister) {
    reg.insert("Add", |_, _| Ok((Box::new(tractops::math::Add::default()),vec!())));
    reg.insert("Sub", |_, _| Ok((Box::new(tractops::math::Sub::default()),vec!())));
    reg.insert("Mul", |_, _| Ok((Box::new(tractops::math::Mul::default()),vec!())));
    reg.insert("Div", |_, _| Ok((Box::new(tractops::math::Div::default()),vec!())));

    reg.insert("Sum", |_, _| Ok((Box::new(tractops::math::AddN::default()),vec!())));
    reg.insert("Max", |_, _| Ok((Box::new(tractops::math::MaxN::default()),vec!())));
    reg.insert("Min", |_, _| Ok((Box::new(tractops::math::MinN::default()),vec!())));
    reg.insert("Mean", |_, _| Ok((Box::new(tractops::math::MeanN::default()),vec!())));

    reg.insert("Abs", |_, _| Ok((Box::new(tractops::math::Abs::default()),vec!())));
    reg.insert("Ceil", |_, _| Ok((Box::new(tractops::math::Ceil::default()),vec!())));
    reg.insert("Floor", |_, _| Ok((Box::new(tractops::math::Floor::default()),vec!())));
    reg.insert("Clip", clip);

    reg.insert("Cos", |_, _| Ok((Box::new(tractops::math::Cos::default()),vec!())));
    reg.insert("Sin", |_, _| Ok((Box::new(tractops::math::Sin::default()),vec!())));
    reg.insert("Tan", |_, _| Ok((Box::new(tractops::math::Tan::default()),vec!())));
    reg.insert("Acos", |_, _| Ok((Box::new(tractops::math::Acos::default()),vec!())));
    reg.insert("Asin", |_, _| Ok((Box::new(tractops::math::Asin::default()),vec!())));
    reg.insert("Atan", |_, _| Ok((Box::new(tractops::math::Atan::default()),vec!())));

    reg.insert("Cosh", |_, _| Ok((Box::new(tractops::math::Cosh::default()),vec!())));
    reg.insert("Sinh", |_, _| Ok((Box::new(tractops::math::Sinh::default()),vec!())));
    reg.insert("Tanh", |_, _| Ok((Box::new(tractops::nn::Tanh::default()),vec!())));
    reg.insert("Acosh", |_, _| Ok((Box::new(tractops::math::Acosh::default()),vec!())));
    reg.insert("Asinh", |_, _| Ok((Box::new(tractops::math::Asinh::default()),vec!())));
    reg.insert("Atanh", |_, _| Ok((Box::new(tractops::math::Atanh::default()),vec!())));

    reg.insert("Erf", |_, _| Ok((Box::new(Erf::default()), vec!())));
    reg.insert("Exp", |_, _| Ok((Box::new(tractops::math::Exp::default()),vec!())));
    reg.insert("Log", |_, _| Ok((Box::new(tractops::math::Ln::default()),vec!())));
    reg.insert("Sqrt", |_, _| Ok((Box::new(tractops::math::Sqrt::default()),vec!())));
    reg.insert("Rsqrt", |_, _| Ok((Box::new(tractops::math::Rsqrt::default()),vec!())));

    reg.insert("IsNaN", |_, _| Ok((Box::new(tractops::math::IsNan::default()),vec!())));
    reg.insert("Neg", |_, _| Ok((Box::new(tractops::math::Neg::default()),vec!())));
    reg.insert("Sign", |_, _| Ok((Box::new(tractops::math::Sign::default()),vec!())));
    reg.insert("Reciprocal", |_, _| Ok((Box::new(tractops::math::Recip::default()),vec!())));

    reg.insert("Pow", |_, _| Ok((Box::new(tractops::math::Pow::default()),vec!())));

    reg.insert("MatMul", |_, _| Ok((Box::new(tractops::math::MatMul::default()),vec!())));
    reg.insert("Gemm", gemm);
}

pub fn clip(_ctx: &ParsingContext, node: &NodeProto) -> TractResult<(Box<InferenceOp>, Vec<String>)> {
    let min = node.get_attr_opt("min")?.unwrap_or(::std::f32::MIN);
    let max = node.get_attr_opt("max")?.unwrap_or(::std::f32::MAX);
    Ok((Box::new(tractops::math::Clip::new(min, max)),vec!()))
}

pub fn gemm(_ctx: &ParsingContext, node: &NodeProto) -> TractResult<(Box<InferenceOp>, Vec<String>)> {
    let alpha = node.get_attr_opt("alpha")?.unwrap_or(1.);
    let beta = node.get_attr_opt("beta")?.unwrap_or(1.);
    let trans_a = node.get_attr_opt("transA")?.unwrap_or(false);
    let trans_b = node.get_attr_opt("transB")?.unwrap_or(false);
    Ok((Box::new(tractops::math::Gemm::new(alpha, beta, trans_a, trans_b, true)),vec!()))
}

element_map!(Erf, [f32], erf_f32);

#[allow(non_upper_case_globals)]
fn erf_f32(x: f32) -> f32 {
    const a1: f32 = 0.0705230784;
    const a2: f32 = 0.0422820123;
    const a3: f32 = 0.0092705272;
    const a4: f32 = 0.0001520143;
    const a5: f32 = 0.0002765672;
    const a6: f32 = 0.0000430638;

    let signum = x.signum();
    let x = x.abs();
    let y = a6 * x;
    let y = (a5 + y) * x;
    let y = (a4 + y) * x;
    let y = (a3 + y) * x;
    let y = (a2 + y) * x;
    let y = (a1 + y) * x;
    let y = 1.0 - (y + 1.0).powi(16).recip();

    y.copysign(signum)
}
