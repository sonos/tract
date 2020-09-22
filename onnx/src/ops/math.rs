use crate::model::OnnxOpRegister;
use crate::model::ParsingContext;
use crate::pb::*;
use tract_hir::internal::*;
use tract_hir::ops;
use tract_hir::ops::binary::Nary;

mod clip;
mod gemm;
mod mat_mul_integer;
mod pow;
mod rem;

pub fn register_all_ops(reg: &mut OnnxOpRegister) {
    reg.insert("Add", |_, _| Ok((ops::math::Add.into_hir(), vec![])));
    reg.insert("Sub", |_, _| Ok((ops::math::Sub.into_hir(), vec![])));
    reg.insert("Mul", |_, _| Ok((ops::math::Mul.into_hir(), vec![])));
    reg.insert("Div", |_, _| Ok((ops::math::Div.into_hir(), vec![])));
    reg.insert("Mod", rem::rem);

    reg.insert("Sum", |_, _| Ok((Box::new(Nary(Box::new(ops::math::Add), false)), vec![])));
    reg.insert("Max", |_, _| Ok((Box::new(Nary(Box::new(ops::math::Max), false)), vec![])));
    reg.insert("Min", |_, _| Ok((Box::new(Nary(Box::new(ops::math::Min), false)), vec![])));
    reg.insert("Mean", |_, _| Ok((Box::new(Nary(Box::new(ops::math::Add), true)), vec![])));

    reg.insert("Abs", |_, _| Ok((Box::new(ops::math::abs()), vec![])));
    reg.insert("Ceil", |_, _| Ok((Box::new(ops::math::ceil()), vec![])));
    reg.insert("Floor", |_, _| Ok((Box::new(ops::math::floor()), vec![])));
    reg.insert("Round", |_, _| Ok((Box::new(ops::math::round_half_to_even()), vec![])));
    reg.insert("Clip", clip::clip);

    reg.insert("Cos", |_, _| Ok((Box::new(ops::math::cos()), vec![])));
    reg.insert("Sin", |_, _| Ok((Box::new(ops::math::sin()), vec![])));
    reg.insert("Tan", |_, _| Ok((Box::new(ops::math::tan()), vec![])));
    reg.insert("Acos", |_, _| Ok((Box::new(ops::math::acos()), vec![])));
    reg.insert("Asin", |_, _| Ok((Box::new(ops::math::asin()), vec![])));
    reg.insert("Atan", |_, _| Ok((Box::new(ops::math::atan()), vec![])));

    reg.insert("Cosh", |_, _| Ok((Box::new(ops::math::cosh()), vec![])));
    reg.insert("Sinh", |_, _| Ok((Box::new(ops::math::sinh()), vec![])));
    reg.insert("Tanh", |_, _| Ok((Box::new(ops::math::tanh()), vec![])));
    reg.insert("Acosh", |_, _| Ok((Box::new(ops::math::acosh()), vec![])));
    reg.insert("Asinh", |_, _| Ok((Box::new(ops::math::asinh()), vec![])));
    reg.insert("Atanh", |_, _| Ok((Box::new(ops::math::atanh()), vec![])));

    reg.insert("Erf", |_, _| Ok((Box::new(tract_onnx_opl::erf::erf()), vec![])));
    reg.insert("Exp", |_, _| Ok((Box::new(ops::math::exp()), vec![])));
    reg.insert("Log", |_, _| Ok((Box::new(ops::math::ln()), vec![])));
    reg.insert("Sqrt", |_, _| Ok((Box::new(ops::math::sqrt()), vec![])));
    reg.insert("Rsqrt", |_, _| Ok((Box::new(ops::math::rsqrt()), vec![])));

    reg.insert("IsNaN", |_, _| Ok((Box::new(tract_onnx_opl::is_nan::is_nan()), vec![])));
    reg.insert("IsInf", isinf);
    reg.insert("Neg", |_, _| Ok((Box::new(ops::math::neg()), vec![])));
    reg.insert("Sign", |_, _| Ok((Box::new(ops::math::sign()), vec![])));
    reg.insert("Reciprocal", |_, _| Ok((Box::new(ops::math::recip()), vec![])));

    reg.insert("Pow", pow::pow);

    reg.insert("MatMul", |_, _| Ok((expand(ops::matmul::MatMulInference::default()), vec![])));
    reg.insert("MatMulInteger", mat_mul_integer::mat_mul_integer);
    reg.insert("QLinearMatMul", mat_mul_integer::q_linear_mat_mul);
    reg.insert("Gemm", gemm::gemm);
}

fn isinf(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let detect_positive = node.get_attr_opt("detect_positive")?.unwrap_or(1) != 0;
    let detect_negative = node.get_attr_opt("detect_negative")?.unwrap_or(1) != 0;
    Ok((Box::new(tract_onnx_opl::is_inf::is_inf(detect_positive, detect_negative)), vec![]))
}


