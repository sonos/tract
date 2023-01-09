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

    reg.insert("BitShift", bitshift);
    reg.insert("BitwiseAnd", |_, _| Ok((ops::logic::BitAnd.into_hir(), vec![])));
    reg.insert("BitwiseOr", |_, _| Ok((ops::logic::BitOr.into_hir(), vec![])));
    reg.insert("BitwiseXor", |_, _| Ok((ops::logic::BitXor.into_hir(), vec![])));
    reg.insert("BitwiseNot", |_, _| Ok((ops::logic::bitnot().into_hir(), vec![])));

    reg.insert("Sum", |_, _| Ok((Box::new(Nary(Box::new(ops::math::Add), false)), vec![])));
    reg.insert("Max", |_, _| Ok((Box::new(Nary(Box::new(ops::math::Max), false)), vec![])));
    reg.insert("Min", |_, _| Ok((Box::new(Nary(Box::new(ops::math::Min), false)), vec![])));
    reg.insert("Mean", |_, _| Ok((Box::new(Nary(Box::new(ops::math::Add), true)), vec![])));

    reg.insert("Abs", |_, _| Ok((ops::math::abs().into_hir(), vec![])));
    reg.insert("Ceil", |_, _| Ok((ops::math::ceil().into_hir(), vec![])));
    reg.insert("Floor", |_, _| Ok((ops::math::floor().into_hir(), vec![])));
    reg.insert("Round", |_, _| Ok((ops::math::round_half_to_even().into_hir(), vec![])));
    reg.insert("Clip", clip::clip);

    reg.insert("Cos", |_, _| Ok((ops::math::cos().into_hir(), vec![])));
    reg.insert("Sin", |_, _| Ok((ops::math::sin().into_hir(), vec![])));
    reg.insert("Tan", |_, _| Ok((ops::math::tan().into_hir(), vec![])));
    reg.insert("Acos", |_, _| Ok((ops::math::acos().into_hir(), vec![])));
    reg.insert("Asin", |_, _| Ok((ops::math::asin().into_hir(), vec![])));
    reg.insert("Atan", |_, _| Ok((ops::math::atan().into_hir(), vec![])));

    reg.insert("Cosh", |_, _| Ok((ops::math::cosh().into_hir(), vec![])));
    reg.insert("Sinh", |_, _| Ok((ops::math::sinh().into_hir(), vec![])));
    reg.insert("Tanh", |_, _| Ok((ops::math::tanh().into_hir(), vec![])));
    reg.insert("Acosh", |_, _| Ok((ops::math::acosh().into_hir(), vec![])));
    reg.insert("Asinh", |_, _| Ok((ops::math::asinh().into_hir(), vec![])));
    reg.insert("Atanh", |_, _| Ok((ops::math::atanh().into_hir(), vec![])));

    reg.insert("Erf", |_, _| Ok((ops::math::erf().into_hir(), vec![])));
    reg.insert("Exp", |_, _| Ok((ops::math::exp().into_hir(), vec![])));
    reg.insert("Log", |_, _| Ok((ops::math::ln().into_hir(), vec![])));
    reg.insert("Sqrt", |_, _| Ok((ops::math::sqrt().into_hir(), vec![])));
    reg.insert("Rsqrt", |_, _| Ok((ops::math::rsqrt().into_hir(), vec![])));

    reg.insert("IsNaN", |_, _| Ok((tract_onnx_opl::is_nan::is_nan().into_hir(), vec![])));
    reg.insert("IsInf", isinf);
    reg.insert("Neg", |_, _| Ok((ops::math::neg().into_hir(), vec![])));
    reg.insert("Sign", |_, _| Ok((ops::math::sign().into_hir(), vec![])));
    reg.insert("Reciprocal", |_, _| Ok((ops::math::recip().into_hir(), vec![])));

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
    Ok((tract_onnx_opl::is_inf::is_inf(detect_positive, detect_negative).into_hir(), vec![]))
}

fn bitshift(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let op: Box<dyn InferenceOp> = if node.get_attr_opt("direction")?.unwrap_or("LEFT") == "RIGHT" {
        ops::math::ShiftRight.into_hir()
    } else {
        ops::math::ShiftLeft.into_hir()
    };
    Ok((op, vec![]))
}
