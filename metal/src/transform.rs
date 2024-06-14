use crate::ops;
use anyhow::Result;
use std::borrow::Cow;
use std::fmt::Debug;
use tract_core::internal::*;
use tract_core::ops::binary::TypedBinOp;
use tract_core::ops::einsum::{rewrite_einsums_as_matmul, BasicMatMul};
use tract_core::ops::element_wise::ElementWiseOp;

use tract_core::transform::ModelTransform;

#[derive(Debug, Default)]
pub struct MetalTransform;

impl ModelTransform for MetalTransform {
    fn name(&self) -> Cow<str> {
        "metal-transform".into()
    }

    fn transform(&self, model: &mut TypedModel) -> TractResult<()> {
        rewrite_einsums_as_matmul(model)?;
        Rewriter::default()
            .with_rule_for("matmul-to-metal-gemm", matmul_to_gemm)
            .with_rule_for("binops-to-metal", bin_ops_to_metal)
            .with_rule_for("element-wise-to-meta", element_wise_ops_to_metal)
            .rewrite(&(), model)?;
        Ok(())
    }
}

fn matmul_to_gemm(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    _node_name: &str,
    op: &BasicMatMul,
) -> Result<Option<TypedModelPatch>> {
    if !op.transpose_a
        && !op.transpose_b
        && !op.transpose_c
        && op.quantize_output.is_none()
        && (model.node_input_facts(node.id)?.iter().all(|f| f.datum_type == f32::datum_type())
            || model.node_input_facts(node.id)?.iter().all(|f| f.datum_type == f16::datum_type()))
    {
        TypedModelPatch::replace_single_op(model, node, &node.inputs, ops::MetalGemm::default())
            .map(Some)
    } else {
        Ok(None)
    }
}

macro_rules! map_bin_ops {
    ([$(($tract_bin_op:path, $metal_bin_op:ident)),* $(,)?]) => {
        |op: &tract_core::ops::binary::TypedBinOp| {
            $(if let Some(_op) = op.0.downcast_ref::<$tract_bin_op>() {
                return Some($crate::ops::binary::MetalBinOp($crate::ops::binary::BinOps::$metal_bin_op));
            })*
            return None;
        }
    };
}

macro_rules! map_element_wise_ops {
    ([$(($tract_bin_op:path, $metal_bin_op:ident)),* $(,)?]) => {
        |op: &tract_core::ops::element_wise::ElementWiseOp| {
            $(if let Some(_op) = op.0.downcast_ref::<$tract_bin_op>() {
                return Some($crate::ops::element_wise::MetalElementWiseOp($crate::ops::element_wise::ElementWiseOps::Rsqrt));
            })*
            return None;
        }
    };
}

fn bin_ops_to_metal(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    _node_name: &str,
    op: &TypedBinOp,
) -> Result<Option<TypedModelPatch>> {
    if op.1.is_some() {
        return Ok(None);
    }

    map_bin_ops!([
        (tract_core::ops::math::Mul, Mul),
        (tract_core::ops::math::Add, Add),
        (tract_core::ops::math::Div, Div),
        (tract_core::ops::math::Sub, Sub),
        (tract_core::ops::math::Pow, Pow),
        (tract_core::ops::logic::Less, Less),
        (tract_core::ops::logic::LessEqual, LessEqual),
        (tract_core::ops::logic::Greater, Greater),
        (tract_core::ops::logic::GreaterEqual, GreaterEqual),
        (tract_core::ops::logic::Equals, Equals),
        (tract_core::ops::logic::NotEquals, NotEquals),
        (tract_core::ops::logic::And, And),
        (tract_core::ops::logic::Or, Or),
    ])(op)
    .map(|metal_op| TypedModelPatch::replace_single_op(model, node, &node.inputs, metal_op))
    .transpose()
}

fn element_wise_ops_to_metal(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    _node_name: &str,
    op: &ElementWiseOp,
) -> Result<Option<TypedModelPatch>> {
    if op.1.is_some() {
        return Ok(None);
    }

    map_element_wise_ops!([
        (tract_core::ops::math::Abs, Abs),
        (tract_core::ops::math::Exp, Exp),
        (tract_core::ops::math::Ln, Ln),
        (tract_core::ops::nn::Sigmoid, Sigmoid),
        (tract_core::ops::math::Square, Square),
        (tract_core::ops::math::Sqrt, Sqrt),
        (tract_core::ops::math::Rsqrt, Rsqrt),
        (tract_core::ops::math::Recip, Recip),
        (tract_core::ops::math::Ceil, Ceil),
        (tract_core::ops::math::Floor, Floor),
        (tract_core::ops::math::Round, Round),
        (tract_core::ops::math::RoundHalfToEven, RoundHalfToEven),
        (tract_core::ops::math::Cos, Cos),
        (tract_core::ops::math::Acos, Acos),
        (tract_core::ops::math::Acosh, Acosh),
        (tract_core::ops::math::Cosh, Cosh),
        (tract_core::ops::math::Sin, Sin),
        (tract_core::ops::math::Asin, Asin),
        (tract_core::ops::math::Asinh, Asinh),
        (tract_core::ops::math::Sinh, Sinh),
        (tract_core::ops::math::Tan, Tan),
        (tract_core::ops::math::Atan, Atan),
        (tract_core::ops::math::Atanh, Atanh),
        (tract_core::ops::math::Tanh, Tanh),
        (tract_core::ops::math::Erf, Erf),
        (tract_core::ops::math::Neg, Neg),
    ])(op)
    .map(|metal_op| TypedModelPatch::replace_single_op(model, node, &node.inputs, metal_op))
    .transpose()
}
