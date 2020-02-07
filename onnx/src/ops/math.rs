use tract_core::ops as tractops;

use crate::model::{OnnxOpRegister, ParsingContext};
use crate::pb::*;
use tract_core::infer::*;
use tract_core::internal::*;
use tract_core::ops::binary::Nary;

mod mat_mul_integer;

pub fn register_all_ops(reg: &mut OnnxOpRegister) {
    reg.insert("Add", |_, _| Ok((Box::new(tractops::math::add::bin()), vec![])));
    reg.insert("Sub", |_, _| Ok((Box::new(tractops::math::sub::bin()), vec![])));
    reg.insert("Mul", |_, _| Ok((Box::new(tractops::math::mul::bin()), vec![])));
    reg.insert("Div", |_, _| Ok((Box::new(tractops::math::div::bin()), vec![])));

    reg.insert("Sum", |_, _| Ok((Box::new(Nary(Box::new(tractops::math::Add), false)), vec![])));
    reg.insert("Max", |_, _| Ok((Box::new(Nary(Box::new(tractops::math::Max), false)), vec![])));
    reg.insert("Min", |_, _| Ok((Box::new(Nary(Box::new(tractops::math::Min), false)), vec![])));
    reg.insert("Mean", |_, _| Ok((Box::new(Nary(Box::new(tractops::math::Add), true)), vec![])));

    reg.insert("Abs", |_, _| Ok((Box::new(tractops::math::abs()), vec![])));
    reg.insert("Ceil", |_, _| Ok((Box::new(tractops::math::ceil()), vec![])));
    reg.insert("Floor", |_, _| Ok((Box::new(tractops::math::floor()), vec![])));
    reg.insert("Clip", clip);

    reg.insert("Cos", |_, _| Ok((Box::new(tractops::math::cos()), vec![])));
    reg.insert("Sin", |_, _| Ok((Box::new(tractops::math::sin()), vec![])));
    reg.insert("Tan", |_, _| Ok((Box::new(tractops::math::tan()), vec![])));
    reg.insert("Acos", |_, _| Ok((Box::new(tractops::math::acos()), vec![])));
    reg.insert("Asin", |_, _| Ok((Box::new(tractops::math::asin()), vec![])));
    reg.insert("Atan", |_, _| Ok((Box::new(tractops::math::atan()), vec![])));

    reg.insert("Cosh", |_, _| Ok((Box::new(tractops::math::cosh()), vec![])));
    reg.insert("Sinh", |_, _| Ok((Box::new(tractops::math::sinh()), vec![])));
    reg.insert("Tanh", |_, _| Ok((Box::new(tractops::math::tanh()), vec![])));
    reg.insert("Acosh", |_, _| Ok((Box::new(tractops::math::acosh()), vec![])));
    reg.insert("Asinh", |_, _| Ok((Box::new(tractops::math::asinh()), vec![])));
    reg.insert("Atanh", |_, _| Ok((Box::new(tractops::math::atanh()), vec![])));

    reg.insert("Erf", |_, _| Ok((Box::new(erf()), vec![])));
    reg.insert("Exp", |_, _| Ok((Box::new(tractops::math::exp()), vec![])));
    reg.insert("Log", |_, _| Ok((Box::new(tractops::math::ln()), vec![])));
    reg.insert("Sqrt", |_, _| Ok((Box::new(tractops::math::sqrt()), vec![])));
    reg.insert("Rsqrt", |_, _| Ok((Box::new(tractops::math::rsqrt()), vec![])));

    reg.insert("IsNaN", |_, _| Ok((Box::new(is_nan()), vec![])));
    reg.insert("Neg", |_, _| Ok((Box::new(tractops::math::neg()), vec![])));
    reg.insert("Sign", |_, _| Ok((Box::new(tractops::math::sign()), vec![])));
    reg.insert("Reciprocal", |_, _| Ok((Box::new(tractops::math::recip()), vec![])));

    reg.insert("Pow", |_, _| Ok((Box::new(tractops::math::pow::bin()), vec![])));

    reg.insert("MatMul", |_, _| Ok((Box::new(tractops::matmul::MatMul::default()), vec![])));
    reg.insert("MatMulInteger", mat_mul_integer::mat_mul_integer);
    reg.insert("QLinearMatMul", mat_mul_integer::q_linear_mat_mul);
    reg.insert("Gemm", gemm);
}

pub fn clip(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let min: Option<f32> = node.get_attr_opt("min")?;
    let max: Option<f32> = node.get_attr_opt("max")?;
    let op: Box<dyn InferenceOp> = match (min, max) {
        (Some(min), Some(max)) => Box::new(tractops::math::scalar_min_max(max.into(), min.into())),
        (None, Some(max)) => Box::new(tractops::math::scalar_min(max.into())),
        (Some(min), None) => Box::new(tractops::math::scalar_max(min.into())),
        (None, None) => Box::new(tractops::identity::Identity::default()),
    };
    Ok((op, vec![]))
}

element_wise!(erf, Erf,
    [f32] => |_, xs| {
        xs.iter_mut().for_each(|x| *x = erf_f32(*x));
        Ok(())
    };
    prefix: "onnx."
);
element_wise_oop!(is_nan, IsNan,
    [f32] => bool |_, xs, ys| {
        xs.iter().zip(ys.iter_mut()).for_each(|(x,y)| *y = x.is_nan());
        Ok(())
    };
    prefix: "onnx."
);

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

pub fn gemm(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let alpha = node.get_attr_opt("alpha")?.unwrap_or(1.);
    let beta = node.get_attr_opt("beta")?.unwrap_or(1.);
    let trans_a = node.get_attr_opt("transA")?.unwrap_or(false);
    let trans_b = node.get_attr_opt("transB")?.unwrap_or(false);
    Ok((Box::new(Gemm::new(alpha, beta, trans_a, trans_b)), vec![]))
}

#[derive(Debug, Clone, new)]
pub struct Gemm {
    alpha: f32,
    beta: f32,
    trans_a: bool,
    trans_b: bool,
}

impl Op for Gemm {
    fn name(&self) -> Cow<str> {
        "Gemm".into()
    }

    not_a_typed_op!();
}

impl StatelessOp for Gemm {
    fn eval(&self, _inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        unreachable!();
    }
}

impl InferenceRulesOp for Gemm {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 3)?;
        s.equals(&inputs[2].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].rank, 2)?;
        s.equals(&inputs[1].rank, 2)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&outputs[0].rank, 2)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[1].datum_type, &outputs[0].datum_type)?;
        let (ca, ra) = if self.trans_a { (0, 1) } else { (1, 0) };
        let (cb, rb) = if self.trans_b { (0, 1) } else { (1, 0) };
        s.equals(&inputs[0].shape[ra], &outputs[0].shape[0])?;
        s.equals(&inputs[0].shape[ca], &inputs[1].shape[rb])?;
        s.equals(&inputs[1].shape[cb], &outputs[0].shape[1])?;
        Ok(())
    }

    fn incorporate(
        &self,
        model: &InferenceModel,
        node: &InferenceNode,
    ) -> TractResult<Option<InferenceModelPatch>> {
        use tract_core::ops;
        let mut patch = InferenceModelPatch::default();
        let a = patch.tap_model(model, node.inputs[0])?;
        let b = patch.tap_model(model, node.inputs[1])?;
        let mut result = patch.wire_node(
            format!("{}-ab", node.name),
            ops::matmul::MatMul::default().with_a_trans(self.trans_a).with_b_trans(self.trans_b),
            &[a, b].as_ref(),
        )?[0];
        if self.alpha != 1.0 {
            let alpha: OutletId =
                patch.add_const(format!("{}-alpha", node.name), rctensor0(self.alpha))?.into();
            result = patch.wire_node(
                format!("{}-alpha_ab", node.name),
                ops::math::mul::bin(),
                &[alpha, result].as_ref(),
            )?[0];
        }
        if self.beta != 0.0f32 {
            let mut beta_c: OutletId = patch.tap_model(model, node.inputs[2])?.into();
            if self.beta != 1.0f32 {
                let beta: OutletId =
                    patch.add_const(format!("{}-beta", node.name), rctensor0(self.beta))?.into();
                beta_c = patch.wire_node(
                    format!("{}-beta_c", node.name),
                    ops::math::mul::bin(),
                    &[beta, beta_c].as_ref(),
                )?[0];
            }
            result = patch.wire_node(
                format!("{}-gemm", node.name),
                ops::math::add::bin(),
                &[beta_c, result].as_ref(),
            )?[0];
        }
        patch.node_mut(result.node).name = node.name.clone();
        patch.shunt_outside(node.id.into(), result)?;
        Ok(Some(patch))
    }
    inference_op_as_op!();
}
