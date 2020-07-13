use crate::model::{OnnxOpRegister, ParsingContext};
use crate::pb::*;
use tract_hir::internal::*;
use tract_hir::ops;
use tract_hir::ops::binary::Nary;

mod mat_mul_integer;

pub fn register_all_ops(reg: &mut OnnxOpRegister) {
    reg.insert("Add", |_, _| Ok((ops::math::Add.into_hir(), vec![])));
    reg.insert("Sub", |_, _| Ok((ops::math::Sub.into_hir(), vec![])));
    reg.insert("Mul", |_, _| Ok((ops::math::Mul.into_hir(), vec![])));
    reg.insert("Div", |_, _| Ok((ops::math::Div.into_hir(), vec![])));

    reg.insert("Sum", |_, _| Ok((Box::new(Nary(Box::new(ops::math::Add), false)), vec![])));
    reg.insert("Max", |_, _| Ok((Box::new(Nary(Box::new(ops::math::Max), false)), vec![])));
    reg.insert("Min", |_, _| Ok((Box::new(Nary(Box::new(ops::math::Min), false)), vec![])));
    reg.insert("Mean", |_, _| Ok((Box::new(Nary(Box::new(ops::math::Add), true)), vec![])));

    reg.insert("Abs", |_, _| Ok((Box::new(ops::math::abs()), vec![])));
    reg.insert("Ceil", |_, _| Ok((Box::new(ops::math::ceil()), vec![])));
    reg.insert("Floor", |_, _| Ok((Box::new(ops::math::floor()), vec![])));
    reg.insert("Clip", clip);

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

    reg.insert("Erf", |_, _| Ok((Box::new(erf()), vec![])));
    reg.insert("Exp", |_, _| Ok((Box::new(ops::math::exp()), vec![])));
    reg.insert("Log", |_, _| Ok((Box::new(ops::math::ln()), vec![])));
    reg.insert("Sqrt", |_, _| Ok((Box::new(ops::math::sqrt()), vec![])));
    reg.insert("Rsqrt", |_, _| Ok((Box::new(ops::math::rsqrt()), vec![])));

    reg.insert("IsNaN", |_, _| Ok((Box::new(is_nan()), vec![])));
    reg.insert("Neg", |_, _| Ok((Box::new(ops::math::neg()), vec![])));
    reg.insert("Sign", |_, _| Ok((Box::new(ops::math::sign()), vec![])));
    reg.insert("Reciprocal", |_, _| Ok((Box::new(ops::math::recip()), vec![])));

    reg.insert("Pow", |_, _| Ok((ops::math::Pow.into_hir(), vec![])));

    reg.insert("MatMul", |_, _| Ok((Box::new(ops::matmul::MatMul::default()), vec![])));
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
    Ok((expand(ops::activations::Clip::new(min, max)), vec!()))
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
    Ok((expand(Gemm::new(alpha, beta, trans_a, trans_b)), vec![]))
}

#[derive(Debug, Clone, new, Educe)]
#[educe(Hash)]
pub struct Gemm {
    #[educe(Hash(method = "hash_f32"))]
    alpha: f32,
    #[educe(Hash(method = "hash_f32"))]
    beta: f32,
    trans_a: bool,
    trans_b: bool,
}

tract_linalg::impl_dyn_hash!(Gemm);

impl Expansion for Gemm {
    fn name(&self) -> Cow<str> {
        "Gemm".into()
    }

    op_onnx!();

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

    fn wire(
        &self,
        name: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let (a, b, mut c) = (inputs[0], inputs[1], inputs[2]);
        let mut wire = model.wire_node(
            format!("{}.ab", name),
            ops::matmul::MatMul::default().with_a_trans(self.trans_a).with_b_trans(self.trans_b),
            &[a, b].as_ref(),
        )?[0];
        if self.alpha != 1.0 {
            let alpha = tensor0(self.alpha).broadcast_into_rank(model.outlet_fact(wire)?.rank())?;
            wire = model.wire_node(
                format!("{}.alpha_ab", self.alpha),
                ops::math::mul::unary(alpha.into_arc_tensor()),
                &[wire],
            )?[0];
        }
        if self.beta != 0.0f32 {
            while model.outlet_fact(wire)?.rank() > model.outlet_fact(c)?.rank() {
                c = model.wire_node(
                    format!("{}.c_broadcast_to_{}", self.beta, model.outlet_fact(c)?.rank()),
                    tract_hir::tract_core::ops::change_axes::AxisOp::Add(0),
                    &[c],
                )?[0];
            }
            let beta = tensor0(self.beta).broadcast_into_rank(model.outlet_fact(wire)?.rank())?;
            let beta_c = model.wire_node(
                format!("{}.beta_c", self.beta),
                ops::math::mul::unary(beta.into_arc_tensor()),
                &[c],
            )?[0];
            wire = model.wire_node(name, ops::math::add::bin_typed(), &[wire, beta_c])?[0];
        }
        Ok(tvec!(wire))
    }
}
