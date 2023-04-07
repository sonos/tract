use crate::model::ParsingContext;
use crate::pb::*;
use tract_hir::internal::*;
use tract_hir::ops;
use tract_hir::tract_core::ops::binary::wire_bin;
use tract_hir::tract_core::ops::einsum::EinSum;

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

#[derive(Debug, Clone, new)]
pub struct Gemm {
    alpha: f32,
    beta: f32,
    trans_a: bool,
    trans_b: bool,
}

impl Expansion for Gemm {
    fn name(&self) -> Cow<str> {
        "Gemm".into()
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(inputs, 3)?;
        s.equals(&inputs[2].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].rank, 2)?;
        s.equals(&inputs[1].rank, 2)?;
        check_output_arity(outputs, 1)?;
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
        let axes = AxesMapping::for_numpy_matmul(2, self.trans_a, self.trans_b, false)?;
        let mut wire = model.wire_node(
            format!("{name}.ab"),
            EinSum::new(axes, model.outlet_fact(a)?.datum_type),
            [a, b].as_ref(),
        )?[0];
        if self.alpha != 1.0 {
            let alpha = tensor0(self.alpha).broadcast_into_rank(model.outlet_fact(wire)?.rank())?;
            let alpha = model.add_const(name.to_string() + ".alpha_ab.cst", alpha)?;
            wire =
                wire_bin(name.to_string() + ".alpha_ab", model, ops::math::Mul, &[alpha, wire])?[0];
        }
        if self.beta != 0.0f32 {
            while model.outlet_fact(wire)?.rank() > model.outlet_fact(c)?.rank() {
                c = model.wire_node(
                    format!("{}.c_add_axis_{}", name, model.outlet_fact(c)?.rank()),
                    tract_hir::tract_core::ops::change_axes::AxisOp::Add(0),
                    &[c],
                )?[0];
            }
            let beta = tensor0(self.beta).broadcast_into_rank(model.outlet_fact(wire)?.rank())?;
            let beta = model.add_const(name.to_string() + ".beta_c.cst", beta)?;
            let beta_c =
                wire_bin(name.to_string() + ".beta_c", model, ops::math::Mul, &[beta, c])?[0];
            wire = wire_bin(name, model, ops::math::Add, &[wire, beta_c])?[0];
        }
        Ok(tvec!(wire))
    }
}
