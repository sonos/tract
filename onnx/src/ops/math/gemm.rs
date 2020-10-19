use crate::model::ParsingContext;
use crate::pb::*;
use tract_hir::internal::*;
use tract_hir::ops;

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

tract_data::impl_dyn_hash!(Gemm);

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
                    format!("{}.c_add_axis_{}", name, model.outlet_fact(c)?.rank()),
                    tract_hir::tract_core::ops::change_axes::AxisOp::Add(0),
                    &[c],
                )?[0];
            }
            let beta = tensor0(self.beta).broadcast_into_rank(model.outlet_fact(wire)?.rank())?;
            let beta_c = model.wire_node(
                format!("{}.beta_c", name),
                ops::math::mul::unary(beta.into_arc_tensor()),
                &[c],
            )?[0];
            wire = model.wire_node(name, ops::math::add::bin_typed(), &[wire, beta_c])?[0];
        }
        Ok(tvec!(wire))
    }
}
