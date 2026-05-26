use crate::model::ParsingContext;
use crate::pb::NodeProto;
use tract_hir::internal::*;
use tract_hir::ops::logic::wire_with_rank_broadcast;

pub fn lp_normalization(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let axis = node.get_attr_opt("axis")?.unwrap_or(-1);
    let p: i64 = node.get_attr_opt("p")?.unwrap_or(2);
    ensure!(p == 1 || p == 2, "LpNormalization only supports p=1 or p=2, got p={p}");
    Ok((expand(LpNorm { axis, p }), vec![]))
}

#[derive(Debug, Clone, new)]
struct LpNorm {
    axis: i64,
    p: i64,
}

impl Expansion for LpNorm {
    fn name(&self) -> StaticName {
        "LpNorm".into()
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(inputs, 1)?;
        check_output_arity(outputs, 1)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].shape, &outputs[0].shape)?;
        Ok(())
    }

    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let rank = model.outlet_fact(inputs[0])?.rank() as i64;
        let axis = if self.axis < 0 { self.axis + rank } else { self.axis };
        // Lp norm along `axis`, keeping the reduced dim so it broadcasts against the input.
        let reducer = if self.p == 1 {
            tract_hir::ops::nn::Reducer::L1
        } else {
            tract_hir::ops::nn::Reducer::L2
        };
        let norm = tract_hir::ops::nn::Reduce::new(Some(vec![axis]), true, reducer).wire(
            &format!("{prefix}.norm"),
            model,
            &inputs[0..1],
        )?;
        wire_with_rank_broadcast(prefix, model, tract_hir::ops::math::div(), &[inputs[0], norm[0]])
    }
}
