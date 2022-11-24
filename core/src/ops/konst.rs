use crate::internal::*;

#[derive(Debug, Clone, new, Hash)]
pub struct Const(pub Arc<Tensor>);

impl_dyn_hash!(Const);

impl Op for Const {
    fn name(&self) -> Cow<str> {
        "Const".into()
    }

    op_as_typed_op!();
}

impl EvalOp for Const {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, _inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        Ok(tvec![self.0.clone().into_tvalue()])
    }
}

impl TypedOp for Const {
    as_op!();

    fn output_facts(&self, _inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(self.0.as_ref().into()))
    }

    fn change_axes(
        &self,
        _model: &TypedModel,
        _node: &TypedNode,
        io: InOut,
        change: &AxisOp,
    ) -> TractResult<Option<AxisChangeConsequence>> {
        anyhow::ensure!(io == InOut::Out(0));
        let mut new_tensor = self.0.clone().into_tensor();
        if change.change_tensor(&mut new_tensor, false).is_ok() {
            Ok(Some(AxisChangeConsequence {
                substitute_op: Some(Box::new(Const(new_tensor.into_arc_tensor()))),
                wire_changes: tvec!((io, change.clone())),
            }))
        } else {
            Ok(None)
        }
    }

    fn invariants(&self, inputs: &[&TypedFact], outputs: &[&TypedFact]) -> TractResult<Invariants> {
        Invariants::new_element_wise(inputs, outputs)
    }

    fn cost(&self, _inputs: &[&TypedFact]) -> TractResult<TVec<(Cost, TDim)>> {
        Ok(tvec!((Cost::Params(f32::datum_type()), self.0.len().into())))
    }
}
