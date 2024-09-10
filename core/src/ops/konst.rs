use crate::internal::*;

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct Const(pub Arc<Tensor>, pub Option<Box<dyn OpaqueFact>>);

impl Const {
    pub fn new(tensor: Arc<Tensor>) -> Const {
        Const(tensor, None)
    }

    pub fn new_with_opaque_fact(tensor: Arc<Tensor>, fact: Box<dyn OpaqueFact>) -> Const {
        Const(tensor, Some(fact))
    }
}

impl Op for Const {
    fn name(&self) -> Cow<str> {
        "Const".into()
    }

    op_as_typed_op!();
    impl_op_same_as!();
}

impl EvalOp for Const {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, _inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        Ok(tvec![Arc::clone(&self.0).into_tvalue()])
    }
}

impl TypedOp for Const {
    as_op!();

    fn output_facts(&self, _inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let fact = TypedFact::from(&self.0);
        if let Some(opaque) = &self.1 {
            Ok(tvec!(fact.with_opaque_fact(opaque.clone())))
        } else {
            Ok(tvec!(fact))
        }
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
                substitute_op: Some(Box::new(Const(new_tensor.into_arc_tensor(), self.1.clone()))),
                wire_changes: tvec!((io, change.clone())),
            }))
        } else {
            Ok(None)
        }
    }

    fn cost(&self, _inputs: &[&TypedFact]) -> TractResult<TVec<(Cost, TDim)>> {
        Ok(tvec!((Cost::Params(self.0.datum_type().unquantized()), self.0.len().into())))
    }

    fn concretize_dims(
        &self,
        _source: &TypedModel,
        node: &TypedNode,
        target: &mut TypedModel,
        _mapping: &HashMap<OutletId, OutletId>,
        values: &SymbolValues,
    ) -> TractResult<TVec<OutletId>> {
        let op = if self.0.datum_type() == TDim::datum_type() {
            let mut tensor = self.0.clone().into_tensor();
            for d in tensor.as_slice_mut::<TDim>()? {
                *d = d.eval(values);
            }
            Const(tensor.into_arc_tensor(), self.1.clone())
        } else {
            self.clone()
        };
        target.wire_node(&node.name, op, &[])
    }
}
