use std::fmt::Debug;

use dyn_clone::DynClone;

use crate::internal::*;

#[derive(Debug, Clone, new, Hash, Eq, PartialEq)]
pub struct Const(pub Arc<Tensor>);

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
        Ok(tvec![self.0.clone().into_tvalue()])
    }
}

impl TypedOp for Const {
    as_op!();

    fn output_facts(&self, _inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(Arc::clone(&self.0).into()))
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
            Const(tensor.into_arc_tensor())
        } else {
            self.clone()
        };
        target.wire_node(&node.name, op, &[])
    }
}

#[derive(Debug, Clone, new)]
pub struct LazyConst(pub Arc<dyn LazyConstProvider>);

impl Op for LazyConst {
    fn name(&self) -> Cow<str> {
        "LazyConst".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec!(format!("{:?}", self.0)))
    }

    op_as_typed_op!();
}

impl EvalOp for LazyConst {
    fn is_stateless(&self) -> bool {
        false
    }

    fn state(
        &self,
        _session: &mut SessionState,
        _node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(Some(Box::new(self.clone())))
    }
}

impl OpState for LazyConst {
    fn eval(
        &mut self,
        _session: &mut SessionState,
        _op: &dyn Op,
        _inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        Ok(tvec!(self.0.eval()?))
    }
}

trivial_op_state_freeeze!(LazyConst);

impl TypedOp for LazyConst {
    as_op!();

    fn output_facts(&self, _inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(self.0.output_fact()?))
    }
}

pub trait LazyConstProvider: DynClone + Debug + Send + Sync + 'static {
    fn output_fact(&self) -> TractResult<TypedFact>;
    fn eval(&self) -> TractResult<TValue>;
}

dyn_clone::clone_trait_object!(LazyConstProvider);
