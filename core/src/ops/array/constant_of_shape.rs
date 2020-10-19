use crate::internal::*;

#[derive(Debug, Clone, new, Hash)]
pub struct ConstantOfShape {
    shape: TVec<TDim>,
    scalar: Arc<Tensor>,
}

tract_data::impl_dyn_hash!(ConstantOfShape);

impl Op for ConstantOfShape {
    fn name(&self) -> Cow<str> {
        "ConstantOfShape".into()
    }

    op_core!();
    op_as_typed_op!();
}

impl TypedOp for ConstantOfShape {
    fn output_facts(&self, _inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        if self.scalar.rank() > 0 {
            bail!("ConstantOfShape attribute must be a scalar, {:?}", self.scalar)
        }
        Ok(tvec!(TypedFact::dt_shape(self.scalar.datum_type(), &*self.shape)?))
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        if let Ok(shape) =
            self.shape.iter().map(|d| d.to_usize()).collect::<TractResult<Vec<usize>>>()
        {
            let tensor = self.scalar.broadcast_scalar_to_shape(&*shape)?.into_arc_tensor();
            Ok(Some(TypedModelPatch::replace_single_op(
                model,
                node,
                &[],
                crate::ops::konst::Const::new(tensor),
            )?))
        } else {
            Ok(None)
        }
    }

    as_op!();
}

impl EvalOp for ConstantOfShape {
    fn eval(&self, _inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let shape: TVec<_> = self.shape.iter().map(|d| d.to_usize()).collect::<TractResult<_>>()?;
        Ok(tvec!(self.scalar.broadcast_scalar_to_shape(&*shape)?.into_arc_tensor()))
    }

    fn is_stateless(&self) -> bool {
        self.shape.iter().all(|d| d.to_usize().is_ok())
    }

    fn state(
        &self,
        _session: &mut SessionState,
        _node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(Some(Box::new(ConstantOfShapeState)))
    }
}

#[derive(Clone, Debug)]
struct ConstantOfShapeState;

impl OpState for ConstantOfShapeState {
    fn eval(
        &mut self,
        session: &mut SessionState,
        op: &dyn Op,
        _inputs: TVec<Arc<Tensor>>,
    ) -> TractResult<TVec<Arc<Tensor>>> {
        let op = op.downcast_ref::<ConstantOfShape>().unwrap();
        let shape = op
            .shape
            .iter()
            .map(|d| Ok(d.eval(&session.resolved_symbols).to_usize()?))
            .collect::<TractResult<TVec<_>>>()?;
        Ok(tvec!(op.scalar.broadcast_scalar_to_shape(&*shape)?.into_arc_tensor()))
    }
}

