use crate::internal::*;

#[derive(Debug, Clone, new, Hash)]
pub struct ConstantOfShape {
    shape: TVec<TDim>,
    scalar: Arc<Tensor>,
}

tract_linalg::impl_dyn_hash!(ConstantOfShape);

impl Op for ConstantOfShape {
    fn name(&self) -> Cow<str> {
        "ConstantOfShape".into()
    }

    op_core!();
    op_as_typed_op!();
}

impl TypedOp for ConstantOfShape {
    fn output_facts(&self, _inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
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
            let tensor = make_tensor(&*shape, &*self.scalar);
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

impl StatefullOp for ConstantOfShape {
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
        Ok(tvec!(make_tensor(&*shape, &op.scalar)))
    }
}

pub fn make_tensor(shape: &[usize], scalar: &Tensor) -> Arc<Tensor> {
    unsafe {
        let mut t = Tensor::uninitialized_dt(scalar.datum_type(), &*shape).unwrap();
        unsafe fn init<T: Datum>(t: &mut Tensor, scalar: &Tensor) {
            let scalar = scalar.to_scalar_unchecked::<T>();
            t.as_slice_mut_unchecked().iter_mut().for_each(|x| *x = scalar.clone())
        }
        dispatch_datum!(init(scalar.datum_type())(&mut t, &scalar));
        t.into_arc_tensor()
    }
}
