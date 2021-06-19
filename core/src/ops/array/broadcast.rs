use crate::internal::*;

#[derive(Debug, Clone, new, Hash)]
pub struct MultiBroadcastTo {
    pub shape: ShapeFact,
}

impl_dyn_hash!(MultiBroadcastTo);

impl MultiBroadcastTo {
    pub fn eval_t<T: Datum>(input: &Tensor, shape: &[usize]) -> TractResult<TVec<Arc<Tensor>>> {
        unsafe {
            let view = input.to_array_view_unchecked::<T>();
            let mut output = view
                .broadcast(&*shape)
                .with_context(|| format!("Broadcasting {:?} to {:?}", view, shape))?
                .into_owned()
                .into_tensor();
            output.set_datum_type(input.datum_type());
            Ok(tvec![output.into_arc_tensor()])
        }
    }
}

impl Op for MultiBroadcastTo {
    fn name(&self) -> Cow<str> {
        "MultiBroadcastTo".into()
    }

    op_core_mir!();
    op_as_typed_op!();
}

impl EvalOp for MultiBroadcastTo {
    fn is_stateless(&self) -> bool {
        self.shape.is_concrete()
    }

    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        let dims: Vec<usize> =
            self.shape.iter().map(|d| Ok(d.to_usize()?)).collect::<TractResult<_>>()?;
        dispatch_datum!(Self::eval_t(input.datum_type())(&*input, &*dims))
    }

    fn state(
        &self,
        _session: &mut SessionState,
        _node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(Some(Box::new(MultiBroadcastToState)))
    }
}

#[derive(Clone, Debug)]
struct MultiBroadcastToState;

impl OpState for MultiBroadcastToState {
    fn eval(
        &mut self,
        session: &mut SessionState,
        op: &dyn Op,
        inputs: TVec<Arc<Tensor>>,
    ) -> TractResult<TVec<Arc<Tensor>>> {
        let op = op.downcast_ref::<MultiBroadcastTo>().context("Wrong op")?;
        let shape = op.shape.eval_to_usize(&session.resolved_symbols)?;
        dispatch_datum_by_size!(MultiBroadcastTo::eval_t(inputs[0].datum_type())(
            &inputs[0], &*shape
        ))
    }
}

impl TypedOp for MultiBroadcastTo {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let mut fact = TypedFact::dt_shape(inputs[0].datum_type, self.shape.clone());
        fact.uniform = inputs[0].uniform.clone();
        Ok(tvec!(fact))
    }

    fn concretize_dims(
        &self,
        _source: &TypedModel,
        node: &TypedNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
        values: &SymbolValues,
    ) -> TractResult<TVec<OutletId>> {
        let input = mapping[&node.inputs[0]];
        let op =
            Self { shape: self.shape.iter().map(|d| d.eval(&values)).collect::<TVec<_>>().into() };
        target.wire_node(&node.name, op, &[input])
    }

    as_op!();
}
