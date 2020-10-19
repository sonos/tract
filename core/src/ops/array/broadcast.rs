use crate::internal::*;

#[derive(Debug, Clone, new, Default, Hash)]
pub struct MultiBroadcastTo {
    pub shape: TVec<TDim>,
}
tract_data::impl_dyn_hash!(MultiBroadcastTo);

impl MultiBroadcastTo {
    pub fn eval_t<T: Datum>(input: &Tensor, shape: &[usize]) -> TractResult<TVec<Arc<Tensor>>> {
        let input = input.to_array_view::<T>()?;
        let output = input.broadcast(&*shape).ok_or_else(|| format_err!("incompatible shapes"))?;
        Ok(tvec![output.to_owned().into_arc_tensor()])
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
        true
    }

    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        let dims: Vec<usize> =
            self.shape.iter().map(|d| Ok(d.to_usize()?)).collect::<TractResult<_>>()?;
        dispatch_datum!(Self::eval_t(input.datum_type())(&*input, &*dims))
    }
}

impl TypedOp for MultiBroadcastTo {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(TypedFact::dt_shape(inputs[0].datum_type, &*self.shape)?))
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
        let op = Self { shape: self.shape.iter().map(|d| d.eval(&values)).collect() };
        target.wire_node(&node.name, op, &[input])
    }

    as_op!();
}
