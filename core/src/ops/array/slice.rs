use crate::internal::*;
use crate::num_traits::Zero;

#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct Slice {
    pub axis: usize,
    pub start: TDim,
    pub end: TDim,
}

impl Slice {
    pub fn new(axis: usize, start: impl ToDim, end: impl ToDim) -> Slice {
        Slice { axis, start: start.to_dim(), end: end.to_dim() }
    }

    pub fn suffix(&self, name: &str) -> String {
        format!("{}.axis{}_{}_{}", name, self.axis, self.start, self.end)
    }

    pub fn declutter_slice_after_slice(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let prec = model.node(node.inputs[0].node);
        if let Some(other) = prec.op_as::<Slice>() {
            if other.axis == self.axis {
                return TypedModelPatch::replace_single_op(
                    model,
                    node,
                    &prec.inputs,
                    Slice {
                        axis: self.axis,
                        start: self.start.clone() + &other.start,
                        end: self.end.clone() + &other.start,
                    },
                )
                .map(Some);
            }
        }
        Ok(None)
    }
}

impl Op for Slice {
    fn name(&self) -> Cow<str> {
        "Slice".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("axis: {}, {}..{}", self.axis, self.start, self.end)])
    }

    op_as_typed_op!();

    fn same_as(&self, other: &dyn Op) -> bool {
        if let Some(other) = other.downcast_ref::<Self>() {
            other == self
        } else {
            false
        }
    }
}

impl EvalOp for Slice {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        session: &SessionState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        let start = self.start.eval(&session.resolved_symbols).to_usize()?;
        let end = self.end.eval(&session.resolved_symbols).to_usize()?;
        eval_slice(&input, self.axis, start, end)
    }
}

fn eval_slice(input: &Tensor, axis: usize, start: usize, end: usize) -> TractResult<TVec<TValue>> {
    if end > input.shape()[axis] || start > end {
        bail!("Invalid range {}..{} for slicing {:?} on axis {}", start, end, input, axis);
    }
    unsafe {
        let mut shape: TVec<_> = input.shape().into();
        shape[axis] = end - start;
        let mut tensor = Tensor::uninitialized_dt(input.datum_type(), &shape)?;
        tensor.assign_slice_unchecked(.., input, start..end, axis);
        Ok(tvec!(tensor.into_tvalue()))
    }
}

impl TypedOp for Slice {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        anyhow::ensure!(inputs.len() == 1, "Slice has one single input");
        if let (Ok(start), Ok(end), Ok(len)) =
            (self.start.to_usize(), self.end.to_usize(), inputs[0].shape[self.axis].to_usize())
        {
            ensure!(start <= end);
            ensure!(end <= len);
        }
        let mut fact = inputs[0].without_value();
        fact.shape.set(self.axis, (self.end.clone() - &self.start).to_dim());
        Ok(tvec!(fact))
    }

    fn axes_mapping(
        &self,
        inputs: &[&TypedFact],
        outputs: &[&TypedFact],
    ) -> TractResult<AxesMapping> {
        let mut mapping = AxesMapping::disconnected(inputs, outputs)?;
        for (axis, repr) in (0..inputs[0].rank()).zip('a'..) {
            if self.axis != axis {
                mapping = mapping
                    .renaming((InOut::In(0), axis), repr)?
                    .linking(repr, (InOut::Out(0), axis))?;
            }
        }
        Ok(mapping)
    }

    fn change_axes(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        _io: InOut,
        change: &AxisOp,
    ) -> TractResult<Option<AxisChangeConsequence>> {
        if let Some(axis) = change.transform_axis(self.axis) {
            if axis != self.axis {
                Ok(Some(AxisChangeConsequence::new(
                    model,
                    node,
                    Some(Box::new(Slice { axis, ..self.clone() }) as _),
                    change,
                )))
            } else {
                Ok(Some(AxisChangeConsequence::new(model, node, None, change)))
            }
        } else {
            Ok(None)
        }
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        if self.start.is_zero() && (self.end == model.outlet_fact(node.inputs[0])?.shape[self.axis])
        {
            TypedModelPatch::shunt_one_op(model, node)
        } else if let Some(p) = self.declutter_slice_after_slice(model, node)? {
            Ok(Some(p))
        } else {
            Ok(None)
        }
    }

    fn concretize_dims(
        &self,
        _source: &TypedModel,
        node: &TypedNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
        values: &SymbolValues,
    ) -> TractResult<TVec<OutletId>> {
        let op =
            Slice { axis: self.axis, start: self.start.eval(values), end: self.end.eval(values) };
        let inputs = node.inputs.iter().map(|i| mapping[i]).collect::<TVec<_>>();
        target.wire_node(&node.name, op, &inputs)
    }

    fn slice(
        &self,
        patch: &mut TypedModelPatch,
        _model: &TypedModel,
        node: &TypedNode,
        _prefix: &str,
        inputs: &[OutletId],
        _output_axis: usize,
        _start: &TDim,
        _end: &TDim,
    ) -> TractResult<Option<TVec<OutletId>>> {
        patch.wire_node(&node.name, &node.op, inputs).map(Some)
    }

    as_op!();
}
