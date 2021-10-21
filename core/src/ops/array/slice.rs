use crate::internal::*;
use crate::num_traits::Zero;

#[derive(Debug, Clone, Default, PartialEq, Hash)]
pub struct Slice {
    pub axis: usize,
    pub start: TDim,
    pub end: TDim,
}

impl DynHash for Slice {
    fn dyn_hash(&self, hasher: &mut dyn std::hash::Hasher) {
        dyn_hash(&self, hasher)
    }
}

impl Slice {
    pub fn new(axis: usize, start: impl ToDim, end: impl ToDim) -> Slice {
        Slice { axis, start: start.to_dim(), end: end.to_dim() }
    }

    pub fn suffix(&self) -> String {
        format!("axis{}_{}_{}", self.axis, self.start, self.end)
    }
}

impl Op for Slice {
    fn name(&self) -> Cow<str> {
        "Slice".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("axis: {}, {}..{}", self.axis, self.start, self.end)])
    }

    op_core_lir_mir!();
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

    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        unsafe {
            let start = self.start.to_usize()?;
            let end = self.end.to_usize()?;
            let mut shape: TVec<_> = input.shape().into();
            shape[self.axis] = end - start;
            let mut tensor = Tensor::uninitialized_dt(input.datum_type(), &shape)?;
            tensor.assign_slice_unchecked(.., &input, start..end, self.axis);
            Ok(tvec!(tensor.into_arc_tensor()))
        }
    }
}

impl TypedOp for Slice {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let mut fact = inputs[0].clone();
        fact.shape.set(self.axis, (self.end.clone() - &self.start).to_dim());
        Ok(tvec!(fact))
    }

    fn invariants(
        &self,
        inputs: &[&TypedFact],
        _outputs: &[&TypedFact],
    ) -> TractResult<Invariants> {
        let axes = (0..inputs[0].rank())
            .filter(|&ax| self.axis != ax)
            .map(|axis| AxisInfo::simple(axis))
            .collect();
        Ok(axes)
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
        let prec = model.node(node.inputs[0].node);
        if self.start.is_zero() && (self.end == model.outlet_fact(node.inputs[0])?.shape[self.axis])
        {
            return Ok(Some(TypedModelPatch::shunt_one_op(model, node)?.with_context("noop")));
        }
        let (start, end) = if let (Ok(s), Ok(e)) = (self.start.to_usize(), self.end.to_usize()) {
            (s, e)
        } else {
            return Ok(None);
        };
        let mut patch = TypedModelPatch::default();
        if let Some(wire) = prec.op().as_typed().unwrap().slice_output(
            model,
            prec,
            &mut patch,
            &self.suffix(),
            node.inputs[0].slot,
            self.axis,
            start,
            end,
        )? {
            patch.shunt_outside(model, OutletId::new(node.id, 0), wire)?;
            if patch.model.nodes.len() == 2 && patch.model.node(1).op().same_as(self) {
                return Ok(None);
            } else if patch.model.nodes.len() == 3 {
                let other = model.node(node.inputs[0].node);
                if other.op_is::<Self>() {
                    patch.dont_apply_twice = Some(format!("Swap {} and {}", node.name, other.name));
                }
            }
            return Ok(Some(patch));
        }
        Ok(None)
    }

    fn slice_output(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        patch: &mut TypedModelPatch,
        suffix: &str,
        _output_slot: usize,
        axis: usize,
        start: usize,
        end: usize,
    ) -> TractResult<Option<OutletId>> {
        let prec = model.node(node.inputs[0].node);
        if axis != self.axis {
            let suffix = suffix.to_string() + "." + &self.suffix();
            return prec
                .op()
                .as_typed()
                .unwrap()
                .slice_output(model, &prec, patch, &suffix, node.inputs[0].slot, axis, start, end)?
                .map(|w| {
                    Ok(patch.wire_node(format!("{}.{}", node.name, &suffix), self.clone(), &[w])?[0])
                })
                .transpose();
        }
        Ok(None)
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

    as_op!();
}
