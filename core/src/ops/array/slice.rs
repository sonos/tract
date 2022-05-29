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

    pub fn suffix(&self, name: &str) -> String {
        format!("{}.axis{}_{}_{}", name, self.axis, self.start, self.end)
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
        self.start.to_usize().is_ok() && self.end.to_usize().is_ok()
    }

    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        let start = self.start.to_usize()?;
        let end = self.end.to_usize()?;
        eval_slice(&input, self.axis, start, end)
    }

    fn state(
        &self,
        _session: &mut SessionState,
        _node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(if !self.is_stateless() { Some(Box::new(self.clone())) } else { None })
    }
}

impl OpState for Slice {
    fn eval(
        &mut self,
        session: &mut SessionState,
        _op: &dyn Op,
        mut inputs: TVec<Arc<Tensor>>,
    ) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        let start = self.start.eval(&session.resolved_symbols).to_usize()?;
        let end = self.end.eval(&session.resolved_symbols).to_usize()?;
        eval_slice(&input, self.axis, start, end)
    }
}

fn eval_slice(
    input: &Tensor,
    axis: usize,
    start: usize,
    end: usize,
) -> TractResult<TVec<Arc<Tensor>>> {
    if end > input.shape()[axis] || start > end {
        bail!("Invalid range {}..{} for slicing {:?} on axis {}", start, end, input, axis);
    }
    unsafe {
        let mut shape: TVec<_> = input.shape().into();
        shape[axis] = end - start;
        let mut tensor = Tensor::uninitialized_dt(input.datum_type(), &shape)?;
        tensor.assign_slice_unchecked(.., &input, start..end, axis);
        Ok(tvec!(tensor.into_arc_tensor()))
    }
}

impl TypedOp for Slice {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let mut fact = inputs[0].without_value();
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
        if let Some((wire, no_slice_op)) = prec.op().as_typed().unwrap().slice_output(
            model,
            prec,
            &mut patch,
            &self.suffix(&node.name),
            node.inputs[0].slot,
            self.axis,
            start,
            end,
        )? {
            /*
            dbg!(node);
            dbg!(prec);
            dbg!(&patch);
            dbg!(no_slice_op);
            */
            if !no_slice_op {
                return Ok(None);
            }
            patch.shunt_outside(model, OutletId::new(node.id, 0), wire)?;
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
    ) -> TractResult<Option<(OutletId, bool)>> {
        let prec = model.node(node.inputs[0].node);
        if axis != self.axis {
            let suffix = self.suffix(&node.name) + "." + suffix;
            return prec
                .op()
                .as_typed()
                .unwrap()
                .slice_output(model, &prec, patch, &suffix, node.inputs[0].slot, axis, start, end)?
                .map(|(w, no_slice_op)| {
                    Ok((
                        patch.wire_node(
                            format!("{}.{}", node.name, &suffix),
                            self.clone(),
                            &[w],
                        )?[0],
                        no_slice_op,
                    ))
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
