use crate::internal::*;

#[derive(Debug, Clone, new)]
pub struct RmDims {
    pub axes: Vec<usize>,
}

impl RmDims {
    fn compute_shape<D: DimLike>(&self, input: &[D]) -> TVec<D> {
        input
            .iter()
            .enumerate()
            .filter(|(ix, _d)| !self.axes.contains(ix))
            .map(|(_ix, d)| d.clone())
            .collect()
    }

    /// Evaluates the operation given the input tensors.
    fn eval_t<T: Datum>(&self, input: Arc<Tensor>) -> TractResult<TVec<Arc<Tensor>>> {
        let shape = self.compute_shape(input.shape());
        Ok(tvec![input.into_tensor().into_array::<T>()?.into_shape(&*shape)?.into_arc_tensor()])
    }
}

impl Op for RmDims {
    fn name(&self) -> Cow<str> {
        "RmDims".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("axes: {:?}", self.axes)])
    }

    canonic!();
    op_as_typed_op!();
}

impl StatelessOp for RmDims {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        dispatch_datum!(Self::eval_t(input.datum_type())(self, input))
    }
}

impl InferenceRulesOp for RmDims {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_output_arity(&outputs, 1)?;
        s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
        s.equals(&outputs[0].rank, (&inputs[0].rank).bex() - self.axes.len() as i32)?;
        for axis in &self.axes {
            s.equals(&inputs[0].shape[*axis], 1.to_dim())?;
        }
        s.given(&inputs[0].shape, move |s, shape| {
            let output_shape = self.compute_shape(&shape);
            s.equals(&outputs[0].shape, output_shape)
        })
    }

    inference_op_as_op!();
    to_typed!();
}

impl TypedOp for RmDims {
    typed_op_as_op!();

    fn output_facts(&self, inputs: &[&TypedTensorInfo]) -> TractResult<TVec<TypedTensorInfo>> {
        Ok(tvec!(TypedTensorInfo::dt_shape(
            inputs[0].datum_type,
            self.compute_shape(&*inputs[0].shape.to_tvec()).as_ref(),
        )?))
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        'axis: for &rm_axis in &self.axes {
            let mut current = node;
            let mut axis = rm_axis;
            while let Some(prec) = model.single_prec(current.id)? {
                if let Some(add_dims) = prec.op_as::<super::AddDims>() {
                    if add_dims.axes.contains(&axis) {
                        let mut patch = TypedModelPatch::default();
                        let mut wire: OutletId = patch.tap_model(model, prec.inputs[0])?.into();
                        if add_dims.axes.len() > 1 {
                            let mut add_dims = add_dims.clone();
                            add_dims.axes.retain(|&a| a != axis);
                            wire = patch.wire_node(&*prec.name, add_dims, [wire].as_ref())?[0];
                        }
                        let mut next = model.single_succ(prec.id)?.unwrap();
                        while next.id != node.id {
                            let op = next
                                .op
                                .dispose_dummy_axis(model, next, axis)?
                                .unwrap_or_else(|| next.op.clone());
                            wire = patch.wire_node(&*next.name, op, [wire].as_ref())?[0];
                            axis = next
                                .op
                                .axes_info(model, next)?
                                .unary_track_axis_down(axis, true)
                                .unwrap();
                            next = model.single_succ(next.id)?.unwrap();
                        }
                        if self.axes.len() > 1 {
                            let mut rm_dims = self.clone();
                            rm_dims.axes.retain(|&a| a != rm_axis);
                            wire = patch.wire_node(&*node.name, rm_dims, [wire].as_ref())?[0];
                        }
                        patch.shunt_outside(OutletId::new(node.id, 0), wire)?;
                        return Ok(Some(patch));
                    }
                }
                let invariants = prec.op.axes_info(model, prec)?;
                if let Some(up_axis) = invariants.unary_track_axis_up(axis, true) {
                    current = prec;
                    axis = up_axis;
                } else {
                    continue 'axis;
                }
            }
        }
        Ok(None)
    }

    fn pulsify(
        &self,
        _source: &NormalizedModel,
        node: &NormalizedNode,
        target: &mut PulsedModel,
        mapping: &HashMap<OutletId, OutletId>,
        _pulse: usize,
    ) -> TractResult<TVec<OutletId>> {
        let input = mapping[&node.inputs[0]];
        let mut fact = target.outlet_fact(input)?.clone();
        fact.shape = self.compute_shape(&fact.shape);
        fact.axis -= self.axes.iter().filter(|&ax| *ax <= fact.axis).count();
        let id = target.chain_after(input, &*node.name, self.clone(), tvec!(fact))?;
        Ok(tvec!(OutletId::new(id, 0)))
    }
}
