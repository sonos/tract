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
    not_a_typed_op!();
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

    fn to_typed(
        &self,
        _source: &InferenceModel,
        node: &InferenceNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        let mut wire = mapping[&node.inputs[0]];
        let mut axes = self.axes.clone();
        axes.sort();
        for axis in axes.into_iter().rev() {
            wire = target.wire_node(
                format!("{}-axis-{}", node.name, axis),
                RmDim::new(axis),
                &[wire],
            )?[0];
        }
        Ok(tvec!(wire))
    }

    inference_op_as_op!();
}

#[derive(Debug, Clone, new)]
pub struct RmDim {
    pub axis: usize,
}

impl Op for RmDim {
    fn name(&self) -> Cow<str> {
        "RmDim".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("axis: {:?}", self.axis)])
    }

    canonic!();
    op_as_typed_op!();
    op_as_pulsed_op!();
}

impl StatelessOp for RmDim {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let mut input = args_1!(inputs).into_tensor();
        input.remove_axis(self.axis)?;
        Ok(tvec!(input.into_arc_tensor()))
    }
}

impl TypedOp for RmDim {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let mut shape = inputs[0].shape.clone();
        shape.remove_axis(self.axis)?;
        Ok(tvec!(TypedFact::dt_shape(inputs[0].datum_type, shape)?))
    }

    fn invariants(&self, model: &TypedModel, node: &TypedNode) -> TractResult<Invariants> {
        let mut out = 0;
        let mut axes = tvec!();
        for in_ in 0..model.outlet_fact(node.inputs[0])?.shape.rank() {
            if self.axis != out {
                axes.push(AxisInfo {
                    inputs: tvec!(Some(in_)),
                    outputs: tvec!(Some(out)),
                    period: 1,
                    disposable: true,
                });
                out += 1;
            }
        }
        Ok(axes.into_iter().collect())
    }

    fn change_axes(
        &self,
        _model: &TypedModel,
        _node: &TypedNode,
        io: InOut,
        change: &AxisOp,
    ) -> TractResult<Option<AxisChangeConsequence>> {
        /*
        fn rm_axis(axes: &[usize], axis: usize) -> Vec<usize> {
            axes.iter().filter(|&a| a != &axis).map(|&a| a - (a > axis) as usize).collect()
        }
        match change {
            AxisOp::Rm(axis) => match io {
                InOut::In(_) if self.axes.contains(&axis) => Ok(Some(AxisChangeConsequence {
                    substitute_op: Some(Box::new(RmDims::new(rm_axis(&self.axes, *axis)))),
                    wire_changes: tvec!(),
                })),
                InOut::In(_) => Ok(Some(AxisChangeConsequence {
                    substitute_op: Some(Box::new(RmDims::new(rm_axis(&self.axes, *axis)))),
                    wire_changes: tvec!((
                        InOut::Out(0),
                        AxisOp::Rm(axis - self.axes.iter().filter(|&x| x < axis).count())
                    )),
                })),
                InOut::Out(_) => Ok(Some(AxisChangeConsequence {
                    substitute_op: Some(Box::new(RmDims::new(rm_axis(&self.axes, *axis)))),
                    wire_changes: tvec!((
                        InOut::In(0),
                        AxisOp::Rm(axis + self.axes.iter().filter(|&x| x < axis).count())
                    )),
                })),
            },
        }
        */
        todo!();
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
        target.wire_node(&*node.name, self.clone(), &[input])
    }

    typed_op_as_op!();
}

impl PulsedOp for RmDim {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let mut fact = inputs[0].clone();
        fact.shape.remove(self.axis);
        fact.axis -= (self.axis <= fact.axis) as usize;
        Ok(tvec!(fact))
    }

    pulsed_op_as_op!();
    pulsed_op_to_typed_op!();
}
