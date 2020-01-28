use crate::internal::*;

#[derive(Debug, Clone, new)]
pub struct AddDims {
    pub axes: Vec<usize>,
}

impl AddDims {
    fn compute_shape<D: DimLike>(&self, input: &[D]) -> TVec<D> {
        let mut shape: TVec<D> = input.iter().cloned().collect();
        for &axis in &self.axes {
            shape.insert(axis, D::one())
        }
        shape
    }
}

impl Op for AddDims {
    fn name(&self) -> Cow<str> {
        "AddDims".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("Axes: {:?}", self.axes)])
    }

    not_a_typed_op!();
}

impl StatelessOp for AddDims {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        let shape = self.compute_shape(input.shape());
        Ok(unsafe { tvec![input.into_tensor().into_shape(&*shape)?.into_arc_tensor()] })
    }
}

impl InferenceRulesOp for AddDims {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_output_arity(&outputs, 1)?;
        s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
        s.equals(&outputs[0].rank, (&inputs[0].rank).bex() + self.axes.len() as i32)?;
        s.given(&inputs[0].shape, move |s, shape| {
            let output_shape = self.compute_shape(&shape);
            s.equals(&outputs[0].shape, output_shape)
        })
    }

    #[allow(unused_variables)]
    fn to_typed(
        &self,
        source: &InferenceModel,
        node: &InferenceNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        let mut wire = mapping[&node.inputs[0]];
        let mut axes = self.axes.clone();
        axes.sort();
        for axis in axes {
            wire = target.wire_node(
                format!("{}-axis-{}", node.name, axis),
                AddDim::new(axis),
                &[wire],
            )?[0];
        }
        Ok(tvec!(wire))
    }

    inference_op_as_op!();
}

#[derive(Debug, Clone, new)]
pub struct AddDim {
    pub axis: usize,
}

impl Op for AddDim {
    fn name(&self) -> Cow<str> {
        "AddDim".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("Axis: {:?}", self.axis)])
    }

    canonic!();
    op_as_typed_op!();
    op_as_pulsed_op!();
}

impl StatelessOp for AddDim {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        let mut tensor = input.into_tensor();
        tensor.insert_axis(self.axis)?;
        Ok(tvec!(tensor.into_arc_tensor()))
    }
}

impl TypedOp for AddDim {
    typed_op_as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let mut shape = inputs[0].shape.clone();
        shape.insert_axis(self.axis)?;
        Ok(tvec!(TypedFact::dt_shape(inputs[0].datum_type, shape)?))
    }

    fn invariants(&self, _model: &TypedModel, node: &TypedNode) -> TractResult<Invariants> {
        let mut i = 0;
        let mut axes = tvec!();
        for out in 0..node.outputs[0].fact.shape.rank() {
            if out != self.axis {
                axes.push(AxisInfo {
                    inputs: tvec!(Some(i)),
                    outputs: tvec!(Some(out)),
                    period: 1,
                    disposable: true,
                });
                i += 1;
            }
        }
        Ok(axes.into_iter().collect())
    }

    fn dispose_dummy_axis(
        &self,
        _model: &TypedModel,
        _node: &TypedNode,
        axes: &[Option<usize>],
    ) -> TractResult<Option<Box<dyn TypedOp>>> {
        Ok(Some(Box::new(AddDim::new(self.axis - (self.axis > axes[0].unwrap()) as usize))))
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
}

impl PulsedOp for AddDim {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let mut fact = inputs[0].clone();
        fact.shape = inputs[0].shape.clone();
        fact.shape.insert(self.axis, 1);
        fact.axis += (self.axis >= fact.axis) as usize;
        Ok(tvec!(fact))
    }

    pulsed_op_as_op!();
    pulsed_op_to_typed_op!();
}
