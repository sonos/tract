use crate::internal::*;

// FIXME: pulsification is very fragile

#[derive(Debug, Clone, new)]
pub struct PermuteAxes {
    pub axes: Option<Vec<usize>>,
}

impl PermuteAxes {
    fn compute_shape<D: DimLike>(&self, input: &[D]) -> TVec<D> {
        if let Some(ref axes) = self.axes {
            let mut new_shape = tvec![D::zero(); input.len()];
            for (ix, &d) in axes.iter().enumerate() {
                new_shape[ix] = input[d].clone();
            }
            new_shape
        } else {
            let mut new_shape: TVec<D> = input.iter().cloned().collect();
            new_shape.reverse();
            new_shape
        }
    }

    /// Evaluates the operation given the input tensors.
    fn eval_t<T: Datum>(&self, input: Arc<Tensor>) -> TractResult<TVec<Arc<Tensor>>> {
        if let Some(ref axes) = self.axes {
            Ok(tvec![input
                .into_tensor()
                .into_array::<T>()?
                .permuted_axes(&**axes)
                .into_arc_tensor()])
        } else {
            Ok(tvec![input.into_tensor().into_array::<T>()?.reversed_axes().into_arc_tensor()])
        }
    }
}

impl Op for PermuteAxes {
    fn name(&self) -> Cow<str> {
        "PermuteAxes".into()
    }

    canonic!();
    op_as_typed_op!();
    op_as_pulsed_op!();
}

impl StatelessOp for PermuteAxes {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        dispatch_datum!(Self::eval_t(input.datum_type())(self, input))
    }
}

impl InferenceRulesOp for PermuteAxes {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_output_arity(&outputs, 1)?;
        s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
        s.equals(&outputs[0].rank, &inputs[0].rank)?;
        s.given(&inputs[0].shape, move |s, shape| {
            let output_shape = self.compute_shape(&shape);
            s.equals(&outputs[0].shape, output_shape)
        })
    }

    inference_op_as_op!();
    to_typed!();
}

impl TypedOp for PermuteAxes {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(TypedFact::dt_shape(
            inputs[0].datum_type,
            self.compute_shape(&*inputs[0].shape.to_tvec()).as_ref(),
        )?))
    }

    fn invariants(&self, model: &TypedModel, node: &TypedNode) -> TractResult<Invariants> {
        let permutation = if let Some(axes) = self.axes.clone() {
            axes
        } else {
            (0..model.outlet_fact(node.inputs[0])?.shape.rank()).rev().collect()
        };
        let mut infos = tvec!();
        for (from, to) in permutation.iter().enumerate() {
            infos.push(AxisInfo {
                inputs: tvec!(Some(from)),
                outputs: tvec!(Some(*to)),
                period: 1,
                disposable: true,
            })
        }
        Ok(infos.into())
    }

    fn dispose_dummy_axis(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        axes: &[Option<usize>],
    ) -> TractResult<Option<Box<dyn TypedOp>>> {
        let axis = axes[0].unwrap();
        let permutation = if let Some(axes) = self.axes.clone() {
            axes
        } else {
            (0..model.outlet_fact(node.inputs[0])?.shape.rank()).rev().collect()
        };
        let output_axis = permutation[axis];
        let new_permutation = permutation
            .into_iter()
            .filter(|&src| axis != src)
            .map(|dst| dst - (dst >= output_axis) as usize)
            .collect();
        Ok(Some(Box::new(PermuteAxes::new(Some(new_permutation)))))
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
        if let Some(axes) = &self.axes {
            fact.axis = axes.iter().position(|x| x == &fact.axis).ok_or_else(|| {
                format!("Could not find streaming axis {} if permute axes {:?}", fact.axis, axes)
            })?;
            fact.shape = axes.iter().map(|idx| fact.shape[*idx]).collect();
        }
        target.wire_node(&*node.name, self.clone(), &[input])
    }

    typed_op_as_op!();
}

impl PulsedOp for PermuteAxes {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let mut fact = inputs[0].clone();
        fact.axis = if let Some(axes) = &self.axes {
            axes[fact.axis]
        } else {
            fact.shape.len() - 1 - fact.axis
        };
        fact.shape = self.compute_shape(&*inputs[0].shape);
        Ok(tvec!(fact))
    }

    pulsed_op_as_op!();
    pulsed_op_to_typed_op!();
}
