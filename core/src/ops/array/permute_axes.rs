use crate::internal::*;

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

    fn pulsify(
        &self,
        _source: &NormalizedModel,
        node: &NormalizedNode,
        target: &mut PulsedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        let input = mapping[&node.inputs[0]];
        let mut fact = target.outlet_fact(input)?.clone();
        if let Some(axes) = &self.axes {
            fact.axis = axes.iter().position(|x| x == &fact.axis).ok_or_else(|| {
                format!("Could not find streaming axis {} if permute axes {:?}", fact.axis, axes)
            })?;
            fact.shape = axes.iter().map(|idx| fact.shape[*idx]).collect();
        }
        let id = target.chain_after(input, &*node.name, self.clone(), tvec!(fact))?;
        Ok(tvec!(OutletId::new(id, 0)))
    }
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
}
