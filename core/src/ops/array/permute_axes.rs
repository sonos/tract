use crate::internal::*;
use crate::infer::*;

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

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("{:?}", self.axes)])
    }

    not_a_typed_op!();
    not_a_pulsed_op!();
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

    #[allow(unused_variables)]
    fn to_typed(
        &self,
        source: &InferenceModel,
        node: &InferenceNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        if let Some(axes) = &self.axes {
            let op = AxisOp::Permute(axes.iter().cloned().collect());
            target.wire_node(&*node.name, op, &[mapping[&node.inputs[0]]])
        } else if let Some(rank) = source.outlet_fact(node.inputs[0])?.shape.rank().concretize() {
            let axes = (0..rank as usize).rev().collect();
            let op = AxisOp::Permute(axes);
            target.wire_node(&*node.name, op, &[mapping[&node.inputs[0]]])
        } else {
            bail!("Can not typed: no known input rank, and no permutation specified")
        }
    }

    inference_op_as_op!();
}
