use crate::internal::*;
use crate::infer::*;

#[derive(Debug, Clone, new, Default)]
pub struct Crop {
    pub axis: usize,
    pub start: usize,
    pub end: usize,
}

impl Op for Crop {
    fn name(&self) -> Cow<str> {
        "Crop".into()
    }

    not_a_typed_op!();
    not_a_pulsed_op!();
}

impl StatelessOp for Crop {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let slice =
            super::Slice::new(self.axis, self.start, inputs[0].shape()[self.axis] - self.end);
        slice.eval(inputs)
    }
}

impl InferenceRulesOp for Crop {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 1)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].rank, &outputs[0].rank)?;
        s.given(&inputs[0].rank, move |s, rank| {
            (0..rank as usize).try_for_each(|ax| {
                if self.axis == ax {
                    s.equals(
                        &inputs[0].shape[ax],
                        outputs[0].shape[ax].bex() + self.start.to_dim() + self.end.to_dim(),
                    )
                } else {
                    s.equals(&inputs[0].shape[ax], &outputs[0].shape[ax])
                }
            })
        })?;
        Ok(())
    }

    fn to_typed(
        &self,
        _source: &InferenceModel,
        node: &InferenceNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        let len = target.outlet_fact(mapping[&node.inputs[0]])?.shape.dim(self.axis);
        target.wire_node(
            &*node.name,
            super::Slice::new(self.axis as usize, self.start.to_dim(), len - self.end.to_dim()),
            [mapping[&node.inputs[0]]].as_ref(),
        )
    }

    inference_op_as_op!();
}
