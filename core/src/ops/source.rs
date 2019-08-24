use crate::internal::*;

#[derive(Debug, Clone, new)]
pub struct Source;

impl Op for Source {
    fn name(&self) -> Cow<str> {
        "Source".into()
    }
}

impl StatelessOp for Source {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, _inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        panic!("Source should not get evaluated")
    }
}

impl InferenceRulesOp for Source {
    /// Registers the inference rules of the operator.
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        _s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 0)?;
        check_output_arity(&outputs, 1)?;
        Ok(())
    }

    inference_op_as_op!();

    fn to_typed(
        &self,
        source: &InferenceModel,
        node: &InferenceNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        use std::convert::TryInto;
        if let Ok(fact) = node.outputs[0].fact.clone().try_into() {
            crate::ops::trivial_inference_op_to_typed(
                Box::new(TypedSource::new(fact)),
                source,
                node,
                target,
                mapping,
            )
        } else {
            bail!("Output type not determined")
        }
    }
}

#[derive(Debug, Clone, new)]
pub struct TypedSource {
    fact: TypedTensorInfo,
}

impl Op for TypedSource {
    fn name(&self) -> Cow<str> {
        "TypedSource".into()
    }
}

impl StatelessOp for TypedSource {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, _inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        panic!("Source should not get evaluated")
    }
}

impl TypedOp for TypedSource {
    typed_op_as_op!();

    fn output_facts(&self, _inputs: &[&TypedTensorInfo]) -> TractResult<TVec<TypedTensorInfo>> {
        Ok(tvec!(self.fact.clone()))
    }

    fn pulsify(
        &self,
        _source: &NormalizedModel,
        node: &NormalizedNode,
        target: &mut PulsedModel,
        _mapping: &HashMap<OutletId, OutletId>,
        pulse: usize,
    ) -> TractResult<TVec<OutletId>> {
        let pulsed_fact =
            crate::pulse::PulsedTensorFact::from_tensor_fact_pulse(&node.outputs[0].fact, pulse)?;
        let id = target.add_source(node.name.clone(), pulsed_fact)?;
        Ok(tvec!(OutletId::new(id, 0)))
    }
}
