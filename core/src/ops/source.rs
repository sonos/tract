use crate::internal::*;
use std::convert::TryFrom;

#[derive(Debug, Clone, new)]
pub struct Source {
    fact: Box<dyn TensorInfo>
}

impl Op for Source {
    fn name(&self) -> Cow<str> {
        "Source".into()
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        if self.fact.is::<NormalizedTensorInfo>() {
            return Ok(None)
        }
        let fact = model.node_output_facts(node.id)?[0];
        match TypedTensorInfo::try_from(fact.to_tensor_fact()) {
            Ok(fact) => Ok(Some(TypedModelPatch::replace_single_op(model, node, &node.inputs, Source::new(Box::new(fact)))?)),
            _ => Ok(None)
        }
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
}

impl TypedOp for Source {
    typed_op_as_op!();

    fn output_facts(&self, _inputs: TVec<&NormalizedTensorInfo>) -> TractResult<TVec<NormalizedTensorInfo>> {
        if let Some(fact) = self.fact.downcast_ref::<NormalizedTensorInfo>() {
            Ok(tvec!(fact.clone()))
        } else {
            bail!("Untyped source")
        }
    }
}
