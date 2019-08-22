use crate::internal::*;

#[derive(Debug, Clone, new)]
pub struct Dummy;

impl Op for Dummy {
    fn name(&self) -> Cow<str> {
        "Dummy".into()
    }

    to_typed!();
}

impl StatelessOp for Dummy {
    fn eval(&self, _inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        bail!("eval() called on a Dummy op. This is a bug.")
    }
}

impl TypedOp for Dummy {
    typed_op_as_op!();

    fn output_facts(
        &self,
        _inputs: TVec<&NormalizedTensorInfo>,
    ) -> TractResult<TVec<NormalizedTensorInfo>> {
        Ok(tvec!())
    }
}

impl InferenceRulesOp for Dummy {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        _s: &mut Solver<'r>,
        _inputs: &'p [TensorProxy],
        _outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        Ok(())
    }

    inference_op_as_op!();
}
