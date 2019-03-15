use ndarray::*;
use tract_core::ops::identity::Identity;
use tract_core::ops::prelude::*;

#[derive(Debug, Clone, new, Default)]
pub struct Dropout;

impl Op for Dropout {
    fn name(&self) -> Cow<str> {
        "onnx.Dropout".into()
    }

    fn normalize(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        if node.outputs.len() == 1 || node.outputs[1].successors.len() == 0 {
            Ok(Some(TypedModelPatch::single_unary_op(model, node, Identity)?))
        } else {
            Ok(None)
        }
    }
}

impl StatelessOp for Dropout {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, mut inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        let input = args_1!(inputs);
        let mask = ArrayD::from_elem(input.shape(), true);
        Ok(tvec!(input, mask.into()))
    }
}

impl InferenceRulesOp for Dropout {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 1)?;
        if outputs.len() > 2 || outputs.len() == 0 {
            bail!("Dropout shoud have 1 or 2 outputs, found {}", outputs.len());
        }
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].shape, &outputs[0].shape)?;
        if outputs.len() == 2 {
            s.equals(&outputs[1].datum_type, bool::datum_type())?;
            s.equals(&inputs[0].shape, &outputs[1].shape)?;
        }
        Ok(())
    }
}
