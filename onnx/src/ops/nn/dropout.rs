use crate::model::ParsingContext;
use crate::pb::*;
use ndarray::*;
use tract_core::internal::*;
use tract_core::ops::identity::Identity;

pub fn dropout(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    Ok((Box::new(Dropout::new(node.get_output().len() == 2)), vec![]))
}

#[derive(Debug, Clone, new, Default)]
pub struct Dropout {
    output_mask: bool,
}

impl Op for Dropout {
    fn name(&self) -> Cow<str> {
        "onnx.Dropout".into()
    }

    op_as_typed_op!();
}

impl StatelessOp for Dropout {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        if self.output_mask {
            let input = args_1!(inputs);
            let mask = ArrayD::from_elem(input.shape(), true);
            Ok(tvec!(input, mask.into_arc_tensor()))
        } else {
            Ok(inputs)
        }
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
        check_output_arity(&outputs, 1 + self.output_mask as usize)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].shape, &outputs[0].shape)?;
        if outputs.len() == 2 {
            s.equals(&outputs[1].datum_type, bool::datum_type())?;
            s.equals(&inputs[0].shape, &outputs[1].shape)?;
        }
        Ok(())
    }

    fn nboutputs(&self) -> TractResult<usize> {
        Ok(1 + self.output_mask as usize)
    }

    inference_op_as_op!();
    to_typed!();
}

impl TypedOp for Dropout {
    typed_op_as_op!();
    fn output_facts(&self, inputs: &[&TypedTensorInfo]) -> TractResult<TVec<TypedTensorInfo>> {
        Ok(tvec!(inputs[0].clone()))
    }

    fn declutter(
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
