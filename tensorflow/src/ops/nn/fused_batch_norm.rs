use crate::tfpb::node_def::NodeDef;
use tract_core::internal::*;

pub fn fused_batch_norm(node: &NodeDef) -> TractResult<Box<Op>> {
    let epsilon = node.get_attr_float::<f32>("epsilon")?;
    Ok(Box::new(FusedBatchNorm::new(epsilon)))
}

#[derive(Debug, Clone, new)]
struct FusedBatchNorm {
    epsilon: f32,
}

impl Op for FusedBatchNorm {
    fn name(&self) -> Cow<str> {
        "tf.FusedBatchNorm".into()
    }

    fn rounding_errors(&self) -> bool {
        true
    }
}

impl StatelessOp for FusedBatchNorm {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, mut inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        let (data, scale, offset, mean, variance) = args_5!(inputs);
        let mut data = data.into_tensor().into_array::<f32>()?;
        let scale = scale.to_array_view::<f32>()?;
        let offset = offset.to_array_view::<f32>()?;
        let mean = mean.to_array_view::<f32>()?;
        let variance = variance.to_array_view::<f32>()?;
        let rsqrt_var = variance.mapv(|x| (x + self.epsilon).sqrt().recip());
        data -= &mean;
        data *= &rsqrt_var;
        data *= &scale;
        data += &offset;
        Ok(tvec!(data.into_arc_tensor()))
    }
}

impl InferenceRulesOp for FusedBatchNorm {
    /// Registers the inference rules of the operator.
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 5)?;
        s.equals(&inputs[0].datum_type, f32::datum_type())?;
        s.equals(&inputs[1].datum_type, f32::datum_type())?;
        s.equals(&inputs[2].datum_type, f32::datum_type())?;
        s.equals(&inputs[3].datum_type, f32::datum_type())?;
        s.equals(&inputs[4].datum_type, f32::datum_type())?;
        s.equals(&outputs[0].datum_type, f32::datum_type())?;
        s.equals(&inputs[0].rank, 4)?;
        s.equals(&inputs[1].rank, 1)?;
        s.equals(&inputs[2].rank, 1)?;
        s.equals(&inputs[3].rank, 1)?;
        s.equals(&inputs[4].rank, 1)?;
        s.equals(&inputs[0].shape, &outputs[0].shape)?;
        s.equals(&inputs[1].shape[0], &inputs[0].shape[3])?;
        s.equals(&inputs[2].shape[0], &inputs[0].shape[3])?;
        s.equals(&inputs[3].shape[0], &inputs[0].shape[3])?;
        s.equals(&inputs[4].shape[0], &inputs[0].shape[3])?;
        Ok(())
    }
}
