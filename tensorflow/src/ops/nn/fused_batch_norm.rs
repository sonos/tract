use tract_core::internal::*;

use crate::model::ParsingContext;
use crate::tfpb::node_def::NodeDef;

pub fn fused_batch_norm(_ctx: &ParsingContext, pb: &NodeDef) -> TractResult<Box<dyn InferenceOp>> {
    let epsilon = pb.get_attr_float::<f32>("epsilon")?;
    Ok(Box::new(FusedBatchNorm::new(epsilon)))
}

#[derive(Debug, Clone, new)]
struct FusedBatchNorm {
    epsilon: f32,
}

impl FusedBatchNorm {
    // (x - mean)*rsqrt_var*scale+offset
    // x*(rsqrt_var*scale) + (offset - mean*rsqrt_var*scale)

    fn coeffs(
        &self,
        scale: &[f32],
        offset: &[f32],
        mean: &[f32],
        variance: &[f32],
    ) -> TractResult<(Vec<f32>, Vec<f32>)> {
        use itertools::izip;
        let alpha = izip!(variance, scale).map(|(v, s)| s / (v + self.epsilon).sqrt()).collect();
        let beta = izip!(offset, mean, &alpha).map(|(o, m, s)| o - m * s).collect();
        Ok((alpha, beta))
    }
}

impl Op for FusedBatchNorm {
    fn name(&self) -> Cow<str> {
        "tf.FusedBatchNorm".into()
    }

    fn validation(&self) -> Validation {
        Validation::Rounding
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let facts = model.node_input_facts(node.id)?;
        if let (Some(scale), Some(offset), Some(mean), Some(variance)) =
            (&facts[1].konst, &facts[2].konst, &facts[3].konst, &facts[4].konst)
        {
            let scale = scale.as_slice::<f32>()?;
            let offset = offset.as_slice::<f32>()?;
            let mean = mean.as_slice::<f32>()?;
            let variance = variance.as_slice::<f32>()?;
            let (alpha, beta) = self.coeffs(scale, offset, mean, variance)?;
            let mut patch = TypedModelPatch::default();
            patch.tap_model(&model, node.inputs[0])?;
            let mul = patch.chain(
                format!("{}-mul", node.name),
                tract_core::ops::math::mul::bin(),
                tvec!(node.outputs[0].fact.clone()),
            )?;
            let id = patch.chain(
                format!("{}-add", node.name),
                tract_core::ops::math::add::bin(),
                tvec!(node.outputs[0].fact.clone()),
            )?;
            patch.plug_const(
                InletId::new(mul, 1),
                format!("{}-slope", node.name),
                tensor1(&*alpha).into_arc_tensor(),
            )?;
            patch.plug_const(
                InletId::new(id, 1),
                format!("{}-offset", node.name),
                tensor1(&*beta).into_arc_tensor(),
            )?;
            patch.shunt_outside(OutletId::new(node.id, 0), OutletId::new(id, 0))?;
            return Ok(Some(patch));
        };
        Ok(None)
    }

    op_as_typed_op!(); // FIXME use to_fixed instead of declutter
}

impl StatelessOp for FusedBatchNorm {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let (data, scale, offset, mean, variance) = args_5!(inputs);
        let mut data = data.into_tensor().into_array::<f32>()?;
        let scale = scale.as_slice::<f32>()?;
        let offset = offset.as_slice::<f32>()?;
        let mean = mean.as_slice::<f32>()?;
        let variance = variance.as_slice::<f32>()?;
        let (alpha, beta) = self.coeffs(scale, offset, mean, variance)?;
        let alpha = tract_core::ndarray::arr1(&*alpha);
        let beta = tract_core::ndarray::arr1(&*beta);
        data *= &alpha;
        data += &beta;
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

    inference_op_as_op!();
    to_typed!();
}

impl TypedOp for FusedBatchNorm {
    typed_op_as_op!();

    fn output_facts(&self, inputs: &[&TypedTensorInfo]) -> TractResult<TVec<TypedTensorInfo>> {
        Ok(tvec!(inputs[0].clone()))
    }
}

