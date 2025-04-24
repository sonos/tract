use crate::kernels::nn::ScaledMaskedSoftmax;
use crate::ops::MetalEvalOp;
use crate::MetalStream;
use derive_new::new;
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensorExt;

/// A = SOFTMAX(INPUT * SCALE + MASK, AXIS=2)
/// Only input of rank of 3 is supported
#[derive(Clone, Debug, new, Hash)]
pub struct MetalScaledMaskedSoftmax {
    pub scale: Arc<Tensor>,
}

impl Op for MetalScaledMaskedSoftmax {
    fn name(&self) -> Cow<str> {
        "MetalScaledMaskedSoftmax".into()
    }

    op_as_typed_op!();
}

impl MetalEvalOp for MetalScaledMaskedSoftmax {
    fn metal_eval(
        &self,
        stream: &MetalStream,
        node_id: usize,
        session: &mut SessionState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let (opaque_input, opaque_mask) = args_2!(inputs);
        let input = opaque_input.to_device_tensor()?;
        let mask = opaque_mask.to_device_tensor()?;
        let output =
            crate::ops::make_tensor_for_node(session, node_id, input.datum_type(), input.shape())?;
        ScaledMaskedSoftmax.dispatch_eval(stream, input, &self.scale, mask, &output)?;
        Ok(tvec!(output.into_opaque_tensor().into_tvalue()))
    }
}

impl TypedOp for MetalScaledMaskedSoftmax {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        tract_gpu::utils::facts_to_device_facts(inputs, |facts| {
            ensure!(facts.len() == 2);
            let dt = facts[0].datum_type;
            ensure!(dt == facts[1].datum_type);
            ensure!(facts[0].rank() == 3 && facts[1].rank() == 3);
            let fact = dt.fact(facts[0].shape.clone());
            Ok(tvec!(fact))
        })
        .with_context(|| format!("Error while computing facts for {:?}", self.name()))
    }

    as_op!();
}

crate::impl_eval_op_for_metal_op!(MetalScaledMaskedSoftmax);
