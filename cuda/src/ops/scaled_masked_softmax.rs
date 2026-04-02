use crate::kernels::nn::ScaledMaskedSoftmax;
use derive_new::new;
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensorExt;

/// A = SOFTMAX(INPUT * SCALE + MASK, AXIS=2)
/// Only input of rank of 3 is supported
#[derive(Clone, Debug, new, Hash, PartialEq, Eq)]
pub struct CudaScaledMaskedSoftmax {
    pub scale: Arc<Tensor>,
}

impl Op for CudaScaledMaskedSoftmax {
    fn name(&self) -> StaticName {
        "CudaScaledMaskedSoftmax".into()
    }

    op_as_typed_op!();
}

impl EvalOp for CudaScaledMaskedSoftmax {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        node_id: usize,
        session: &TurnState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        crate::with_cuda_stream(|stream| {
            let (input_val, mask_val) = args_2!(inputs);
            let input = input_val.to_device_tensor()?;
            let mask = mask_val.to_device_tensor()?;
            let output = tract_gpu::session_handler::make_tensor_for_node(
                session,
                node_id,
                input.datum_type(),
                input.shape(),
            )?;
            ScaledMaskedSoftmax.dispatch_eval(stream, input, &self.scale, mask, &output)?;
            Ok(tvec!(output.into_tensor().into_tvalue()))
        })
    }
}

impl TypedOp for CudaScaledMaskedSoftmax {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        tract_gpu::utils::facts_to_device_facts(inputs, |facts| {
            ensure!(facts.len() == 2);
            let dt = facts[0].datum_type;
            ensure!(dt == facts[1].datum_type);
            ensure!(facts[0].rank() <= 5);
            ensure!(facts[0].rank() >= 2);
            ensure!(facts[0].rank() == facts[1].rank());
            let fact = dt.fact(facts[0].shape.clone());
            Ok(tvec!(fact))
        })
        .with_context(|| format!("Error while computing facts for {:?}", self.name()))
    }

    as_op!();
}
