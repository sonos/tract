use derive_new::new;
use tract_core::internal::*;
use tract_core::tract_linalg::block_quant::{BlockQuantFact, Q8_1};
use tract_gpu::session_handler::{make_scalar_opaque_tensor_for_node};
use tract_gpu::tensor::{DeviceTensorExt};

use crate::context::{CUDA_STREAM};
use crate::kernels::matmul::GgmlQuantQ81;
#[derive(Clone, Debug, new, Hash)]
pub struct CudaGgmlQuantQ81Op {
    mmq: bool
}

impl Op for CudaGgmlQuantQ81Op {
    fn name(&self) -> StaticName {
        "CudaGgmlQuantQ81Op".into()
    }

    op_as_typed_op!();
}

impl EvalOp for CudaGgmlQuantQ81Op {
    fn eval_with_session(
        &self,
        node_id: usize,
        session: &SessionState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        CUDA_STREAM.with(|stream| {
            let opaque = args_1!(inputs);
            let input = opaque.to_device_tensor()?;
            let output = make_scalar_opaque_tensor_for_node(session, node_id, DatumType::Opaque, &[])?;
            GgmlQuantQ81.dispatch_eval(stream, input, &output, self.mmq)?;

            Ok(tvec!(output.into_opaque_tensor().into_tvalue()))
        })
    }

    fn is_stateless(&self) -> bool {
        true
    }
}

impl TypedOp for CudaGgmlQuantQ81Op {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        tract_gpu::utils::facts_to_device_facts(inputs, |facts| {
            let inp_shape = facts[0].shape.as_concrete().expect("Expecting concrete input shape");
            let out_shape = GgmlQuantQ81::output_shape(inp_shape, self.mmq)?;
            let bqf = BlockQuantFact::new(Box::new(Q8_1), out_shape);
            let fact = TypedFact::dt_scalar(DatumType::Opaque).with_opaque_fact(bqf);
            Ok(tvec!(fact))
        })
        .with_context(|| format!("Error while computing facts for {:?}", self.name()))
    }

    as_op!();
}
