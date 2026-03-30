use crate::context::StreamExt;
use crate::kernels;
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensorExt;

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct CudaCast {
    pub to: DatumType,
}

impl CudaCast {
    pub fn is_supported_dt(dt: DatumType) -> bool {
        kernels::array::Cast::is_supported_dt(dt)
    }

    pub fn new(to: DatumType) -> Option<Self> {
        Self::is_supported_dt(to).then_some(Self { to })
    }
}

impl Op for CudaCast {
    fn name(&self) -> StaticName {
        "CudaCast".into()
    }

    op_as_typed_op!();
}

impl EvalOp for CudaCast {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        node_id: usize,
        session: &TurnState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let input_value = args_1!(inputs);
        let input = input_value.to_device_tensor()?;
        if input.datum_type() == self.to {
            Ok(tvec!(input_value))
        } else {
            let output = tract_gpu::session_handler::make_tensor_for_node(
                session,
                node_id,
                self.to,
                input.shape(),
            )?;
            tract_gpu::with_stream(|stream| {
                let stream = stream.cuda()?;
                kernels::array::Cast.dispatch_eval(stream, input, &output)
            })?;
            Ok(tvec![output.into_tensor().into_tvalue()])
        }
    }
}

impl TypedOp for CudaCast {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        tract_gpu::utils::facts_to_device_facts(inputs, |facts| {
            Ok(tvec!(self.to.fact(facts[0].shape.clone())))
        })
        .with_context(|| format!("Error while computing facts for {:?}", self.name()))
    }

    as_op!();
}
