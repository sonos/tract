use crate::MetalStream;
use crate::kernels;
use crate::utils::with_borrowed_metal_stream;
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensorExt;

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct MetalCast {
    pub to: DatumType,
}

impl MetalCast {
    pub fn is_supported_dt(dt: DatumType) -> bool {
        kernels::array::Cast::is_supported_dt(dt)
    }

    pub fn new(to: DatumType) -> Option<Self> {
        Self::is_supported_dt(to).then_some(Self { to })
    }
}

impl Op for MetalCast {
    fn name(&self) -> Cow<str> {
        "MetalCast".into()
    }

    op_as_typed_op!();
    impl_op_same_as!();
}

impl EvalOp for MetalCast {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
            &self,
            node_id: usize,
            session: &SessionState,
            inputs: TVec<TValue>,
        ) -> TractResult<TVec<TValue>> {
        let opaque = args_1!(inputs);
        let input = opaque.to_device_tensor()?;
        if input.datum_type() == self.to {
            Ok(tvec!(opaque))
        } else {
            let output =
                tract_gpu::session_handler::make_tensor_for_node(session, node_id, self.to, input.shape())?;
            with_borrowed_metal_stream(|stream| {
                kernels::array::Cast.dispatch_eval(stream, input, &output)
            })?;
            Ok(tvec![output.into_opaque_tensor().into_tvalue()])
        }
    }
}

impl TypedOp for MetalCast {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        tract_gpu::utils::facts_to_device_facts(inputs, |facts| {
            Ok(tvec!(self.to.fact(facts[0].shape.clone())))
        })
        .with_context(|| format!("Error while computing facts for {:?}", self.name()))
    }

    as_op!();
}
