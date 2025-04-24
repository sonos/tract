use crate::kernels;
use crate::ops::MetalEvalOp;
use crate::MetalStream;
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

crate::impl_eval_op_for_metal_op!(MetalCast);

impl MetalEvalOp for MetalCast {
    fn metal_eval(
        &self,
        stream: &MetalStream,
        node_id: usize,
        session: &mut SessionState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let opaque = args_1!(inputs);
        let input = opaque.to_device_tensor()?;
        if input.datum_type() == self.to {
            Ok(tvec!(opaque))
        } else {
            let output =
                crate::ops::make_tensor_for_node(session, node_id, self.to, input.shape())?;
            kernels::array::Cast.dispatch_eval(stream, input, &output)?;
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
