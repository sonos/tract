use crate::fact::MetalTypedFactExt;
pub use crate::kernels::BinOps;
use crate::tensor::MetalTensorExt;
use crate::IntoMetal;
use crate::MetalFact;
use derive_new::new;
use std::fmt;
use tract_core::internal::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetalSyncKind {
    ToCpu,
    ToGpu,
}

impl fmt::Display for MetalSyncKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Debug, Clone, new, Copy, PartialEq, Eq, Hash)]
pub struct MetalSync {
    pub kind: MetalSyncKind,
}

impl Op for MetalSync {
    fn name(&self) -> Cow<str> {
        format!("MetalSync{}", self.kind).into()
    }

    fn same_as(&self, other: &dyn Op) -> bool {
        let Some(other) = other.downcast_ref::<MetalSync>() else { return false };
        self == other
    }

    op_as_typed_op!();
}

impl EvalOp for MetalSync {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        match self.kind {
            MetalSyncKind::ToCpu => crate::METAL_CONTEXT.with_borrow(|context| {
                context.wait_until_completed()?;
                let metal_tensor = input
                    .to_metal_tensor()
                    .with_context(|| anyhow!("Error while syncing metal tensor to cpu"))?;
                let tvalue = metal_tensor.to_cpu().into();
                Ok(tvec![tvalue])
            }),
            MetalSyncKind::ToGpu => {
                let metal_input = input.into_tensor().into_metal()?;
                Ok(tvec![metal_input.into_opaque_tensor().into()])
            }
        }
    }
}

impl TypedOp for MetalSync {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let input = inputs[0];
        match self.kind {
            MetalSyncKind::ToCpu => Ok(tvec![input
                .to_metal_fact()
                .with_context(|| anyhow!(
                    "Cannot sync to CPU a tensor without metal fact as metadata in its TypedFact"
                ))?
                .clone()
                .into_typed_fact()]),
            MetalSyncKind::ToGpu => {
                ensure!(input.datum_type != DatumType::Opaque, "Cannot sync Opaque Tensor to GPU");
                Ok(tvec![TypedFact::dt_scalar(DatumType::Opaque)
                    .with_opaque_fact(MetalFact::new(input.clone())?)])
            }
        }
    }

    as_op!();
}
