use crate::fact::MetalTypedFactExt;
pub use crate::kernels::BinOps;
use crate::ops::MetalEvalOp;
use crate::tensor::MetalTensorExt;
use crate::{IntoMetal, MetalContext, MetalFact};
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

crate::impl_eval_op_for_metal_op!(MetalSync);

impl MetalEvalOp for MetalSync {
    fn metal_eval(
        &self,
        _context: &MetalContext,
        _node_id: usize,
        _session: &mut SessionState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        match self.kind {
            MetalSyncKind::ToCpu => {
                let tvalue = input
                    .to_metal_tensor()
                    .and_then(|t| t.to_cpu())
                    .map(|t| t.into())
                    .with_context(|| anyhow!("Error while syncing metal tensor to cpu"))?;
                Ok(tvec![tvalue])
            }
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
                Ok(tvec![MetalFact::from_cpu(input.clone())?.into_opaque_fact()])
            }
        }
    }

    as_op!();
}
