use crate::fact::{GpuFact, GpuTypedFactExt};
use crate::tensor::{GpuTensorExt, IntoGpu};
use derive_new::new;
use std::fmt;
use tract_core::internal::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuSyncKind {
    ToHost,
    ToDevice,
}

impl fmt::Display for GpuSyncKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Debug, Clone, new, Copy, PartialEq, Eq, Hash)]
pub struct GpuSync {
    pub kind: GpuSyncKind,
}

impl Op for GpuSync {
    fn name(&self) -> Cow<str> {
        format!("GpuSync{}", self.kind).into()
    }

    fn same_as(&self, other: &dyn Op) -> bool {
        let Some(other) = other.downcast_ref::<GpuSync>() else { return false };
        self == other
    }

    op_as_typed_op!();
}

impl EvalOp for GpuSync {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        match self.kind {
            GpuSyncKind::ToHost => {
                let metal_tensor = input.to_gpu_tensor()?;

                let tensor = metal_tensor
                    .to_cpu()
                    .with_context(|| anyhow!("Error while syncing metal tensor to cpu"))?;
                Ok(tvec![tensor.into_tvalue()])
            }
            GpuSyncKind::ToDevice => {
                let metal_input = if let Some(t) = input.as_arc_tensor() {
                    Arc::clone(t).into_gpu()?
                } else {
                    input.into_tensor().into_gpu()?
                };
                Ok(tvec![metal_input.into_opaque_tensor().into()])
            }
        }
    }
}

impl TypedOp for GpuSync {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let input = inputs[0];
        match self.kind {
            GpuSyncKind::ToHost => Ok(tvec![input
                .to_gpu_fact()
                .with_context(|| anyhow!(
                    "Cannot sync to CPU a tensor without metal fact as metadata in its TypedFact"
                ))?
                .clone()
                .into_typed_fact()]),
            GpuSyncKind::ToDevice => {
                ensure!(input.datum_type != DatumType::Opaque, "Cannot sync Opaque Tensor to GPU");
                Ok(tvec![GpuFact::from_cpu(input.clone())?.into_opaque_fact()])
            }
        }
    }

    as_op!();
}
