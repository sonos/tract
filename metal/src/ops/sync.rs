pub use crate::kernels::BinOps;
use crate::IntoMetal;
use crate::MetalTensor;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MetalSync {
    kind: MetalSyncKind,
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
            MetalSyncKind::ToCpu => {
                crate::METAL_CONTEXT.with_borrow(|context| 
                    context.wait_until_completed())?;
                    let input = input.into_tensor();
                    let opaque = input.to_scalar::<Opaque>()?;
                    let metal_tensor = opaque.downcast_ref::<MetalTensor>().ok_or_else(|| {
                        anyhow::anyhow!(
                            "Could not sync to cpu because opaque tensor is not a metal tensor"
                        )
                    })?;
                Ok(tvec![metal_tensor.tensor().clone().into()])
            }
            MetalSyncKind::ToGpu => {
                Ok(tvec![input.into_tensor().into_metal()?.into_opaque_tensor().into()])
            }
        }
    }
}

impl TypedOp for MetalSync {
    fn output_facts(&self, _inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        todo!();
    }

    as_op!();
}
