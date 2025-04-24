use crate::fact::{DeviceFact, DeviceTypedFactExt};
use crate::tensor::{DeviceTensorExt, IntoDevice};
use derive_new::new;
use std::fmt;
use tract_core::internal::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceSyncKind {
    ToHost,
    ToDevice,
}

impl fmt::Display for DeviceSyncKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Debug, Clone, new, Copy, PartialEq, Eq, Hash)]
pub struct DeviceSync {
    pub kind: DeviceSyncKind,
}

impl Op for DeviceSync {
    fn name(&self) -> Cow<str> {
        format!("DeviceSync{}", self.kind).into()
    }

    fn same_as(&self, other: &dyn Op) -> bool {
        let Some(other) = other.downcast_ref::<DeviceSync>() else { return false };
        self == other
    }

    op_as_typed_op!();
}

impl EvalOp for DeviceSync {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        match self.kind {
            DeviceSyncKind::ToHost => {
                let device_tensor = input.to_device_tensor()?;

                let tensor = device_tensor
                    .to_host()
                    .with_context(|| "Error while syncing device tensor to host")?;
                Ok(tvec![tensor.into_tvalue()])
            }
            DeviceSyncKind::ToDevice => {
                let device_input = if let Some(t) = input.as_arc_tensor() {
                    Arc::clone(t).into_device()?
                } else {
                    input.into_tensor().into_device()?
                };
                Ok(tvec![device_input.into_opaque_tensor().into()])
            }
        }
    }
}

impl TypedOp for DeviceSync {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let input = inputs[0];
        match self.kind {
            DeviceSyncKind::ToHost => Ok(tvec![input
                .to_device_fact()
                .with_context(|| {
                    "Cannot sync to Host a tensor without DeviceFact as metadata in its TypedFact"
                })?
                .clone()
                .into_typed_fact()]),
            DeviceSyncKind::ToDevice => {
                ensure!(
                    input.datum_type != DatumType::Opaque,
                    "Cannot sync Opaque Tensor to Device"
                );
                Ok(tvec![DeviceFact::from_host(input.clone())?.into_opaque_fact()])
            }
        }
    }

    as_op!();
}
