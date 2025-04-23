use std::fmt;
use tract_core::internal::*;

/// Origin of the GPU tensor
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DeviceTensorOrigin {
    /// Tensor outputted by a device operator
    /// Can be either a Host or ArenaView tensor
    /// Note: Tensors marked as Device are from asynchronous operations.
    FromDevice,
    /// Tensor built from a CPU tensor (CPU op output or Const)
    /// Can be only Host tensor.
    /// Note: Tensors marked as Host are from synchronous operations.
    FromHost,
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct DeviceFact {
    pub origin: DeviceTensorOrigin,
    pub fact: TypedFact,
}

impl DeviceFact {
    pub fn new(origin: DeviceTensorOrigin, fact: TypedFact) -> TractResult<Self> {
        ensure!(fact.as_device_fact().is_none());
        let mut fact_wo_cst = fact.clone();
        if fact.opaque_fact.is_some() {
            fact_wo_cst.konst = None;
            fact_wo_cst.uniform = None;
        }
        Ok(Self { origin, fact: fact_wo_cst })
    }

    pub fn from_host(fact: TypedFact) -> TractResult<Self> {
        Self::new(DeviceTensorOrigin::FromHost, fact)
    }

    pub fn is_from_device(&self) -> bool {
        matches!(self.origin, DeviceTensorOrigin::FromDevice)
    }

    pub fn is_from_host(&self) -> bool {
        matches!(self.origin, DeviceTensorOrigin::FromHost)
    }

    pub fn into_typed_fact(self) -> TypedFact {
        self.fact
    }

    pub fn into_opaque_fact(self) -> TypedFact {
        TypedFact::dt_scalar(DatumType::Opaque).with_opaque_fact(self)
    }
}

impl OpaqueFact for DeviceFact {
    fn clarify_dt_shape(&self) -> Option<(DatumType, &[usize])> {
        self.fact.shape.as_concrete().map(|s| (self.fact.datum_type, s))
    }

    fn mem_size(&self) -> TDim {
        self.fact.mem_size()
    }
    fn same_as(&self, other: &dyn OpaqueFact) -> bool {
        other.downcast_ref::<Self>().is_some_and(|o| o == self)
    }
    fn compatible_with(&self, other: &dyn OpaqueFact) -> bool {
        other.is::<Self>()
    }
}

impl fmt::Debug for DeviceFact {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match self.origin {
            DeviceTensorOrigin::FromHost => write!(fmt, "FromHost({:?})", self.fact),
            DeviceTensorOrigin::FromDevice => write!(fmt, "FromDevice({:?})", self.fact),
        }
    }
}

pub trait DeviceTypedFactExt {
    fn to_device_fact(&self) -> TractResult<&DeviceFact>;
    fn as_device_fact(&self) -> Option<&DeviceFact>;
}

impl DeviceTypedFactExt for TypedFact {
    fn to_device_fact(&self) -> TractResult<&DeviceFact> {
        ensure!(
            self.datum_type == DatumType::Opaque,
            "Cannot retrieve DeviceFact from a non Opaque Tensor"
        );
        self.opaque_fact
            .as_ref()
            .and_then(|m| m.downcast_ref::<DeviceFact>())
            .ok_or_else(|| anyhow!("DeviceFact not found in Opaque Tensor"))
    }
    fn as_device_fact(&self) -> Option<&DeviceFact> {
        self.opaque_fact.as_ref().and_then(|m| m.downcast_ref::<DeviceFact>())
    }
}

impl std::ops::Deref for DeviceFact {
    type Target = TypedFact;
    fn deref(&self) -> &Self::Target {
        &self.fact
    }
}

impl std::convert::AsRef<TypedFact> for DeviceFact {
    fn as_ref(&self) -> &TypedFact {
        &self.fact
    }
}
