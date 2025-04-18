use std::fmt;
use tract_core::internal::*;

/// Origin of the GPU tensor
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GpuTensorOrigin {
    /// GPU tensor outputted by a GPU operator
    /// Can be either: Host or ArenaView
    /// Note: Tensors marked as FromGPU are from asynchronous operations.
    Device,
    /// GPU tensor built from a CPU tensor (CPU op output or Const)
    /// Can be only Host GPU tensor.
    /// Note: Tensors marked as FromCPU are from synchronous operations.
    Host,
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct GpuFact {
    pub origin: GpuTensorOrigin,
    pub fact: TypedFact,
}

impl GpuFact {
    pub fn new(origin: GpuTensorOrigin, fact: TypedFact) -> TractResult<Self> {
        ensure!(fact.as_gpu_fact().is_none());
        let mut fact_wo_cst = fact.clone();
        if fact.opaque_fact.is_some() {
            fact_wo_cst.konst = None;
            fact_wo_cst.uniform = None;
        }
        Ok(Self { origin, fact: fact_wo_cst })
    }

    pub fn from_cpu(fact: TypedFact) -> TractResult<Self> {
        Self::new(GpuTensorOrigin::Host, fact)
    }

    pub fn is_from_gpu(&self) -> bool {
        matches!(self.origin, GpuTensorOrigin::Device)
    }

    pub fn is_from_cpu(&self) -> bool {
        matches!(self.origin, GpuTensorOrigin::Host)
    }

    pub fn into_typed_fact(self) -> TypedFact {
        self.fact
    }

    pub fn into_opaque_fact(self) -> TypedFact {
        TypedFact::dt_scalar(DatumType::Opaque).with_opaque_fact(self)
    }
}

impl OpaqueFact for GpuFact {
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

impl fmt::Debug for GpuFact {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match self.origin {
            GpuTensorOrigin::Host => write!(fmt, "Host({:?})", self.fact),
            GpuTensorOrigin::Device => write!(fmt, "Device({:?})", self.fact),
        }
    }
}

pub trait GpuTypedFactExt {
    fn to_gpu_fact(&self) -> TractResult<&GpuFact>;
    fn as_gpu_fact(&self) -> Option<&GpuFact>;
}

impl GpuTypedFactExt for TypedFact {
    fn to_gpu_fact(&self) -> TractResult<&GpuFact> {
        ensure!(
            self.datum_type == DatumType::Opaque,
            "Cannot retrieve GpuFact from a non Opaque Tensor"
        );
        self.opaque_fact
            .as_ref()
            .and_then(|m| m.downcast_ref::<GpuFact>())
            .ok_or_else(|| anyhow!("GpuFact not found in Opaque Tensor"))
    }
    fn as_gpu_fact(&self) -> Option<&GpuFact> {
        self.opaque_fact.as_ref().and_then(|m| m.downcast_ref::<GpuFact>())
    }
}

impl std::ops::Deref for GpuFact {
    type Target = TypedFact;
    fn deref(&self) -> &Self::Target {
        &self.fact
    }
}

impl std::convert::AsRef<TypedFact> for GpuFact {
    fn as_ref(&self) -> &TypedFact {
        &self.fact
    }
}
