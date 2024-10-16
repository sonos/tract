use std::fmt;
use tract_core::internal::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MetalFactKind {
    Temporary,
    Shared,
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct MetalFact {
    pub kind: MetalFactKind,
    pub fact: TypedFact,
}

impl MetalFact {
    pub fn new(kind: MetalFactKind, fact: TypedFact) -> TractResult<Self> {
        ensure!(fact.as_metal_fact().is_none());
        Ok(Self { kind, fact })
    }

    pub fn shared(fact: TypedFact) -> TractResult<Self> {
        Self::new(MetalFactKind::Shared, fact)
    }

    pub fn marked_as_temporary(self) -> Self {
        Self { kind: MetalFactKind::Temporary, ..self }
    }

    pub fn into_typed_fact(self) -> TypedFact {
        self.fact
    }

    pub fn into_opaque_fact(self) -> TypedFact {
        TypedFact::dt_scalar(DatumType::Opaque).with_opaque_fact(self)
    }
}

impl OpaqueFact for MetalFact {
    fn clarify_dt_shape(&self) -> Option<(DatumType, &[usize])> {
        self.fact.shape.as_concrete().map(|s| (self.fact.datum_type, s))
    }

    fn mem_size(&self) -> TDim {
        self.fact.mem_size()
    }
}

impl fmt::Debug for MetalFact {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match self.kind {
            MetalFactKind::Shared => write!(fmt, "Metal,Shared({:?})", self.fact),
            MetalFactKind::Temporary => write!(fmt, "Metal,Tmp({:?})", self.fact),
        }
    }
}

pub trait MetalTypedFactExt {
    fn to_metal_fact(&self) -> TractResult<&MetalFact>;
    fn as_metal_fact(&self) -> Option<&MetalFact>;
}

impl MetalTypedFactExt for TypedFact {
    fn to_metal_fact(&self) -> TractResult<&MetalFact> {
        ensure!(
            self.datum_type == DatumType::Opaque,
            "Cannot retrieve MetalFact from a non Opaque Tensor"
        );
        self.opaque_fact
            .as_ref()
            .and_then(|m| m.downcast_ref::<MetalFact>())
            .ok_or_else(|| anyhow!("MetalFact not found in Opaque Tensor"))
    }
    fn as_metal_fact(&self) -> Option<&MetalFact> {
        self.opaque_fact.as_ref().and_then(|m| m.downcast_ref::<MetalFact>())
    }
}

impl std::ops::Deref for MetalFact {
    type Target = TypedFact;
    fn deref(&self) -> &Self::Target {
        &self.fact
    }
}

impl std::convert::AsRef<TypedFact> for MetalFact {
    fn as_ref(&self) -> &TypedFact {
        &self.fact
    }
}
