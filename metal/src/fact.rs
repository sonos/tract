use tract_core::internal::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MetalFact(pub TypedFact);

impl MetalFact {
    pub fn new(fact: TypedFact) -> TractResult<Self> {
        ensure!(fact.as_metal_fact().is_none());
        Ok(Self(fact))
    }

    pub fn into_typed_fact(self) -> TypedFact {
        self.0
    }

    pub fn into_opaque_fact(self) -> TypedFact {
        TypedFact::dt_scalar(DatumType::Opaque).with_opaque_fact(self)
    }
}

impl OpaqueFact for MetalFact {}

pub trait MetalTypedFactExt {
    fn into_opaque_metal_fact(self) -> TractResult<TypedFact>;
    fn to_metal_fact(&self) -> TractResult<&MetalFact>;
    fn as_metal_fact(&self) -> Option<&MetalFact>;
}

impl MetalTypedFactExt for TypedFact {
    fn into_opaque_metal_fact(self) -> TractResult<TypedFact> {
        Ok(MetalFact::new(self)?.into_opaque_fact())
    }

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
        &self.0
    }
}

impl std::convert::AsRef<TypedFact> for MetalFact {
    fn as_ref(&self) -> &TypedFact {
        &self.0
    }
}
