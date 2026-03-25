#![allow(clippy::derived_hash_with_manual_eq)]
use crate::datum::DatumType;
use crate::dim::TDim;
use crate::internal::{TVec, Tensor, TractResult};
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::ops::Deref;
use std::sync::Arc;

use downcast_rs::{Downcast, impl_downcast};
use dyn_hash::DynHash;

pub trait OpaquePayload: DynHash + Send + Sync + Debug + Display + Downcast {
    fn clarify_to_tensor(&self) -> TractResult<Option<Arc<Tensor>>> {
        Ok(None)
    }

    fn same_as(&self, other: &dyn OpaquePayload) -> bool;
}
impl_downcast!(OpaquePayload);
dyn_hash::hash_trait_object!(OpaquePayload);

pub trait ExoticFact: DynHash + Send + Sync + Debug + dyn_clone::DynClone + Downcast {
    fn same_as(&self, other: &dyn ExoticFact) -> bool;

    /// Whether or not it is acceptable for a Patch to substitute `self` by `other`.
    ///
    /// In other terms, all operators consuming `self` MUST accept also accept `other` without being altered.
    fn compatible_with(&self, other: &dyn ExoticFact) -> bool {
        self.same_as(other)
    }

    fn clarify_dt_shape(&self) -> Option<(DatumType, TVec<TDim>)> {
        None
    }

    fn buffer_sizes(&self) -> TVec<TDim>;

    fn mem_size(&self) -> TDim {
        self.buffer_sizes().iter().sum::<TDim>()
    }
}

impl_downcast!(ExoticFact);
dyn_hash::hash_trait_object!(ExoticFact);
dyn_clone::clone_trait_object!(ExoticFact);

impl<T: ExoticFact> From<T> for Box<dyn ExoticFact> {
    fn from(v: T) -> Self {
        Box::new(v)
    }
}

impl PartialEq for Box<dyn ExoticFact> {
    fn eq(&self, other: &Self) -> bool {
        self.as_ref().same_as(other.as_ref())
    }
}

impl Eq for Box<dyn ExoticFact> {}

impl ExoticFact for TVec<Box<dyn ExoticFact>> {
    fn same_as(&self, other: &dyn ExoticFact) -> bool {
        other.downcast_ref::<Self>().is_some_and(|o| self == o)
    }

    fn buffer_sizes(&self) -> TVec<TDim> {
        self.iter().flat_map(|it| it.buffer_sizes()).collect()
    }
}
impl ExoticFact for TVec<Option<Box<dyn ExoticFact>>> {
    fn same_as(&self, other: &dyn ExoticFact) -> bool {
        other.downcast_ref::<Self>().is_some_and(|o| self == o)
    }

    fn buffer_sizes(&self) -> TVec<TDim> {
        self.iter().flatten().flat_map(|it| it.buffer_sizes()).collect()
    }
}

#[derive(Debug, Hash, PartialEq, Eq)]
pub struct DummyPayload;

impl OpaquePayload for DummyPayload {
    fn same_as(&self, other: &dyn OpaquePayload) -> bool {
        other.downcast_ref::<Self>().is_some()
    }
}

impl Display for DummyPayload {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DummyPayload")
    }
}

#[derive(Clone, Debug, Hash)]
pub struct Opaque(pub Arc<dyn OpaquePayload>);

impl Opaque {
    pub fn downcast_ref<T: OpaquePayload>(&self) -> Option<&T> {
        (*self.0).downcast_ref::<T>()
    }

    pub fn downcast_mut<T: OpaquePayload>(&mut self) -> Option<&mut T> {
        Arc::get_mut(&mut self.0).and_then(|it| it.downcast_mut::<T>())
    }
}

impl Deref for Opaque {
    type Target = dyn OpaquePayload;
    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}

impl Display for Opaque {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Default for Opaque {
    fn default() -> Self {
        Opaque(Arc::new(DummyPayload))
    }
}

impl PartialEq for Opaque {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0) && self.0.same_as(&*other.0)
    }
}
