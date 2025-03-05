#![allow(clippy::derived_hash_with_manual_eq)]
use crate::datum::DatumType;
use crate::dim::TDim;
use crate::internal::{TVec, Tensor, TractResult};
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::ops::Deref;
use std::sync::Arc;

use downcast_rs::{impl_downcast, Downcast};
use dyn_hash::DynHash;

pub trait OpaquePayload: DynHash + Send + Sync + Debug + Display + Downcast {
    fn clarify_to_tensor(&self) -> TractResult<Option<Arc<Tensor>>> {
        Ok(None)
    }

    fn same_as(&self, other: &dyn OpaquePayload) -> bool;
}
impl_downcast!(OpaquePayload);
dyn_hash::hash_trait_object!(OpaquePayload);

pub trait OpaqueFact: DynHash + Send + Sync + Debug + dyn_clone::DynClone + Downcast {
    fn same_as(&self, other: &dyn OpaqueFact) -> bool;

    /// Whether or not it is acceptable for a Patch to substitute `self` by `other`.
    ///
    /// In other terms, all operators consuming `self` MUST accept also accept `other` without being altered.
    fn compatible_with(&self, other: &dyn OpaqueFact) -> bool {
        self.same_as(other)
    }

    fn clarify_dt_shape(&self) -> Option<(DatumType, &[usize])> {
        None
    }

    fn mem_size(&self) -> TDim;
}

impl_downcast!(OpaqueFact);
dyn_hash::hash_trait_object!(OpaqueFact);
dyn_clone::clone_trait_object!(OpaqueFact);

impl<T: OpaqueFact> From<T> for Box<dyn OpaqueFact> {
    fn from(v: T) -> Self {
        Box::new(v)
    }
}

impl PartialEq for Box<dyn OpaqueFact> {
    fn eq(&self, other: &Self) -> bool {
        self.as_ref().same_as(other.as_ref())
    }
}

impl Eq for Box<dyn OpaqueFact> {}

impl OpaqueFact for TVec<Box<dyn OpaqueFact>> {
    fn mem_size(&self) -> TDim {
        self.iter().map(|it| it.mem_size()).sum()
    }

    fn same_as(&self, other: &dyn OpaqueFact) -> bool {
        other.downcast_ref::<Self>().is_some_and(|o| self == o)
    }
}
impl OpaqueFact for TVec<Option<Box<dyn OpaqueFact>>> {
    fn mem_size(&self) -> TDim {
        self.iter().flatten().map(|it| it.mem_size()).sum()
    }

    fn same_as(&self, other: &dyn OpaqueFact) -> bool {
        other.downcast_ref::<Self>().is_some_and(|o| self == o)
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
