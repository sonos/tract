#![allow(clippy::derived_hash_with_manual_eq)]
use crate::TVec;
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::ops::Deref;
use std::sync::Arc;

use downcast_rs::{impl_downcast, Downcast};
use dyn_hash::DynHash;

pub trait OpaquePayload: DynHash + Send + Sync + Debug + Display + Downcast { }
impl_downcast!(OpaquePayload);
dyn_hash::hash_trait_object!(OpaquePayload);


pub trait OpaqueFact: DynHash + Send + Sync + Debug + dyn_clone::DynClone + Downcast {
    fn same_as(&self, _other: &dyn OpaqueFact) -> bool {
        false
    }
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

impl Eq for Box<dyn OpaqueFact> { }


impl OpaqueFact for TVec<Box<dyn OpaqueFact>> { }
impl OpaqueFact for TVec<Option<Box<dyn OpaqueFact>>> { }



#[derive(Debug, Hash, PartialEq, Eq)]
pub struct DummyPayload;

impl OpaquePayload for DummyPayload {}

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
        Arc::ptr_eq(&self.0, &other.0)
    }
}
