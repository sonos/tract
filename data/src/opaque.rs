#![allow(clippy::derived_hash_with_manual_eq)]
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::ops::Deref;
use std::sync::Arc;

use downcast_rs::{impl_downcast, Downcast};
use dyn_hash::DynHash;

pub trait OpaquePayload: DynHash + Send + Sync + Debug + Display + Downcast {}
impl_downcast!(OpaquePayload);
dyn_hash::hash_trait_object!(OpaquePayload);

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
