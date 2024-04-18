#![allow(clippy::derived_hash_with_manual_eq)]
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::sync::Arc;

use dyn_hash::DynHash;

pub trait Payload: DynHash + Send + Sync + Debug + Display {}
dyn_hash::hash_trait_object!(Payload);

#[derive(Debug, Hash, PartialEq, Eq)]
pub struct DummyPayload;

impl Payload for DummyPayload {}

impl Display for DummyPayload {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DummyPayload")
    }
}

#[derive(Clone, Debug, Hash)]
pub struct PayloadWrapper(Arc<dyn Payload>);

impl Display for PayloadWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Default for PayloadWrapper {
    fn default() -> Self {
        PayloadWrapper(Arc::new(DummyPayload))
    }
}

impl PartialEq for PayloadWrapper {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

