#![allow(clippy::derived_hash_with_manual_eq)]
use crate::datum::DatumType;
use crate::dim::TDim;
use crate::internal::TVec;
use std::fmt::Debug;

use downcast_rs::{Downcast, impl_downcast};
use dyn_hash::DynHash;

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
