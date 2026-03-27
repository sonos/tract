#![allow(clippy::derived_hash_with_manual_eq)]
use crate::datum::DatumType;
use crate::dim::TDim;
use crate::internal::TVec;
use std::fmt::Debug;

use downcast_rs::{Downcast, impl_downcast};
use dyn_eq::DynEq;
use dyn_hash::DynHash;

pub trait ExoticFact:
    DynHash + dyn_eq::DynEq + Send + Sync + Debug + dyn_clone::DynClone + Downcast
{
    /// Whether or not it is acceptable for a Patch to substitute `self` by `other`.
    ///
    /// In other terms, all operators consuming `self` MUST accept also accept `other` without being altered.
    fn compatible_with(&self, other: &dyn ExoticFact) -> bool {
        self.dyn_eq(other)
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
dyn_eq::eq_trait_object!(ExoticFact);

impl<T: ExoticFact> From<T> for Box<dyn ExoticFact> {
    fn from(v: T) -> Self {
        Box::new(v)
    }
}

impl ExoticFact for TVec<Box<dyn ExoticFact>> {
    fn buffer_sizes(&self) -> TVec<TDim> {
        self.iter().flat_map(|it| it.buffer_sizes()).collect()
    }
}
impl ExoticFact for TVec<Option<Box<dyn ExoticFact>>> {
    fn buffer_sizes(&self) -> TVec<TDim> {
        self.iter().flatten().flat_map(|it| it.buffer_sizes()).collect()
    }
}
