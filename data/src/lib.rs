#[macro_use]
extern crate educe;
#[macro_use]
pub extern crate itertools;

#[macro_use]
mod macros;

/// A Smallvec instantiation with 4 embeddable values.
///
/// Used about everywhere in tract, for node inputs and outputs, or
/// tensor dimensions.
pub type TVec<T> = smallvec::SmallVec<[T; 4]>;

pub type TractError = anyhow::Error;
pub type TractResult<T> = anyhow::Result<T>;

pub mod prelude {
    pub use crate::datum::{round_ties_to_even, Blob, Datum, DatumType, QParams};
    pub use crate::dim::{Symbol, SymbolValues, TDim, ToDim};
    pub use crate::tensor::litteral::*;
    pub use crate::tensor::{natural_strides, IntoArcTensor, IntoTensor, Tensor};
    pub use crate::tvec;
    pub use crate::TVec;
    pub use crate::{
        dispatch_copy, dispatch_copy_by_size, dispatch_datum, dispatch_datum_by_size,
        dispatch_floatlike, dispatch_hash, dispatch_numbers, dispatch_signed,
    };
    pub use crate::{TractError, TractResult};
    pub use num_complex::Complex;
    pub use itertools as tract_itertools;
    pub use half::f16;
}

pub mod internal {
    pub use crate::datum::ClampCast;
    pub use crate::dim::{DimLike, TDim, ToDim};
    pub use crate::hash::{dyn_hash, DynHash};
    pub use crate::impl_dyn_hash;
    pub use crate::prelude::*;
    pub use crate::tensor::view::TensorView;
    pub use anyhow::{bail, ensure};
    pub use ndarray as tract_ndarray;
    pub use num_integer;
    pub use smallvec as tract_smallvec;
    pub use num_traits as tract_num_traits;
}

pub use anyhow;
pub use dim::UndeterminedSymbol;
pub use half;

mod datum;
mod dim;
pub mod hash;
mod scatter;
mod tensor;
