//! Extended dimension support
use std::fmt;
use std::ops;

mod tree;

pub use self::tree::TDim;
use crate::TractResult;

/// A super-trait for value acting as tensor dimensions in tract.
///
/// Implemented by:
///
/// * `usize` for regular dimensions
/// * `TDim` supporting regular and streaming dimensions
pub trait DimLike:
    Clone
    + Default
    + PartialEq
    + From<usize>
    + ::num_traits::Zero
    + fmt::Debug
    + fmt::Display
    + std::hash::Hash
    + ops::Add<Self, Output = Self>
    + ops::Add<usize, Output = Self>
    + for<'a> ops::Add<&'a Self, Output = Self>
    + ops::Sub<Self, Output = Self>
    + ops::Sub<usize, Output = Self>
    + for<'a> ops::Sub<&'a Self, Output = Self>
    + ops::Mul<usize, Output = Self>
    + ops::Div<usize, Output = Self>
    + ops::Rem<usize, Output = Self>
    + Send
    + Sync
    + 'static
    + std::iter::Sum
    + ToDim
{
    fn maybe_mul(&self, other: &Self) -> TractResult<Self>;

    /// Integer divise, rounding up to next integer.
    fn div_ceil(&self, other: usize) -> Self {
        (self.clone() + other - 1) / other
    }

    /// Convert to regular integer.
    fn to_integer(&self) -> TractResult<i32>;

    /// do not use num_traits::Mul as it implies a regular Mul
    fn one() -> Self;
}

impl DimLike for TDim {
    fn maybe_mul(&self, other: &Self) -> TractResult<Self> {
        if let Ok(d) = other.to_integer() {
            Ok(self.clone() * d)
        } else if let Ok(a) = self.to_integer() {
            Ok(other.clone() * a)
        } else {
            bail!("product with too many symbols")
        }
    }

    fn to_integer(&self) -> TractResult<i32> {
        TDim::to_integer(self)
    }

    fn one() -> Self {
        Self::from(1)
    }
}

impl DimLike for usize {
    fn maybe_mul(&self, other: &Self) -> TractResult<Self> {
        Ok(self * other)
    }

    fn to_integer(&self) -> TractResult<i32> {
        Ok(*self as i32)
    }

    fn one() -> usize {
        1
    }
}

pub trait MaybeProduct<D> {
    fn maybe_product(self) -> TractResult<D>;
}

impl<D: DimLike, A: std::borrow::Borrow<D>, I: Iterator<Item = A>> MaybeProduct<D> for I {
    fn maybe_product(mut self) -> TractResult<D> {
        self.try_fold(D::one(), |acc, d| acc.maybe_mul(d.borrow()))
    }
}

/// Convenience trait to convert values to TDim.
pub trait ToDim {
    /// Convert self to a TDim.
    fn to_dim(self) -> TDim;
}

impl<I: Into<TDim>> ToDim for I {
    fn to_dim(self) -> TDim {
        self.into()
    }
}
