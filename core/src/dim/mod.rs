//! Extended dimension support
use std::fmt;
use std::ops;

mod tree;

pub use self::tree::TDim;
use crate::{TractError, TractResult};

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
    + for<'a> std::convert::TryFrom<&'a TDim, Error = TractError>
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
    fn maybe_div(&self, other: &Self) -> TractResult<(Self, u64)>;
    fn maybe_mul(&self, other: &Self) -> TractResult<Self>;

    /// Integer divise, rounding up to next integer.
    fn div_ceil(&self, other: usize) -> Self {
        (self.clone() + other - 1) / other
    }

    /// Convert to regular integer.
    fn to_integer(&self) -> TractResult<i64>;

    /// do not use num_traits::Mul as it implies a regular Mul
    fn one() -> Self;

    /// Give streaming dimension its value
    fn concretize_stream_dim(&self, stream_dim: usize) -> Self;
}

impl DimLike for TDim {
    fn maybe_div(&self, other: &Self) -> TractResult<(Self, u64)> {
        use crate::num_traits::Zero;
        if self.is_zero() {
            return Ok((TDim::zero(), 1));
        } else if other.is_zero() {
            bail!("Division by zero")
        }
        let quotient = match (self.to_integer(), other.to_integer()) {
            (Ok(p), Ok(q)) => {
                let (p, q) = tree::reduce_ratio(p, q);
                (p.into(), q)
            }
            (_, Ok(q)) => (self.clone() / q, 1),
            (_, _) => {
                let slope_p = self.slope();
                let slope_q = other.slope();
                let (p, q) =
                    tree::reduce_ratio(slope_p.0 * slope_q.1 as i64, slope_q.0 * slope_p.1 as i64);
                (p.into(), q)
            }
        };
        if self == &(other.clone().maybe_mul(&quotient.0).unwrap() / quotient.1) {
            Ok(quotient)
        } else {
            bail!("Quotient is a non linear expression ({} / {})", self, other)
        }
    }

    fn maybe_mul(&self, other: &Self) -> TractResult<Self> {
        if let Ok(d) = other.to_integer() {
            Ok(self.clone() * d)
        } else if let Ok(a) = self.to_integer() {
            Ok(other.clone() * a)
        } else {
            bail!("product with too many symbols")
        }
    }

    fn to_integer(&self) -> TractResult<i64> {
        TDim::to_integer(self)
    }

    fn one() -> Self {
        Self::from(1)
    }

    fn concretize_stream_dim(&self, stream_dim: usize) -> Self {
        self.eval(stream_dim as _).unwrap().to_dim()
    }
}

impl<'a> std::convert::TryFrom<&'a TDim> for TDim {
    type Error = crate::errors::TractError;
    fn try_from(d: &'a TDim) -> TractResult<TDim> {
        Ok(d.clone())
    }
}

impl DimLike for usize {
    fn maybe_mul(&self, other: &Self) -> TractResult<Self> {
        Ok(self * other)
    }

    fn maybe_div(&self, other: &Self) -> TractResult<(Self, u64)> {
        use crate::num_integer::Integer;
        let gcd = self.gcd(other);
        Ok((self / gcd, (other / gcd) as u64))
    }

    fn to_integer(&self) -> TractResult<i64> {
        Ok(*self as i64)
    }

    fn one() -> usize {
        1
    }

    fn concretize_stream_dim(&self, _stream_dim: usize) -> Self {
        *self
    }
}

impl<'a> std::convert::TryFrom<&'a TDim> for usize {
    type Error = crate::errors::TractError;
    fn try_from(d: &'a TDim) -> TractResult<usize> {
        d.to_integer().map(|d| d as _)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn div() {
        assert_eq!(TDim::from(12).maybe_div(&TDim::from(4)).unwrap(), (3.into(), 1));
    }

    #[test]
    fn div_sym_int() {
        assert_eq!((TDim::s() * 12).maybe_div(&TDim::from(4)).unwrap(), (TDim::s() * 3, 1));
    }

    #[test]
    fn div_sym_sym() {
        assert_eq!((TDim::s() * 12).maybe_div(&(TDim::s() * 4)).unwrap(), (3.into(), 1));
    }

    #[test]
    fn div_sym_sym_ratio() {
        assert_eq!((TDim::s() * 13).maybe_div(&(TDim::s() * 4)).unwrap(), (13.into(), 4));
    }

    #[test]
    fn div_sym_sym_rem() {
        assert!((TDim::s() + 1).maybe_div(&(TDim::s() * 4)).is_err());
    }
}
