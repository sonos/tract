//! Extended dimension support
use num_traits::Zero;
use std::fmt;
use std::ops;

mod tree;

pub use self::tree::{Symbol, SymbolValues, TDim, UndeterminedSymbol};
type TractError = anyhow::Error;
type TractResult<T> = anyhow::Result<T>;

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
    + ops::Mul<Self, Output = Self>
    + ops::Mul<usize, Output = Self>
    + for<'a> ops::Mul<&'a Self, Output = Self>
    + ops::Div<usize, Output = Self>
    + ops::Rem<usize, Output = Self>
    + Send
    + Sync
    + 'static
    + std::iter::Sum
    + std::iter::Product
    + ToDim
{
    fn maybe_div(&self, other: &Self) -> TractResult<(Self, u64)>;

    /// Integer divise, rounding up to next integer.
    fn divceil(&self, other: usize) -> Self {
        (self.clone() + other - 1) / other
    }

    /// Convert to regular integer.
    fn to_i64(&self) -> TractResult<i64>;

    fn to_usize(&self) -> TractResult<usize> {
        self.to_i64().map(|d| d as usize)
    }

    fn to_isize(&self) -> TractResult<isize> {
        self.to_i64().map(|d| d as isize)
    }

    fn to_i32(&self) -> TractResult<i32> {
        self.to_i64().map(|d| d as i32)
    }

    /// do not use num_traits::Mul as it implies a regular Mul
    fn one() -> Self;

    fn eval(&self, values: &SymbolValues) -> Self;
}

impl DimLike for TDim {
    fn maybe_div(&self, other: &Self) -> TractResult<(Self, u64)> {
        if self.is_zero() {
            return Ok((TDim::zero(), 1));
        } else if other.is_zero() {
            anyhow::bail!("Division by zero")
        }
        fn expand(dim: &TDim) -> (i64, Vec<TDim>) {
            match dim {
                TDim::Mul(terms) => terms.iter().map(expand).fold((1i64, vec![]), |acc, t| {
                    (acc.0 * t.0, acc.1.into_iter().chain(t.1.into_iter()).collect())
                }),
                TDim::MulInt(a, terms) => {
                    let (b, v) = expand(terms);
                    (a * b, v)
                }
                TDim::Val(x) => (*x, vec![]),
                it => (1, vec![it.clone()]),
            }
        }
        let (mut num_int, mut num) = expand(self);
        let (mut denum_int, denum) = expand(other);
        for it in denum {
            if let Some(pos) = num.iter().position(|n| n == &it) {
                num.remove(pos);
            } else {
                anyhow::bail!("Can't divide {} by {}", self, other)
            }
        }
        use num_integer::Integer;
        if denum_int < 0 {
            num_int *= -1;
            denum_int *= -1;
        }
        let gcd = num_int.gcd(&denum_int);
        num_int /= gcd;
        denum_int /= gcd;
        Ok(((TDim::Mul(num) * num_int).reduce(), denum_int as u64))
    }

    fn to_i64(&self) -> TractResult<i64> {
        TDim::to_i64(self)
    }

    fn one() -> Self {
        Self::from(1)
    }

    fn eval(&self, values: &SymbolValues) -> Self {
        self.eval(values)
    }
}

impl<'a> std::convert::TryFrom<&'a TDim> for TDim {
    type Error = anyhow::Error;
    fn try_from(d: &'a TDim) -> TractResult<TDim> {
        Ok(d.clone())
    }
}

impl DimLike for usize {
    fn maybe_div(&self, other: &Self) -> TractResult<(Self, u64)> {
        use num_integer::Integer;
        let gcd = self.gcd(other);
        Ok((self / gcd, (other / gcd) as u64))
    }

    fn to_i64(&self) -> TractResult<i64> {
        Ok(*self as i64)
    }

    fn one() -> usize {
        1
    }

    fn eval(&self, _values: &SymbolValues) -> Self {
        *self
    }
}

impl<'a> std::convert::TryFrom<&'a TDim> for usize {
    type Error = anyhow::Error;
    fn try_from(d: &'a TDim) -> anyhow::Result<usize> {
        d.to_usize()
    }
}

/// Convenience trait to convert values to TDim.
pub trait ToDim {
    /// Convert self to a TDim.
    fn to_dim(&self) -> TDim;
}

impl<I: Into<TDim> + Clone> ToDim for I {
    fn to_dim(&self) -> TDim {
        self.clone().into()
    }
}

impl<'a> ToDim for &'a TDim {
    fn to_dim(&self) -> TDim {
        self.clone().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    lazy_static::lazy_static! {
        static ref S: Symbol = crate::dim::Symbol::new('S');
    }

    pub fn s() -> TDim {
        (*S).into()
    }

    #[test]
    fn div() {
        assert_eq!(TDim::from(12).maybe_div(&TDim::from(4)).unwrap(), (3.into(), 1));
    }

    #[test]
    fn div_sym_int() {
        assert_eq!((s() * 12).maybe_div(&TDim::from(4)).unwrap(), (s() * 3, 1));
    }

    #[test]
    fn div_sym_sym() {
        assert_eq!((s() * 12).maybe_div(&(s() * 4)).unwrap(), (3.into(), 1));
    }

    #[test]
    fn div_sym_sym_ratio() {
        assert_eq!((s() * 13).maybe_div(&(s() * 4)).unwrap(), (13.into(), 4));
    }

    #[test]
    fn div_sym_sym_rem() {
        assert!((s() + 1).maybe_div(&(s() * 4)).is_err());
    }

    #[test]
    fn div_sym_sym_complex() {
        assert_eq!(
            (256.to_dim() * 's' * 'b').maybe_div(&(1.to_dim() * 's' * 'b')).unwrap(),
            (256.into(), 1)
        );
    }
}
