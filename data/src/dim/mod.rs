//! Extended dimension support
use crate::internal::*;
use num_traits::{One, Zero};
use std::fmt;
use std::ops;

mod assertion;
mod parse;
mod resolve;
mod sym;
mod tree;

pub use self::assertion::Assertion;
pub use self::parse::parse_tdim;
pub use self::resolve::solve_for;
pub use self::sym::{Symbol, SymbolScope, SymbolValues};
pub use self::tree::{TDim, TooEarly};

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
    + One
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

    /// Substitute as many symbols as possible in the dim value.
    fn eval(&self, values: &SymbolValues) -> Self;

    /// Full evaluation of the symbol, failing if a symbol is missing
    fn eval_to_i64(&self, values: &SymbolValues) -> TractResult<i64>;

    fn substitute(&self, from: &Symbol, to: &Self) -> TractResult<Self>;

    fn broadcast(self, other: Self) -> TractResult<Self>;
    fn mini(self, other: Self) -> Self;
    fn maxi(self, other: Self) -> Self;

    fn compatible_with(&self, other: &Self) -> bool;
}

impl DimLike for TDim {
    fn maybe_div(&self, other: &Self) -> TractResult<(Self, u64)> {
        if self.is_zero() {
            return Ok((TDim::zero(), 1));
        } else if other.is_zero() {
            bail!("Division by zero")
        }
        fn expand(dim: &TDim) -> (i64, Vec<TDim>) {
            match dim {
                TDim::Mul(terms) => terms.iter().map(expand).fold((1i64, vec![]), |acc, t| {
                    (acc.0 * t.0, acc.1.into_iter().chain(t.1).collect())
                }),
                TDim::MulInt(a, terms) => {
                    let (b, v) = expand(terms);
                    (a * b, v)
                }
                TDim::Val(x) => (*x, vec![]),
                TDim::Add(terms) => {
                    let gcd =
                        terms.iter().map(expand).map(|(n, _)| n).reduce(|a, b| a.gcd(&b)).unwrap();
                    (
                        gcd,
                        vec![TDim::Add(terms.iter().map(|t| t.clone() / gcd).collect()).simplify()],
                    )
                }
                it => (1, vec![it.clone()]),
            }
        }
        let (mut num_int, mut num) = expand(self);
        let (mut denum_int, mut denum) = expand(other);
        if num == denum {
            num = vec![];
            denum = vec![];
        }
        for it in denum {
            if let Some(pos) = num.iter().position(|n| n == &it) {
                num.remove(pos);
            } else {
                bail!("Can't divide {} by {}", self, other)
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

    fn eval(&self, values: &SymbolValues) -> Self {
        self.eval(values)
    }

    fn substitute(&self, from: &Symbol, to: &Self) -> TractResult<Self> {
        self.substitute(from, to)
    }

    fn eval_to_i64(&self, values: &SymbolValues) -> TractResult<i64> {
        TDim::eval_to_i64(self, values)
    }

    fn broadcast(self, other: Self) -> TractResult<Self> {
        if self.is_one() {
            Ok(other)
        } else if other.is_one() {
            Ok(self)
        } else {
            Ok(TDim::Broadcast(vec![self, other]).simplify())
        }
    }

    fn compatible_with(&self, other: &Self) -> bool {
        self.compatible_with(other)
    }

    fn mini(self, other: Self) -> Self {
        TDim::Min(vec![self, other]).simplify()
    }

    fn maxi(self, other: Self) -> Self {
        TDim::Max(vec![self, other]).simplify()
    }
}

impl<'a> std::convert::TryFrom<&'a TDim> for TDim {
    type Error = TractError;
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

    fn eval(&self, _values: &SymbolValues) -> Self {
        *self
    }

    fn substitute(&self, _from: &Symbol, _to: &Self) -> TractResult<Self> {
        Ok(*self)
    }

    fn eval_to_i64(&self, _: &SymbolValues) -> TractResult<i64> {
        Ok(*self as i64)
    }

    fn broadcast(self, other: Self) -> TractResult<Self> {
        if self == 1 || self == other {
            Ok(other)
        } else if other == 1 {
            Ok(self)
        } else {
            bail!("Can not broadcast {self} against {other}")
        }
    }

    fn compatible_with(&self, other: &Self) -> bool {
        self == other
    }

    fn mini(self, other: Self) -> Self {
        if self < other {
            self
        } else {
            other
        }
    }

    fn maxi(self, other: Self) -> Self {
        if self > other {
            self
        } else {
            other
        }
    }
}

impl<'a> std::convert::TryFrom<&'a TDim> for usize {
    type Error = TractError;
    fn try_from(d: &'a TDim) -> TractResult<usize> {
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

impl ToDim for &TDim {
    fn to_dim(&self) -> TDim {
        (*self).clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    lazy_static::lazy_static! {
        static ref S: (SymbolScope, Symbol) = {
            let table = SymbolScope::default();
            let s = table.new_with_prefix("S");
            (table, s)
        };
    }

    pub fn s() -> TDim {
        S.1.clone().into()
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
    fn div_sym_sym_simply_1() {
        assert_eq!((s()).maybe_div(&(s())).unwrap(), (TDim::Val(1), 1));
    }

    #[test]
    fn div_sym_sym_complex() {
        let s = s();
        let b = S.0.sym("b");
        assert_eq!(
            (256.to_dim() * &s * &b).maybe_div(&(1.to_dim() * &s * &b)).unwrap(),
            (256.into(), 1)
        );
    }

    #[test]
    fn div_sym_sym_with_add() {
        assert_eq!((s() * 80 - 160).maybe_div(&(s() - 2)).unwrap(), (80.into(), 1));
    }
}
