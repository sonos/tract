//! Extended dimension support
use std::fmt;
use std::ops;
use std::str::FromStr;

use num_traits::cast::AsPrimitive;
use num_traits::One;
use num_traits::Zero;

mod stack;
mod tree;

use self::stack::Stack;
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
    + ::num_traits::One
    + ::num_traits::Zero
    + fmt::Debug
    + fmt::Display
    + ops::Add<Self, Output = Self>
    + ops::Add<usize, Output = Self>
    + for<'a> ops::Sub<&'a Self, Output = Self>
    + ops::Sub<Self, Output = Self>
    + ops::Sub<usize, Output = Self>
    + for<'a> ops::Sub<&'a Self, Output = Self>
    + ops::Mul<usize, Output = Self>
    + ops::Div<usize, Output = Self>
    + ops::Rem<usize, Output = Self>
    + Send
    + Sync
    + 'static
    + std::iter::Product
    + std::iter::Sum
{
    /// Integer divise, rounding up to next integer.
    fn div_ceil(&self, other: usize) -> Self {
        (self.clone() + other - 1) / other
    }

    /// Convert to regular integer.
    fn to_integer(&self) -> TractResult<i32>;
}

impl DimLike for TDim {
    fn to_integer(&self) -> TractResult<i32> {
        TDim::to_integer(self)
    }
}

impl DimLike for usize {
    fn to_integer(&self) -> TractResult<i32> {
        Ok(*self as i32)
    }
}

/// An arithmetic expression built with integer and the special value S for
/// the streaming dimension.
#[derive(Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
pub struct TDim(Stack);

impl Default for TDim {
    fn default() -> TDim {
        TDim(0.into())
    }
}

impl fmt::Debug for TDim {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "{:?}", self.0)
    }
}

impl fmt::Display for TDim {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "{:?}", self.0)
    }
}

impl TDim {
    /// Is this value One?
    pub fn is_one(&self) -> bool {
        self.as_const().map(|i| i == 1).unwrap_or(false)
    }

    /// The special value S, for streaming.
    pub fn s() -> TDim {
        TDim(Stack::sym('S'))
    }

    /// The special value S, for streaming.
    pub fn stream() -> TDim {
        Self::s()
    }

    /// Try to convert the value to an integer, if it does not contains S.
    pub fn as_const(&self) -> Option<i32> {
        self.to_integer().ok()
    }

    /// Eval the value for a given value of S.
    pub fn eval(&self, s: i32) -> Option<i32> {
        self.0.eval(&hashmap!('S' => s)).ok()
    }

    /// Is the value dependend on S ?
    pub fn is_stream(&self) -> bool {
        self.as_const().is_none()
    }

    /// Convert to integer if possible.
    pub fn to_integer(&self) -> TractResult<i32> {
        self.0.eval(&hashmap!())
    }

    /// Integer division rounding above.
    pub fn div_ceil(&self, other: u32) -> TDim {
        TDim(self.0.clone().div_ceil(other))
    }
}

impl Zero for TDim {
    fn zero() -> Self {
        Self::from(0)
    }
    fn is_zero(&self) -> bool {
        *self == Self::zero()
    }
}

impl One for TDim {
    fn one() -> Self {
        Self::from(1)
    }
}

impl ops::Neg for TDim {
    type Output = Self;
    fn neg(self) -> Self {
        TDim(-self.0)
    }
}

impl ops::Add<TDim> for TDim {
    type Output = Self;
    fn add(mut self, rhs: TDim) -> Self {
        self += rhs;
        self
    }
}

impl<'a> ops::Add<&'a TDim> for TDim {
    type Output = Self;
    fn add(mut self, rhs: &'a TDim) -> Self {
        self += rhs;
        self
    }
}

impl ops::AddAssign<TDim> for TDim {
    fn add_assign(&mut self, rhs: TDim) {
        self.0 += rhs.0
    }
}

impl<'a> ops::AddAssign<&'a TDim> for TDim {
    fn add_assign(&mut self, rhs: &'a TDim) {
        self.0 += &rhs.0
    }
}

impl ops::Sub<TDim> for TDim {
    type Output = Self;
    fn sub(mut self, rhs: TDim) -> Self {
        self -= rhs;
        self
    }
}

impl<'a> ops::Sub<&'a TDim> for TDim {
    type Output = Self;
    fn sub(mut self, rhs: &'a TDim) -> Self {
        self -= rhs;
        self
    }
}

impl ops::SubAssign<TDim> for TDim {
    fn sub_assign(&mut self, rhs: TDim) {
        self.0 -= rhs.0
    }
}

impl<'a> ops::SubAssign<&'a TDim> for TDim {
    fn sub_assign(&mut self, rhs: &'a TDim) {
        self.0 -= &rhs.0
    }
}

impl ops::Mul<TDim> for TDim {
    type Output = Self;
    fn mul(mut self, rhs: TDim) -> Self {
        self *= rhs;
        self
    }
}

impl<'a> ops::Mul<&'a TDim> for TDim {
    type Output = Self;
    fn mul(mut self, rhs: &'a TDim) -> Self {
        self *= rhs;
        self
    }
}

impl ops::MulAssign<TDim> for TDim {
    fn mul_assign(&mut self, rhs: TDim) {
        self.0 *= rhs.0
    }
}

impl<'a> ops::MulAssign<&'a TDim> for TDim {
    fn mul_assign(&mut self, rhs: &'a TDim) {
        self.0 *= &rhs.0
    }
}

impl ops::DivAssign<u32> for TDim {
    fn div_assign(&mut self, rhs: u32) {
        self.0 /= rhs
    }
}

impl ::std::iter::Sum for TDim {
    fn sum<I: Iterator<Item = TDim>>(iter: I) -> TDim {
        iter.fold(0.to_dim(), |a, b| a + b)
    }
}

impl ::std::iter::Product for TDim {
    fn product<I: Iterator<Item = TDim>>(iter: I) -> TDim {
        iter.fold(1.to_dim(), |a, b| a * b)
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

impl<I: AsPrimitive<i32>> ops::Add<I> for TDim {
    type Output = Self;
    fn add(self, rhs: I) -> Self {
        self + Self::from(rhs.as_())
    }
}

impl<I: AsPrimitive<i32>> ops::Sub<I> for TDim {
    type Output = Self;
    fn sub(self, rhs: I) -> Self {
        self - Self::from(rhs.as_())
    }
}

impl<I: AsPrimitive<i32>> ops::Mul<I> for TDim {
    type Output = Self;
    fn mul(self, rhs: I) -> Self {
        self * Self::from(rhs.as_())
    }
}

impl<I: AsPrimitive<u32>> ops::Div<I> for TDim {
    type Output = Self;
    fn div(self, rhs: I) -> Self {
        TDim(self.0 / rhs.as_())
    }
}

impl<I: AsPrimitive<u32>> ops::Rem<I> for TDim {
    type Output = Self;
    fn rem(self, rhs: I) -> Self {
        TDim(self.0 % rhs.as_())
    }
}

impl From<i64> for TDim {
    fn from(it: i64) -> TDim {
        TDim((it as i32).into())
    }
}

impl From<i32> for TDim {
    fn from(it: i32) -> TDim {
        TDim(it.into())
    }
}

impl From<isize> for TDim {
    fn from(it: isize) -> TDim {
        TDim((it as i32).into())
    }
}

impl From<usize> for TDim {
    fn from(it: usize) -> TDim {
        TDim((it as i32).into())
    }
}

impl<'a> From<&'a usize> for TDim {
    fn from(it: &'a usize) -> TDim {
        TDim((*it as i32).into())
    }
}

impl FromStr for TDim {
    type Err = std::num::ParseIntError;
    fn from_str(s: &str) -> Result<TDim, Self::Err> {
        if s == "S" {
            Ok(TDim::s())
        } else if s.ends_with("S") {
            let number: String = s.chars().take_while(|c| c.is_digit(10)).collect();
            let number: i32 = number.parse::<i32>().map(|i| i.into())?;
            Ok(TDim::s() * number)
        } else {
            s.parse::<i32>().map(|i| i.into())
        }
    }
}
