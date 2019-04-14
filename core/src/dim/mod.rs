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

pub trait DimLike:
    Copy
    + Clone
    + Default
    + PartialEq
    + From<usize>
    + ::num_traits::One
    + ::num_traits::Zero
    + fmt::Debug
    + fmt::Display
    + ops::Add<Self, Output = Self>
    + ops::Add<usize, Output = Self>
    + ops::Sub<Self, Output = Self>
    + ops::Sub<usize, Output = Self>
    + ops::Mul<usize, Output = Self>
    + ops::Mul<Self, Output = Self>
    + ops::Div<usize, Output = Self>
    + ops::Rem<usize, Output = Self>
    + Send
    + Sync
    + 'static
{
    fn div_ceil(&self, other: usize) -> Self {
        (*self + other - 1) / other
    }

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

#[derive(Copy, Clone, PartialEq, Eq)]
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
    pub fn is_one(&self) -> bool {
        self.as_const().map(|i| i == 1).unwrap_or(false)
    }

    pub fn s() -> TDim {
        TDim(Stack::sym('S'))
    }

    pub fn stream() -> TDim {
        Self::s()
    }

    pub fn as_const(&self) -> Option<i32> {
        self.to_integer().ok()
    }

    pub fn eval(&self, s: i32) -> Option<i32> {
        self.0.eval(&hashmap!('S' => s)).ok()
    }

    pub fn is_stream(&self) -> bool {
        self.as_const().is_none()
    }

    pub fn to_integer(&self) -> TractResult<i32> {
        self.0.eval(&hashmap!())
    }

    pub fn div_ceil(&self, other: TDim) -> TDim {
        TDim(self.0.div_ceil(&other.0))
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

impl ops::AddAssign<TDim> for TDim {
    fn add_assign(&mut self, rhs: TDim) {
        self.0 += rhs.0
    }
}

impl ops::Sub<TDim> for TDim {
    type Output = Self;
    fn sub(mut self, rhs: TDim) -> Self {
        self -= rhs;
        self
    }
}

impl ops::SubAssign<TDim> for TDim {
    fn sub_assign(&mut self, rhs: TDim) {
        self.0 -= rhs.0
    }
}

impl ops::Mul<TDim> for TDim {
    type Output = Self;
    fn mul(mut self, rhs: TDim) -> Self {
        self *= rhs;
        self
    }
}

impl ops::MulAssign<TDim> for TDim {
    fn mul_assign(&mut self, rhs: TDim) {
        self.0 *= rhs.0
    }
}

impl ops::Div<TDim> for TDim {
    type Output = Self;
    fn div(mut self, rhs: TDim) -> TDim {
        self /= rhs;
        self
    }
}

impl ops::DivAssign<TDim> for TDim {
    fn div_assign(&mut self, rhs: TDim) {
        self.0 /= rhs.0
    }
}

impl ops::Rem<TDim> for TDim {
    type Output = Self;
    fn rem(mut self, rhs: TDim) -> TDim {
        self %= rhs;
        self
    }
}

impl ops::RemAssign<TDim> for TDim {
    fn rem_assign(&mut self, rhs: TDim) {
        self.0 %= rhs.0
    }
}

impl ::std::iter::Product for TDim {
    fn product<I: Iterator<Item = TDim>>(iter: I) -> TDim {
        iter.fold(1.to_dim(), |a, b| a * b)
    }
}

pub trait ToDim {
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

impl<I: AsPrimitive<i32>> ops::Div<I> for TDim {
    type Output = Self;
    fn div(self, rhs: I) -> Self {
        self / Self::from(rhs.as_())
    }
}

impl<I: AsPrimitive<i32>> ops::Rem<I> for TDim {
    type Output = Self;
    fn rem(self, rhs: I) -> Self {
        self % Self::from(rhs.as_())
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
        } else {
            s.parse::<i32>().map(|i| i.into())
        }
    }
}
