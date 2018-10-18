use std::fmt;
use std::ops;

use num::cast::AsPrimitive;
use num::One;
use num::Zero;

mod stack;
mod tree;

use self::stack::Stack;
use TfdResult;

pub trait DimLike:
    Copy
    + Clone
    + Default
    + PartialEq
    + From<usize>
    + ::num::One
    + ::num::Zero
    + fmt::Debug
    + ops::Add<Self, Output = Self>
    + ops::Add<usize, Output = Self>
    + ops::Sub<Self, Output = Self>
    + ops::Sub<usize, Output = Self>
    + ops::Mul<usize, Output = Self>
    + ops::Div<usize, Output = Self>
    + Send
    + Sync
    + 'static
{
    fn div_ceil(&self, other: usize) -> Self {
        (*self + other - 1) / other
    }

    fn to_integer(&self) -> TfdResult<i64>;
}
impl DimLike for TDim {
    fn to_integer(&self) -> TfdResult<i64> {
        TDim::to_integer(self)
    }
}
impl DimLike for usize {
    fn to_integer(&self) -> TfdResult<i64> {
        Ok(*self as i64)
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

    pub fn as_const(&self) -> Option<i64> {
        self.to_integer().ok()
    }

    pub fn reduce(&mut self) {
        self.0.reduce()
    }

    pub fn eval(&self, s: i64) -> Option<i64> {
        self.0.eval(&hashmap!('S' => s)).ok()
    }

    pub fn is_stream(&self) -> bool {
        self.as_const().is_none()
    }

    pub fn to_integer(&self) -> TfdResult<i64> {
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

pub trait ToDim {
    fn to_dim(self) -> TDim;
}

impl<I: Into<TDim>> ToDim for I {
    fn to_dim(self) -> TDim {
        self.into()
    }
}

impl<I: AsPrimitive<i64>> ops::Add<I> for TDim {
    type Output = Self;
    fn add(self, rhs: I) -> Self {
        self + Self::from(rhs.as_())
    }
}

impl<I: AsPrimitive<i64>> ops::Sub<I> for TDim {
    type Output = Self;
    fn sub(self, rhs: I) -> Self {
        self - Self::from(rhs.as_())
    }
}

impl<I: AsPrimitive<i64>> ops::Mul<I> for TDim {
    type Output = Self;
    fn mul(self, rhs: I) -> Self {
        self * Self::from(rhs.as_())
    }
}

impl<I: AsPrimitive<i64>> ops::Div<I> for TDim {
    type Output = Self;
    fn div(self, rhs: I) -> Self {
        self / Self::from(rhs.as_())
    }
}

impl<I: AsPrimitive<i64>> ops::Rem<I> for TDim {
    type Output = Self;
    fn rem(self, rhs: I) -> Self {
        self % Self::from(rhs.as_())
    }
}

impl<I: AsPrimitive<i64>> From<I> for TDim {
    fn from(it: I) -> TDim {
        TDim(it.as_().into())
    }
}
