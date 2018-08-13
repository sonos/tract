use std::fmt;
use std::ops;

use num::cast::AsPrimitive;
use num::rational::Rational;
use num::One;
use num::Zero;

#[derive(Copy, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
pub struct LinearDim {
    a: Rational,
    b: Rational,
}

impl LinearDim {
    pub fn is_one(&self) -> bool {
        *self == Self::one()
    }

    pub fn stream() -> LinearDim {
        LinearDim {
            a: Rational::one(),
            b: Rational::zero(),
        }
    }

    pub fn is_stream(&self) -> bool {
        !self.a.is_zero()
    }

    pub fn to_integer(&self) -> ::Result<isize> {
        Ok(self.to_rational()?.to_integer())
    }

    pub fn to_rational(&self) -> ::Result<Rational> {
        if self.a.is_zero() {
            Ok(self.b)
        } else {
            bail!("Streaming dimension not be eliminated.")
        }
    }

    pub fn ceil(&self) -> LinearDim {
        LinearDim {
            a: self.a.ceil(),
            b: self.b.ceil(),
        }
    }
}

impl Default for LinearDim {
    fn default() -> LinearDim {
        LinearDim::zero()
    }
}

impl fmt::Debug for LinearDim {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        if self.is_zero() {
            write!(fmt, "0")
        } else if !self.a.is_zero() {
            if self.a.denom().is_one() {
                write!(fmt, "{:-}S", self.a.numer())?
            } else {
                write!(fmt, "{:-}S/{}", self.a.numer(), self.a.denom())?
            }
            if !self.b.is_zero() {
                if self.b.denom().is_one() {
                    write!(fmt, "{:+}", self.b.numer())?
                } else {
                    write!(fmt, "{:+}/{}", self.b.numer(), self.b.denom())?
                }
            }
            Ok(())
        } else if self.b.denom().is_one() {
            write!(fmt, "{:-}", self.b.numer())
        } else {
            write!(fmt, "{:-}/{}", self.b.numer(), self.b.denom())
        }
    }
}

impl Zero for LinearDim {
    fn zero() -> Self {
        LinearDim {
            a: Rational::zero(),
            b: Rational::zero(),
        }
    }
    fn is_zero(&self) -> bool {
        self.a.is_zero() && self.b.is_zero()
    }
}

impl One for LinearDim {
    fn one() -> Self {
        LinearDim {
            a: Rational::zero(),
            b: Rational::one(),
        }
    }
}

impl ops::Neg for LinearDim {
    type Output = Self;
    fn neg(self) -> Self {
        LinearDim {
            a: self.a.neg(),
            b: self.b.neg(),
        }
    }
}

impl ops::Add<LinearDim> for LinearDim {
    type Output = Self;
    fn add(self, rhs: LinearDim) -> Self {
        LinearDim {
            a: self.a + rhs.a,
            b: self.b + rhs.b,
        }
    }
}

impl ops::AddAssign<LinearDim> for LinearDim {
    fn add_assign(&mut self, rhs: LinearDim) {
        self.a += rhs.a;
        self.b += rhs.b;
    }
}

impl ops::Sub<LinearDim> for LinearDim {
    type Output = Self;
    fn sub(self, rhs: LinearDim) -> Self {
        LinearDim {
            a: self.a - rhs.a,
            b: self.b - rhs.b,
        }
    }
}

impl ops::SubAssign<LinearDim> for LinearDim {
    fn sub_assign(&mut self, rhs: LinearDim) {
        self.a -= rhs.a;
        self.b -= rhs.b;
    }
}

impl ops::Mul<LinearDim> for LinearDim {
    type Output = Self;
    fn mul(self, rhs: LinearDim) -> Self {
        if !self.a.is_zero() && !rhs.b.is_zero() {
            panic!("Multiplying streaming dims");
        }
        LinearDim {
            a: self.a * rhs.b + rhs.a * self.b,
            b: self.b + rhs.b,
        }
    }
}

impl ops::MulAssign<LinearDim> for LinearDim {
    fn mul_assign(&mut self, rhs: LinearDim) {
        if !self.a.is_zero() && !rhs.b.is_zero() {
            panic!("Multiplying streaming dims");
        }
        self.a = self.a * rhs.b + rhs.a * self.b;
        self.b = self.b + rhs.b;
    }
}

impl ops::Div<LinearDim> for LinearDim {
    type Output = Self;
    fn div(self, rhs: LinearDim) -> LinearDim {
        if !rhs.a.is_zero() {
            panic!("Dividing by a streaming dim");
        }
        LinearDim {
            a: self.a / rhs.b,
            b: self.b / rhs.b,
        }
    }
}

impl ops::DivAssign<LinearDim> for LinearDim {
    fn div_assign(&mut self, rhs: LinearDim) {
        if !rhs.a.is_zero() {
            panic!("Dividing by a streaming dim");
        }
        self.a /= rhs.b;
        self.b /= rhs.b;
    }
}

impl ops::RemAssign<LinearDim> for LinearDim {
    fn rem_assign(&mut self, _rhs: LinearDim) {
        panic!("Remainder of dimension");
    }
}

pub trait ToDim {
    fn to_dim(self) -> LinearDim;
}

impl<I: Into<LinearDim>> ToDim for I {
    fn to_dim(self) -> LinearDim {
        self.into()
    }
}

impl<I: AsPrimitive<isize>> ops::Add<I> for LinearDim {
    type Output = Self;
    fn add(self, rhs: I) -> Self {
        self + Self::from(rhs.as_())
    }
}

impl<I: AsPrimitive<isize>> ops::Sub<I> for LinearDim {
    type Output = Self;
    fn sub(self, rhs: I) -> Self {
        self - Self::from(rhs.as_())
    }
}

impl<I: AsPrimitive<isize>> ops::Mul<I> for LinearDim {
    type Output = Self;
    fn mul(self, rhs: I) -> Self {
        LinearDim {
            a: self.a * rhs.as_(),
            b: self.b * rhs.as_(),
        }
    }
}

impl<I: AsPrimitive<isize>> ops::Div<I> for LinearDim {
    type Output = Self;
    fn div(self, rhs: I) -> Self {
        LinearDim {
            a: self.a / rhs.as_(),
            b: self.b / rhs.as_(),
        }
    }
}

impl<I: AsPrimitive<isize>> From<I> for LinearDim {
    fn from(it: I) -> LinearDim {
        LinearDim {
            a: 0.into(),
            b: it.as_().into(),
        }
    }
}
