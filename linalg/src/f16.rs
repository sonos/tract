use std::{fmt, ops};

#[allow(non_camel_case_types)]
#[derive(Copy, Clone, Default, PartialEq, PartialOrd, Debug)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
pub struct f16(pub half::f16);

macro_rules! binary_f16 {
    ($f:ident) => {
        fn $f(self, other:f16) -> f16 {
            (self.0).to_f32().$f((other.0).to_f32()).into()
        }
    }
}

macro_rules! unary_as_f32 {
    ($f:ident) => {
        fn $f(self) -> f16 {
            (self.0).to_f32().$f().into()
        }
    };
}

macro_rules! unary_f16 {
    ($f:ident, $t:ty) => {
        fn $f(self) -> $t {
            (self.0).$f()
        }
    }
}

macro_rules! const_f16 {
    ($f:ident, $c:ident) => {
        fn $f() -> f16 {
            f16(half::consts::$c)
        }
    };
}

#[allow(deprecated)]
impl num_traits::Float for f16 {
    unary_as_f32!(floor);
    unary_as_f32!(ceil);
    unary_as_f32!(round);
    unary_as_f32!(trunc);
    unary_as_f32!(fract);
    unary_as_f32!(abs);
    unary_as_f32!(recip);
    unary_as_f32!(sqrt);
    unary_as_f32!(exp);
    unary_as_f32!(exp2);
    unary_as_f32!(ln);
    unary_as_f32!(log2);
    unary_as_f32!(log10);
    unary_as_f32!(cbrt);
    unary_as_f32!(sin);
    unary_as_f32!(cos);
    unary_as_f32!(tan);
    unary_as_f32!(sinh);
    unary_as_f32!(cosh);
    unary_as_f32!(tanh);
    unary_as_f32!(asin);
    unary_as_f32!(acos);
    unary_as_f32!(atan);
    unary_as_f32!(asinh);
    unary_as_f32!(acosh);
    unary_as_f32!(atanh);
    unary_as_f32!(exp_m1);
    unary_as_f32!(ln_1p);
    unary_f16!(classify, ::std::num::FpCategory);
    unary_f16!(is_nan, bool);
    unary_f16!(is_infinite, bool);
    unary_f16!(is_finite, bool);
    unary_f16!(is_normal, bool);
    unary_f16!(is_sign_positive, bool);
    unary_f16!(is_sign_negative, bool);
    binary_f16!(powf);
    binary_f16!(log);
    binary_f16!(max);
    binary_f16!(min);
    binary_f16!(abs_sub);
    binary_f16!(hypot);
    binary_f16!(atan2);
    const_f16!(nan, NAN);
    const_f16!(infinity, INFINITY);
    const_f16!(neg_infinity, NEG_INFINITY);
    const_f16!(neg_zero, NEG_ZERO);
    const_f16!(max_value, MAX);
    const_f16!(min_value, MIN);
    const_f16!(min_positive_value, MIN_POSITIVE);
    fn signum(self) -> f16 {
        f16(self.0.signum())
    }
    fn mul_add(self, a: f16, b: f16) -> f16 {
        (self.0)
            .to_f32()
            .mul_add((a.0).to_f32(), (b.0).to_f32())
            .into()
    }
    fn powi(self, i: i32) -> f16 {
        (self.0).to_f32().powi(i).into()
    }
    fn sin_cos(self) -> (f16, f16) {
        let (s, c) = (self.0).to_f32().sin_cos();
        (s.into(), c.into())
    }
    fn integer_decode(self) -> (u64, i16, i8) {
        (self.0).to_f32().integer_decode()
    }
}

impl num_traits::Num for f16 {
    type FromStrRadixErr = <f32 as num_traits::Num>::FromStrRadixErr;
    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        f32::from_str_radix(str, radix).map(|it| it.into())
    }
}

impl num_traits::Zero for f16 {
    fn is_zero(&self) -> bool {
        f32::from(self.0).is_zero()
    }
    fn zero() -> f16 {
        0.0f32.into()
    }
}

impl num_traits::One for f16 {
    fn one() -> f16 {
        1.0f32.into()
    }
}

impl num_traits::ToPrimitive for f16 {
    fn to_i64(&self) -> Option<i64> {
        f32::from(self.0).to_i64()
    }
    fn to_u64(&self) -> Option<u64> {
        f32::from(self.0).to_u64()
    }
}

impl num_traits::AsPrimitive<usize> for f16 {
    fn as_(self) -> usize {
        self.0.to_f32() as usize
    }
}

impl num_traits::AsPrimitive<f32> for f16 {
    fn as_(self) -> f32 {
        self.0.to_f32()
    }
}

impl num_traits::AsPrimitive<f16> for f32 {
    fn as_(self) -> f16 {
        f16(half::f16::from_f32(self))
    }
}

impl num_traits::AsPrimitive<f64> for f16 {
    fn as_(self) -> f64 {
        self.0.to_f64()
    }
}

impl num_traits::AsPrimitive<f16> for f64 {
    fn as_(self) -> f16 {
        f16(half::f16::from_f64(self))
    }
}

impl num_traits::NumCast for f16 {
    fn from<T: num_traits::ToPrimitive>(n: T) -> Option<Self> {
        n.to_f32().map(|f| f16(half::f16::from_f32(f)))
    }
}

impl ops::Neg for f16 {
    type Output = f16;
    fn neg(self) -> f16 {
        self.0.to_f32().neg().into()
    }
}

impl From<f32> for f16 {
    fn from(f: f32) -> f16 {
        f16(half::f16::from_f32(f))
    }
}

impl fmt::Display for f16 {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        self.0.fmt(fmt)
    }
}

impl num_traits::AsPrimitive<f16> for usize {
    fn as_(self) -> f16 {
        f16(half::f16::from_f64(self as f64))
    }
}

impl ops::Add<f16> for f16 {
    type Output = f16;
    fn add(self, other: f16) -> f16 {
        (self.0.to_f32() + other.0.to_f32()).into()
    }
}

impl ops::Sub<f16> for f16 {
    type Output = f16;
    fn sub(self, other: f16) -> f16 {
        (self.0.to_f32() - other.0.to_f32()).into()
    }
}

impl ops::Mul<f16> for f16 {
    type Output = f16;
    fn mul(self, other: f16) -> f16 {
        (self.0.to_f32() * other.0.to_f32()).into()
    }
}

impl ops::Div<f16> for f16 {
    type Output = f16;
    fn div(self, other: f16) -> f16 {
        (self.0.to_f32() / other.0.to_f32()).into()
    }
}

impl ops::Rem<f16> for f16 {
    type Output = f16;
    fn rem(self, other: f16) -> f16 {
        (self.0.to_f32() % other.0.to_f32()).into()
    }
}

impl std::iter::Sum for f16 {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = f16>,
    {
        iter.fold(0.0f32, |acc, i| acc + i.0.to_f32()).into()
    }
}

impl<'a> std::iter::Sum<&'a f16> for f16 {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a f16>,
    {
        iter.fold(0.0f32, |acc, i| acc + i.0.to_f32()).into()
    }
}
