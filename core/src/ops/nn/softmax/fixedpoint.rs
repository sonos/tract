pub use num_traits::{AsPrimitive, PrimInt};
use std::fmt::{Binary, Debug, LowerHex};

use super::math::*;

macro_rules! impl_fixed_point_func_unary {
    ($func_name: ident) => {
        #[allow(dead_code)]
        pub fn $func_name(&self) -> Self {
            Self::from_raw($func_name(self.as_raw()))
        }
    };
}

macro_rules! impl_fixed_point_func_binary {
    ($func_name: ident) => {
        pub fn $func_name(&self, b: Self) -> Self {
            Self::from_raw($func_name(self.as_raw(), b.as_raw()))
        }
    };
}

pub type Q0_31 = FixedPoint<i32, 0>;
pub type Q1_30 = FixedPoint<i32, 1>;
pub type Q2_29 = FixedPoint<i32, 2>;
pub type Q5_26 = FixedPoint<i32, 5>;

#[derive(PartialEq, Eq,PartialOrd, Copy, Clone)]
pub struct FixedPoint<T: PrimInt, const INTEGER_BITS: usize>(T);

impl<T, const INTEGER_BITS: usize> FixedPoint<T, INTEGER_BITS>
where
    T: PrimInt,
{
    pub fn from_raw(x: T) -> Self {
        Self(x)
    }

    pub fn one() -> Self {
        if INTEGER_BITS == 0 {
            Self(T::max_value())
        } else {
            Self(T::one() << Self::fractional_bits())
        }
    }

    pub fn fractional_bits() -> usize {
        if Self::is_signed() {
            std::mem::size_of::<T>() * 8 - 1 - INTEGER_BITS
        } else {
            std::mem::size_of::<T>() * 8 - INTEGER_BITS
        }
    }

    #[allow(dead_code)]
    pub fn zero() -> Self {
        Self(T::zero())
    }

    pub fn as_raw(&self) -> T {
        self.0
    }

    pub fn is_signed() -> bool {
        is_signed::<T>()
    }
}

impl<T: 'static, const INTEGER_BITS: usize> FixedPoint<T, INTEGER_BITS>
where
    T: PrimInt + Debug,
    usize: AsPrimitive<T>,
{
    pub fn constant_pot(exponent: isize) -> Self {
        let offset = (Self::fractional_bits() as isize + exponent) as usize;
        assert!(offset < 31);
        Self(1_usize.as_() << offset)
    }
}

impl FixedPoint<i32, 0> {
    impl_fixed_point_func_unary!(exp_on_interval_between_negative_one_quarter_and_0_excl);
    impl_fixed_point_func_unary!(one_over_one_plus_x_for_x_in_0_1);
}

impl FixedPoint<i32, 5> {
    #[allow(dead_code)]
    pub fn exp_on_negative_values(&self) -> FixedPoint<i32, 0> {
        FixedPoint::<i32, 0>::from_raw(exp_on_negative_values(self.as_raw()))
    }
}

impl<const INTEGER_BITS: usize> FixedPoint<i32, INTEGER_BITS> {
    impl_fixed_point_func_unary!(mask_if_non_zero);
    impl_fixed_point_func_unary!(mask_if_zero);
    impl_fixed_point_func_binary!(rounding_half_sum);

    pub fn saturating_rounding_multiply_by_pot(&self, exponent: i32) -> Self {
        Self::from_raw(saturating_rounding_multiply_by_pot(self.as_raw(), exponent))
    }

    #[allow(dead_code)]
    pub fn rounding_divide_by_pot(&self, exponent: i32) -> Self {
        Self::from_raw(rounding_divide_by_pot(self.as_raw(), exponent))
    }

    pub fn select_using_mask(mask: i32, a: Self, b: Self) -> Self {
        Self::from_raw(select_using_mask(mask, a.as_raw(), b.as_raw()))
    }

    pub fn rescale<const DST_INTEGER_BITS: usize>(&self) -> FixedPoint<i32, DST_INTEGER_BITS> {
        FixedPoint::<i32, DST_INTEGER_BITS>::from_raw(rescale(
            self.as_raw(),
            INTEGER_BITS,
            DST_INTEGER_BITS,
        ))
    }

    #[allow(dead_code)]
    pub fn get_reciprocal(&self) -> (FixedPoint<i32, 0>, usize) {
        let (raw_res, num_bits_over_units) = get_reciprocal(self.as_raw(), INTEGER_BITS);
        (FixedPoint::<i32, 0>::from_raw(raw_res), num_bits_over_units)
    }
}

impl<T, const INTEGER_BITS: usize> Debug for FixedPoint<T, INTEGER_BITS>
where
    T: AsPrimitive<f32> + PrimInt + LowerHex + Debug + Binary,
    f32: AsPrimitive<T>,
{
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(fmt, "{:032b}({:?})({})", self.0, self.0, self.as_f32())
    }
}

impl<T, const INTEGER_BITS: usize> FixedPoint<T, INTEGER_BITS>
where
    T: AsPrimitive<f32> + PrimInt,
{
    pub fn as_f32(&self) -> f32 {
        self.0.as_() / 2_f32.powi(Self::fractional_bits() as i32)
    }
}

impl<T, const INTEGER_BITS: usize> FixedPoint<T, INTEGER_BITS>
where
    T: AsPrimitive<f32> + PrimInt,
    f32: AsPrimitive<T>,
{
    #[allow(dead_code)]
    pub fn from_f32(x: f32) -> Self {
        Self::from_raw(
            f32::min(
                f32::max(
                    f32::round(x * 2f32.powi(Self::fractional_bits().as_())),
                    T::min_value().as_(),
                ),
                T::max_value().as_(),
            )
            .as_(),
        )
    }
}

impl<T: PrimInt, const INTEGER_BITS: usize> std::ops::Add for FixedPoint<T, INTEGER_BITS> {
    type Output = FixedPoint<T, INTEGER_BITS>;
    fn add(self, rhs: Self) -> Self::Output {
        Self::from_raw(self.0 + rhs.0)
    }
}

impl<T: PrimInt, const INTEGER_BITS: usize> std::ops::Sub for FixedPoint<T, INTEGER_BITS> {
    type Output = FixedPoint<T, INTEGER_BITS>;
    fn sub(self, rhs: Self) -> Self::Output {
        Self::from_raw(self.0 - rhs.0)
    }
}

impl<T: PrimInt, const INTEGER_BITS: usize> std::ops::Shl<usize> for FixedPoint<T, INTEGER_BITS> {
    type Output = FixedPoint<T, INTEGER_BITS>;
    fn shl(self, rhs: usize) -> Self::Output {
        Self::from_raw(self.0 << rhs)
    }
}

impl<T: PrimInt, const INTEGER_BITS: usize> std::ops::Shr<usize> for FixedPoint<T, INTEGER_BITS> {
    type Output = FixedPoint<T, INTEGER_BITS>;
    fn shr(self, rhs: usize) -> Self::Output {
        Self::from_raw(self.0 >> rhs)
    }
}

impl<T: PrimInt, const INTEGER_BITS: usize> std::ops::BitAnd for FixedPoint<T, INTEGER_BITS> {
    type Output = FixedPoint<T, INTEGER_BITS>;
    fn bitand(self, rhs: Self) -> Self::Output {
        Self::from_raw(self.0 & rhs.0)
    }
}

macro_rules! impl_mul {
    ($T: ty, $LHS_INTEGER_BITS: literal, $RHS_INTEGER_BITS: literal, $OUT_INTEGER_BITS: literal) => {
        impl std::ops::Mul<FixedPoint<$T, $RHS_INTEGER_BITS>>
            for FixedPoint<$T, $LHS_INTEGER_BITS>
        {
            type Output = FixedPoint<$T, $OUT_INTEGER_BITS>;
            fn mul(self, rhs: FixedPoint<$T, $RHS_INTEGER_BITS>) -> Self::Output {
                Self::Output::from_raw(saturating_rounding_doubling_high_mul(self.0, rhs.0))
            }
        }
    };
}

impl_mul!(i32, 0, 0, 0);
impl_mul!(i32, 0, 2, 2);
impl_mul!(i32, 2, 0, 2);
impl_mul!(i32, 2, 2, 4);
impl_mul!(i32, 5, 5, 10);

#[cfg(test)]
mod test {
    use super::*;
    use approx::assert_abs_diff_eq;
    pub type Q10_21 = FixedPoint<i32, 10>;
    pub type Q12_19 = FixedPoint<i32, 12>;
    pub type Q26_5 = FixedPoint<i32, 26>;
    type Q0_7 = FixedPoint<i8, 0>;

    #[test]
    fn test_to_f32() {
        let x = Q26_5::from_raw(32);
        assert_eq!(x.as_f32(), 1.0);
    }

    #[test]
    fn test_to_f32_1() {
        let x = Q0_7::from_raw(32);
        assert_eq!(x.as_f32(), 0.25);
    }

    #[test]
    fn test_one() {
        let x = Q26_5::one();
        assert_eq!(x, Q26_5::from_raw(32));
    }

    #[test]
    fn test_one_limit() {
        let x = Q0_31::one();
        assert_eq!(x, Q0_31::from_raw(i32::MAX));
    }

    #[test]
    fn test_mul_1() {
        let a = Q5_26::from_f32(8.0); // 00000001
        let b = Q5_26::from_f32(3.0); // 01000000
        let product = a * b;
        let expected = Q10_21::from_f32(24.0);

        assert_eq!(product, expected);
    }

    #[test]
    fn test_add() {
        let a = Q5_26::from_f32(16.0);
        let b = Q5_26::from_f32(5.0);
        let sum = a + b;
        let expected = Q5_26::from_f32(21.0);
        assert_eq!(sum, expected);
    }

    #[test]
    fn test_one_over_one_plus_x_for_x_in_0_1() {
        let a = Q0_31::from_f32(0.75);
        let expected_res = Q0_31::from_f32(1.0 / 1.75);
        let res = a.one_over_one_plus_x_for_x_in_0_1();
        assert_eq!(res.as_f32(), expected_res.as_f32());
    }

    #[test]
    fn test_one_over_one_plus_x_for_x_in_0_1_1() {
        let a = Q0_31::from_f32(0.0);
        let expected_res = Q0_31::from_f32(1.0 / 1.0);
        let res = a.one_over_one_plus_x_for_x_in_0_1();
        assert_eq!(res.as_f32(), expected_res.as_f32());
    }

    #[test]
    fn test_get_reciprocal_1() {
        let a = Q5_26::from_f32(4.5);
        let expected_res = Q0_31::from_f32(1.0 / 4.5);
        let (shifted_res, num_bits_over_unit) = a.get_reciprocal();
        let res = shifted_res.rounding_divide_by_pot(num_bits_over_unit as i32);
        assert_eq!(res.as_f32(), expected_res.as_f32());
        assert_eq!(num_bits_over_unit, 2);
    }

    #[test]
    fn test_get_reciprocal_2() {
        let a = Q5_26::from_f32(4.5);
        let expected_res = Q0_31::from_f32(1.0 / 4.5);
        let (shifted_res, num_bits_over_unit) = a.get_reciprocal();
        let res = shifted_res.rounding_divide_by_pot(num_bits_over_unit as i32);
        assert_eq!(res.as_f32(), expected_res.as_f32());
        assert_eq!(num_bits_over_unit, 2);
    }

    #[test]
    fn test_get_reciprocal_3() {
        let a = Q12_19::from_f32(2.0);
        let expected_res = Q0_31::from_f32(1.0 / 2.0);
        let (shifted_res, num_bits_over_unit) = a.get_reciprocal();
        let res = shifted_res.rounding_divide_by_pot(num_bits_over_unit as i32);
        assert_eq!(res.as_f32(), expected_res.as_f32());
        assert_eq!(num_bits_over_unit, 1);
    }

    #[test]
    fn test_rescale_1() {
        let a = Q0_31::from_f32(0.75);
        let expeted_res = Q12_19::from_f32(0.75);
        let res = a.rescale::<12>();
        assert_eq!(res, expeted_res);
    }

    #[test]
    fn test_exp_on_interval_between_negative_one_quarter_and_0_excl() {
        let a = Q0_31::from_f32(-0.125);
        let expected_res = Q0_31::from_f32((-0.125_f32).exp());
        let res = a.exp_on_interval_between_negative_one_quarter_and_0_excl();
        assert_eq!(res.as_f32(), expected_res.as_f32());
    }

    #[test]
    fn test_exp_on_negative_values_1() {
        let a = Q5_26::from_f32(-0.125);
        let expected_res = Q0_31::from_f32((-0.125_f32).exp());
        let res = a.exp_on_negative_values();
        assert_abs_diff_eq!(res.as_f32(), expected_res.as_f32(), epsilon = 0.00001);
    }

    #[test]
    fn test_exp_on_negative_values_2() {
        let a = Q5_26::from_f32(0.0);
        let expected_res = Q0_31::from_f32((0_f32).exp());
        let res = a.exp_on_negative_values();
        assert_abs_diff_eq!(res.as_f32(), expected_res.as_f32(), epsilon = 0.00001);
    }

    #[test]
    fn test_exp_on_negative_values_3() {
        let a = Q5_26::from_f32(-0.25);
        let expected_res = Q0_31::from_f32((-0.25_f32).exp());
        let res = a.exp_on_negative_values();
        assert_abs_diff_eq!(res.as_f32(), expected_res.as_f32(), epsilon = 0.00001);
    }

    #[test]
    fn test_exp_on_negative_values_4() {
        let a = Q5_26::from_f32(-1.1875_f32);
        let expected_res = Q0_31::from_f32((-1.1875_f32).exp());
        let res = a.exp_on_negative_values();
        assert_abs_diff_eq!(res.as_f32(), expected_res.as_f32(), epsilon = 0.00001);
    }
}
