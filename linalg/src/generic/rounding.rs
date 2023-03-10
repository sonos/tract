use crate::frame::mmm::*;
use std::hash::{Hash, Hasher};
use std::ops::Mul;
use tract_data::prelude::f16;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Scaler {
    pub scale: f32,
    pub mult: Option<i32>,
    pub shift: isize,
    pub policy: RoundingPolicy,
}

impl Eq for Scaler {}

#[allow(clippy::derived_hash_with_manual_eq)]
impl Hash for Scaler {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        Hash::hash(&self.scale.to_bits(), state)
    }
}

impl Scaler {
    pub fn new(scale: f32, policy: RoundingPolicy) -> Self {
        let (mult, shift) = Self::convert_scale_to_mult_shift(scale);
        Self { scale, mult, shift, policy }
    }

    pub fn as_fused_spec(&self) -> FusedSpec {
        if let Some(multiplier) = self.mult {
            FusedSpec::QScale(self.shift, self.policy, multiplier)
        } else if self.shift > 0 {
            FusedSpec::RoundingShiftRight(self.shift as usize, self.policy)
        } else {
            FusedSpec::ShiftLeft((-self.shift) as usize)
        }
    }

    // FIXME: Only to avoid fused op breaking
    pub fn from_fuse_params(shift: isize, policy: RoundingPolicy, mult: i32) -> Self {
        let scale = mult as f32 * 2f32.powi(-(31 + shift as i32));
        Self { scale, mult: Some(mult), shift, policy }
    }

    #[inline]
    // This function convert a scale (actually a fraction of two integers Q/D)
    // into an integer multiplier and a shift (the multiplier being 1/2D in Q0_31).
    fn convert_scale_to_mult_shift(scale: f32) -> (Option<i32>, isize) {
        // Zero is a special case to handle
        if scale == 0.0 {
            return (None, 0);
        }

        // Convert f32 to bits representation with the following pattern
        // Bit |  31  |  30-23   |   22-0    |
        //     | Sign | Exponent |  Fraction |
        let scale_bits = scale.to_bits();

        // Get actual value of the exponent
        let current_exponent = (scale_bits >> 23) & 0xff;

        // Extract fractional part of the float with:
        // - 0x007fffff that represents the mask of the 23 lower bits (fractional part)
        // (partial because it doesn't include the hidden bit (24) of the float representation)
        let partial_frac = scale_bits & 0x007fffff;

        if partial_frac == 0 {
            let shift = 127 - current_exponent as isize;
            (None, shift)
        } else {
            // We add 0x800000 that represents the hidden bit set to one.
            // Here the frac is encoded as a Q8_23.
            let frac = partial_frac | 0x800000;

            // We rescale the result to be in Q0_31
            // We should have shifted the result by 8 but the frac value is in [1.0, 2.0)
            // so we cannot do that (we would need one bit for the integer).
            // Instead we devide the frac by two to be in [0.5, 1.0) in Q0_31
            // which lead to a shift of (8-1 = 7).
            let half_frac = (frac << 7) as i32;

            // Compute the actual value of the shift
            // Here, we remove one as half_frac needs to be multiplied by 2.
            let shift = 127 - current_exponent as isize - 1;
            (Some(half_frac), shift)
        }
    }
}

impl Mul<f16> for Scaler {
    type Output = f16;

    #[inline]
    fn mul(self, rhs: f16) -> Self::Output {
        f16::from_f32(self.scale) * rhs
    }
}

impl Mul<f32> for Scaler {
    type Output = f32;

    #[inline]
    fn mul(self, rhs: f32) -> Self::Output {
        self.scale * rhs
    }
}

impl Mul<f64> for Scaler {
    type Output = f64;

    #[inline]
    fn mul(self, rhs: f64) -> Self::Output {
        self.scale as f64 * rhs
    }
}

impl Mul<Scaler> for f16 {
    type Output = f16;

    #[inline]
    fn mul(self, rhs: Scaler) -> Self::Output {
        rhs * self
    }
}

impl Mul<Scaler> for f32 {
    type Output = f32;

    #[inline]
    fn mul(self, rhs: Scaler) -> Self::Output {
        rhs * self
    }
}

impl Mul<Scaler> for f64 {
    type Output = f64;

    #[inline]
    fn mul(self, rhs: Scaler) -> Self::Output {
        rhs * self
    }
}

impl Mul<i32> for Scaler {
    type Output = i32;

    #[inline]
    fn mul(self, rhs: i32) -> Self::Output {
        let (val, shift) = if let Some(multiplier) = self.mult {
            (multiplier as i64 * rhs as i64, self.shift + 31)
        } else {
            (rhs as i64, self.shift)
        };

        // Round according to rounding policy
        use RoundingPolicy::*;
        if shift > 0 {
            let half: i64 = 1 << (shift - 1);
            let nudge: i64 = match self.policy {
                Zero => -1,
                MinusInf => -((val >= 0) as i64),
                PlusInf => -((val <= 0) as i64),
                Away => 0,
                Even => ((val.abs() >> shift) & 0x1) - 1,
                Odd => -((val.abs() >> shift) & 0x1),
                _ => panic!(),
            };

            (val.signum() * ((val.abs() + half + nudge) >> shift)) as i32
        } else {
            (val << -shift) as i32
        }
    }
}

impl Mul<Scaler> for i32 {
    type Output = i32;

    #[inline]
    fn mul(self, rhs: Scaler) -> Self::Output {
        rhs * self
    }
}

pub trait ScaleShiftAndRound {
    fn q_scale(self, scaler: Scaler) -> Self;
    fn q_shl(self, shift: usize) -> Self;
    fn q_shr(self, shift: usize, rp: RoundingPolicy) -> Self;
}

impl ScaleShiftAndRound for f64 {
    fn q_scale(self, scaler: Scaler) -> Self {
        self * scaler
    }
    fn q_shl(self, shift: usize) -> Self {
        self * 2f64.powi(shift as i32)
    }
    fn q_shr(self, shift: usize, _rp: RoundingPolicy) -> Self {
        self * 2f64.powi(-(shift as i32))
    }
}

impl ScaleShiftAndRound for f32 {
    fn q_scale(self, scaler: Scaler) -> Self {
        self * scaler
    }
    fn q_shl(self, shift: usize) -> Self {
        self * 2f32.powi(shift as i32)
    }
    fn q_shr(self, shift: usize, _rp: RoundingPolicy) -> Self {
        self * 2f32.powi(-(shift as i32))
    }
}

impl ScaleShiftAndRound for f16 {
    fn q_scale(self, scaler: Scaler) -> Self {
        self * scaler
    }
    fn q_shl(self, shift: usize) -> Self {
        self * f16::from_f32(2f32.powi(shift as i32))
    }
    fn q_shr(self, shift: usize, _rp: RoundingPolicy) -> Self {
        self * f16::from_f32(2f32.powi(-(shift as i32)))
    }
}

impl ScaleShiftAndRound for i32 {
    fn q_scale(self, scaler: Scaler) -> Self {
        self * scaler
    }
    fn q_shr(self, shift: usize, rp: RoundingPolicy) -> Self {
        use RoundingPolicy::*;
        let half: i32 = 1 << (shift - 1);
        let nudge: i32 = match rp {
            Zero => -1,
            MinusInf => -((self >= 0) as i32),
            PlusInf => -((self <= 0) as i32),
            Away => 0,
            Even => ((self.abs() >> shift) & 0x1) - 1,
            Odd => -((self.abs() >> shift) & 0x1),
            _ => panic!(),
        };
        self.signum() * ((self.abs() + half + nudge) >> shift)
    }
    fn q_shl(self, shift: usize) -> Self {
        self << shift
    }
}

// 6 / 4 -> 1.5 -> arrondi: 2.  rien a faire
// 2 / 4 -> 0.5 -> arrondi: 1. veut 0 -> nudge = -1

#[cfg(test)]
mod test {
    use super::RoundingPolicy::*;
    use super::*;

    #[test]
    fn test_scale_rounding_f32() {
        assert_eq!(0f32.q_scale(Scaler::new(0.5, Zero)), 0.0);
        assert_eq!(1f32.q_scale(Scaler::new(0.5, Zero)), 0.5);
        assert_eq!(2f32.q_scale(Scaler::new(0.5, Zero)), 1.0);
        assert_eq!(3f32.q_scale(Scaler::new(0.5, Zero)), 1.5);
        assert_eq!((-1f32).q_scale(Scaler::new(0.5, Zero)), -0.5);
        assert_eq!((-2f32).q_scale(Scaler::new(0.5, Zero)), -1.0);
        assert_eq!((-3f32).q_scale(Scaler::new(0.5, Zero)), -1.5);
    }

    #[test]
    fn test_shift_rounding_zero() {
        assert_eq!(0i32.q_shr(1, Zero), 0);
        assert_eq!(1i32.q_shr(1, Zero), 0);
        assert_eq!(2i32.q_shr(1, Zero), 1);
        assert_eq!(3i32.q_shr(1, Zero), 1);
        assert_eq!(0i32.q_shr(2, Zero), 0);
        assert_eq!(1i32.q_shr(2, Zero), 0);
        assert_eq!(2i32.q_shr(2, Zero), 0);
        assert_eq!(3i32.q_shr(2, Zero), 1);
        assert_eq!(4i32.q_shr(2, Zero), 1);
        assert_eq!(5i32.q_shr(2, Zero), 1);
        assert_eq!(6i32.q_shr(2, Zero), 1);
        assert_eq!((-1i32).q_shr(2, Zero), 0);
        assert_eq!((-2i32).q_shr(2, Zero), 0);
        assert_eq!((-3i32).q_shr(2, Zero), -1);
        assert_eq!((-4i32).q_shr(2, Zero), -1);
        assert_eq!((-5i32).q_shr(2, Zero), -1);
        assert_eq!((-6i32).q_shr(2, Zero), -1);
    }

    #[test]
    fn test_scale_rounding_zero() {
        assert_eq!(0i32.q_scale(Scaler::new(0.5, Zero)), 0);
        assert_eq!(1i32.q_scale(Scaler::new(0.5, Zero)), 0);
        assert_eq!(2i32.q_scale(Scaler::new(0.5, Zero)), 1);
        assert_eq!(3i32.q_scale(Scaler::new(0.5, Zero)), 1);
        assert_eq!((-1i32).q_scale(Scaler::new(0.5, Zero)), 0);
        assert_eq!((-2i32).q_scale(Scaler::new(0.5, Zero)), -1);
        assert_eq!((-3i32).q_scale(Scaler::new(0.5, Zero)), -1);
        assert_eq!(2i32.q_scale(Scaler::new(0.25, Zero)), 0);
        assert_eq!(3i32.q_scale(Scaler::new(0.25, Zero)), 1);
        assert_eq!(4i32.q_scale(Scaler::new(0.25, Zero)), 1);
        assert_eq!(5i32.q_scale(Scaler::new(0.25, Zero)), 1);
        assert_eq!(6i32.q_scale(Scaler::new(0.25, Zero)), 1);
        assert_eq!((-2i32).q_scale(Scaler::new(0.25, Zero)), 0);
        assert_eq!((-3i32).q_scale(Scaler::new(0.25, Zero)), -1);
        assert_eq!((-4i32).q_scale(Scaler::new(0.25, Zero)), -1);
        assert_eq!((-5i32).q_scale(Scaler::new(0.25, Zero)), -1);
        assert_eq!((-6i32).q_scale(Scaler::new(0.25, Zero)), -1);
    }

    #[test]
    fn test_shift_rounding_away() {
        assert_eq!(0i32.q_shr(1, Away), 0);
        assert_eq!(1i32.q_shr(1, Away), 1);
        assert_eq!(2i32.q_shr(1, Away), 1);
        assert_eq!(3i32.q_shr(1, Away), 2);
        assert_eq!(0i32.q_shr(2, Away), 0);
        assert_eq!(1i32.q_shr(2, Away), 0);
        assert_eq!(2i32.q_shr(2, Away), 1);
        assert_eq!(3i32.q_shr(2, Away), 1);
        assert_eq!(4i32.q_shr(2, Away), 1);
        assert_eq!(5i32.q_shr(2, Away), 1);
        assert_eq!(6i32.q_shr(2, Away), 2);
        assert_eq!((-1i32).q_shr(2, Away), 0);
        assert_eq!((-2i32).q_shr(2, Away), -1);
        assert_eq!((-3i32).q_shr(2, Away), -1);
        assert_eq!((-4i32).q_shr(2, Away), -1);
        assert_eq!((-5i32).q_shr(2, Away), -1);
        assert_eq!((-6i32).q_shr(2, Away), -2);
    }

    #[test]
    fn test_scale_rounding_away() {
        assert_eq!(0i32.q_scale(Scaler::new(0.5, Away)), 0);
        assert_eq!(1i32.q_scale(Scaler::new(0.5, Away)), 1);
        assert_eq!(2i32.q_scale(Scaler::new(0.5, Away)), 1);
        assert_eq!(3i32.q_scale(Scaler::new(0.5, Away)), 2);
        assert_eq!((-1i32).q_scale(Scaler::new(0.5, Away)), -1);
        assert_eq!((-2i32).q_scale(Scaler::new(0.5, Away)), -1);
        assert_eq!((-3i32).q_scale(Scaler::new(0.5, Away)), -2);
        assert_eq!(2i32.q_scale(Scaler::new(0.25, Away)), 1);
        assert_eq!(3i32.q_scale(Scaler::new(0.25, Away)), 1);
        assert_eq!(4i32.q_scale(Scaler::new(0.25, Away)), 1);
        assert_eq!(5i32.q_scale(Scaler::new(0.25, Away)), 1);
        assert_eq!(6i32.q_scale(Scaler::new(0.25, Away)), 2);
        assert_eq!((-2i32).q_scale(Scaler::new(0.25, Away)), -1);
        assert_eq!((-3i32).q_scale(Scaler::new(0.25, Away)), -1);
        assert_eq!((-4i32).q_scale(Scaler::new(0.25, Away)), -1);
        assert_eq!((-5i32).q_scale(Scaler::new(0.25, Away)), -1);
        assert_eq!((-6i32).q_scale(Scaler::new(0.25, Away)), -2);
    }

    #[test]
    fn test_shift_rounding_plus_inf() {
        assert_eq!(0i32.q_shr(1, PlusInf), 0);
        assert_eq!(1i32.q_shr(1, PlusInf), 1);
        assert_eq!(2i32.q_shr(1, PlusInf), 1);
        assert_eq!(3i32.q_shr(1, PlusInf), 2);
        assert_eq!(0i32.q_shr(2, PlusInf), 0);
        assert_eq!(1i32.q_shr(2, PlusInf), 0);
        assert_eq!(2i32.q_shr(2, PlusInf), 1);
        assert_eq!(3i32.q_shr(2, PlusInf), 1);
        assert_eq!(4i32.q_shr(2, PlusInf), 1);
        assert_eq!(5i32.q_shr(2, PlusInf), 1);
        assert_eq!(6i32.q_shr(2, PlusInf), 2);
        assert_eq!((-1i32).q_shr(2, PlusInf), 0);
        assert_eq!((-2i32).q_shr(2, PlusInf), 0);
        assert_eq!((-3i32).q_shr(2, PlusInf), -1);
        assert_eq!((-4i32).q_shr(2, PlusInf), -1);
        assert_eq!((-5i32).q_shr(2, PlusInf), -1);
        assert_eq!((-6i32).q_shr(2, PlusInf), -1);
    }

    #[test]
    fn test_scale_rounding_plus_inf() {
        assert_eq!(0i32.q_scale(Scaler::new(0.5, PlusInf)), 0);
        assert_eq!(1i32.q_scale(Scaler::new(0.5, PlusInf)), 1);
        assert_eq!(2i32.q_scale(Scaler::new(0.5, PlusInf)), 1);
        assert_eq!(3i32.q_scale(Scaler::new(0.5, PlusInf)), 2);
        assert_eq!((-1i32).q_scale(Scaler::new(0.5, PlusInf)), 0);
        assert_eq!((-2i32).q_scale(Scaler::new(0.5, PlusInf)), -1);
        assert_eq!((-3i32).q_scale(Scaler::new(0.5, PlusInf)), -1);
        assert_eq!(2i32.q_scale(Scaler::new(0.25, PlusInf)), 1);
        assert_eq!(3i32.q_scale(Scaler::new(0.25, PlusInf)), 1);
        assert_eq!(4i32.q_scale(Scaler::new(0.25, PlusInf)), 1);
        assert_eq!(5i32.q_scale(Scaler::new(0.25, PlusInf)), 1);
        assert_eq!(6i32.q_scale(Scaler::new(0.25, PlusInf)), 2);
        assert_eq!((-2i32).q_scale(Scaler::new(0.25, PlusInf)), 0);
        assert_eq!((-3i32).q_scale(Scaler::new(0.25, PlusInf)), -1);
        assert_eq!((-4i32).q_scale(Scaler::new(0.25, PlusInf)), -1);
        assert_eq!((-5i32).q_scale(Scaler::new(0.25, PlusInf)), -1);
        assert_eq!((-6i32).q_scale(Scaler::new(0.25, PlusInf)), -1);
    }

    #[test]
    fn test_shift_rounding_minus_inf() {
        assert_eq!(0i32.q_shr(1, MinusInf), 0);
        assert_eq!(1i32.q_shr(1, MinusInf), 0);
        assert_eq!(2i32.q_shr(1, MinusInf), 1);
        assert_eq!(3i32.q_shr(1, MinusInf), 1);
        assert_eq!(0i32.q_shr(2, MinusInf), 0);
        assert_eq!(1i32.q_shr(2, MinusInf), 0);
        assert_eq!(2i32.q_shr(2, MinusInf), 0);
        assert_eq!(3i32.q_shr(2, MinusInf), 1);
        assert_eq!(4i32.q_shr(2, MinusInf), 1);
        assert_eq!(5i32.q_shr(2, MinusInf), 1);
        assert_eq!(6i32.q_shr(2, MinusInf), 1);
        assert_eq!((-1i32).q_shr(2, MinusInf), 0);
        assert_eq!((-2i32).q_shr(2, MinusInf), -1);
        assert_eq!((-3i32).q_shr(2, MinusInf), -1);
        assert_eq!((-4i32).q_shr(2, MinusInf), -1);
        assert_eq!((-5i32).q_shr(2, MinusInf), -1);
        assert_eq!((-6i32).q_shr(2, MinusInf), -2);
    }

    #[test]
    fn test_scale_rounding_minus_inf() {
        assert_eq!(0i32.q_scale(Scaler::new(0.5, MinusInf)), 0);
        assert_eq!(1i32.q_scale(Scaler::new(0.5, MinusInf)), 0);
        assert_eq!(2i32.q_scale(Scaler::new(0.5, MinusInf)), 1);
        assert_eq!(3i32.q_scale(Scaler::new(0.5, MinusInf)), 1);
        assert_eq!((-1i32).q_scale(Scaler::new(0.5, MinusInf)), -1);
        assert_eq!((-2i32).q_scale(Scaler::new(0.5, MinusInf)), -1);
        assert_eq!((-3i32).q_scale(Scaler::new(0.5, MinusInf)), -2);
        assert_eq!(2i32.q_scale(Scaler::new(0.25, MinusInf)), 0);
        assert_eq!(3i32.q_scale(Scaler::new(0.25, MinusInf)), 1);
        assert_eq!(4i32.q_scale(Scaler::new(0.25, MinusInf)), 1);
        assert_eq!(5i32.q_scale(Scaler::new(0.25, MinusInf)), 1);
        assert_eq!(6i32.q_scale(Scaler::new(0.25, MinusInf)), 1);
        assert_eq!((-2i32).q_scale(Scaler::new(0.25, MinusInf)), -1);
        assert_eq!((-3i32).q_scale(Scaler::new(0.25, MinusInf)), -1);
        assert_eq!((-4i32).q_scale(Scaler::new(0.25, MinusInf)), -1);
        assert_eq!((-5i32).q_scale(Scaler::new(0.25, MinusInf)), -1);
        assert_eq!((-6i32).q_scale(Scaler::new(0.25, MinusInf)), -2);
        //assert_eq!((-9i32).q_scale(ONE_OVER_TWO_IN_Q0_30, 5, MinusInf), 0);
    }

    #[test]
    fn test_shift_rounding_even() {
        assert_eq!(0i32.q_shr(1, Even), 0);
        assert_eq!(1i32.q_shr(1, Even), 0);
        assert_eq!(2i32.q_shr(1, Even), 1);
        assert_eq!(3i32.q_shr(1, Even), 2);
        assert_eq!(0i32.q_shr(2, Even), 0);
        assert_eq!(1i32.q_shr(2, Even), 0);
        assert_eq!(2i32.q_shr(2, Even), 0);
        assert_eq!(3i32.q_shr(2, Even), 1);
        assert_eq!(4i32.q_shr(2, Even), 1);
        assert_eq!(5i32.q_shr(2, Even), 1);
        assert_eq!(6i32.q_shr(2, Even), 2);
        assert_eq!((-1i32).q_shr(2, Even), 0);
        assert_eq!((-2i32).q_shr(2, Even), 0);
        assert_eq!((-3i32).q_shr(2, Even), -1);
        assert_eq!((-4i32).q_shr(2, Even), -1);
        assert_eq!((-5i32).q_shr(2, Even), -1);
        assert_eq!((-6i32).q_shr(2, Even), -2);
    }

    #[test]
    fn test_scale_rounding_even() {
        assert_eq!(0i32.q_scale(Scaler::new(0.5, Even)), 0);
        assert_eq!(1i32.q_scale(Scaler::new(0.5, Even)), 0);
        assert_eq!(2i32.q_scale(Scaler::new(0.5, Even)), 1);
        assert_eq!(3i32.q_scale(Scaler::new(0.5, Even)), 2);
        assert_eq!((-1i32).q_scale(Scaler::new(0.5, Even)), 0);
        assert_eq!((-2i32).q_scale(Scaler::new(0.5, Even)), -1);
        assert_eq!((-3i32).q_scale(Scaler::new(0.5, Even)), -2);
        assert_eq!(2i32.q_scale(Scaler::new(0.25, Even)), 0);
        assert_eq!(3i32.q_scale(Scaler::new(0.25, Even)), 1);
        assert_eq!(4i32.q_scale(Scaler::new(0.25, Even)), 1);
        assert_eq!(5i32.q_scale(Scaler::new(0.25, Even)), 1);
        assert_eq!(6i32.q_scale(Scaler::new(0.25, Even)), 2);
        assert_eq!((-2i32).q_scale(Scaler::new(0.25, Even)), 0);
        assert_eq!((-3i32).q_scale(Scaler::new(0.25, Even)), -1);
        assert_eq!((-4i32).q_scale(Scaler::new(0.25, Even)), -1);
        assert_eq!((-5i32).q_scale(Scaler::new(0.25, Even)), -1);
        assert_eq!((-6i32).q_scale(Scaler::new(0.25, Even)), -2);
    }

    #[test]
    fn test_shift_rounding_odd() {
        assert_eq!(0i32.q_shr(1, Odd), 0);
        assert_eq!(1i32.q_shr(1, Odd), 1);
        assert_eq!(2i32.q_shr(1, Odd), 1);
        assert_eq!(3i32.q_shr(1, Odd), 1);
        assert_eq!(0i32.q_shr(2, Odd), 0);
        assert_eq!(1i32.q_shr(2, Odd), 0);
        assert_eq!(2i32.q_shr(2, Odd), 1);
        assert_eq!(3i32.q_shr(2, Odd), 1);
        assert_eq!(4i32.q_shr(2, Odd), 1);
        assert_eq!(5i32.q_shr(2, Odd), 1);
        assert_eq!(6i32.q_shr(2, Odd), 1);
        assert_eq!((-1i32).q_shr(2, Odd), 0);
        assert_eq!((-2i32).q_shr(2, Odd), -1);
        assert_eq!((-3i32).q_shr(2, Odd), -1);
        assert_eq!((-4i32).q_shr(2, Odd), -1);
        assert_eq!((-5i32).q_shr(2, Odd), -1);
        assert_eq!((-6i32).q_shr(2, Odd), -1);
    }

    #[test]
    fn test_scale_rounding_odd() {
        assert_eq!(0i32.q_scale(Scaler::new(0.5, Odd)), 0);
        assert_eq!(1i32.q_scale(Scaler::new(0.5, Odd)), 1);
        assert_eq!(2i32.q_scale(Scaler::new(0.5, Odd)), 1);
        assert_eq!(3i32.q_scale(Scaler::new(0.5, Odd)), 1);
        assert_eq!((-1i32).q_scale(Scaler::new(0.5, Odd)), -1);
        assert_eq!((-2i32).q_scale(Scaler::new(0.5, Odd)), -1);
        assert_eq!((-3i32).q_scale(Scaler::new(0.5, Odd)), -1);
        assert_eq!(2i32.q_scale(Scaler::new(0.25, Odd)), 1);
        assert_eq!(3i32.q_scale(Scaler::new(0.25, Odd)), 1);
        assert_eq!(4i32.q_scale(Scaler::new(0.25, Odd)), 1);
        assert_eq!(5i32.q_scale(Scaler::new(0.25, Odd)), 1);
        assert_eq!(6i32.q_scale(Scaler::new(0.25, Odd)), 1);
        assert_eq!((-2i32).q_scale(Scaler::new(0.25, Odd)), -1);
        assert_eq!((-3i32).q_scale(Scaler::new(0.25, Odd)), -1);
        assert_eq!((-4i32).q_scale(Scaler::new(0.25, Odd)), -1);
        assert_eq!((-5i32).q_scale(Scaler::new(0.25, Odd)), -1);
        assert_eq!((-6i32).q_scale(Scaler::new(0.25, Odd)), -1);
    }
}
