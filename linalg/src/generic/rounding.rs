use crate::frame::mmm::*;
use std::hash::{Hash, Hasher};
use std::ops::Mul;

#[derive(Debug, Clone, Copy)]
pub struct Scaler {
    scale: f32,
    mult: Option<i32>,
    shift: isize,
    policy: RoundingPolicy,
}

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

    // FIXME: Only to avoid fused op breaking
    pub fn to_fuse_params(&self) -> (i32, RoundingPolicy, isize) {
        if let Some(multiplier) = self.mult {
            (multiplier, self.policy, self.shift)
        } else {
            // We return 0.5 in Q0_31 and mutliply be 2 in the shift
            // which is equivalent to 1.
            (1 << 30, self.policy, self.shift - 1)
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
        // Convert f32 to bits representation with the following pattern
        // Bit |  31  |  30-23   |   22-0    |
        //     | Sign | Exponent |  Fraction |
        let scale_bits = scale.to_bits();

        // Get actual value of the exponent
        let current_exponent = scale_bits >> 23;

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

impl Mul<f32> for Scaler {
    type Output = f32;

    #[inline]
    fn mul(self, rhs: f32) -> Self::Output {
        self.scale * rhs
    }
}

impl Mul<Scaler> for f32 {
    type Output = f32;

    #[inline]
    fn mul(self, rhs: Scaler) -> Self::Output {
        rhs.scale * self
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
        let (val, shift) = if let Some(multiplier) = rhs.mult {
            (multiplier as i64 * self as i64, rhs.shift + 31)
        } else {
            (self as i64, rhs.shift)
        };

        // Round according to rounding policy
        use RoundingPolicy::*;
        if shift > 0 {
            let half: i64 = 1 << (shift - 1);
            let nudge: i64 = match rhs.policy {
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

pub trait ScaleShiftAndRound {
    fn q_scale(self, scaler: Scaler) -> Self;
}

impl ScaleShiftAndRound for f32 {
    fn q_scale(self, scaler: Scaler) -> Self {
        self * scaler
    }
}

impl ScaleShiftAndRound for i32 {
    fn q_scale(self, scaler: Scaler) -> Self {
        self * scaler
    }
}

#[cfg(test)]
mod test {
    use super::RoundingPolicy::*;
    use super::*;

    #[test]
    fn test_f32() {
        assert_eq!(0f32.q_scale(Scaler::new(0.5, Zero)), 0.0);
        assert_eq!(1f32.q_scale(Scaler::new(0.5, Zero)), 0.5);
        assert_eq!(2f32.q_scale(Scaler::new(0.5, Zero)), 1.0);
        assert_eq!(3f32.q_scale(Scaler::new(0.5, Zero)), 1.5);
        assert_eq!(-1f32.q_scale(Scaler::new(0.5, Zero)), -0.5);
        assert_eq!(-2f32.q_scale(Scaler::new(0.5, Zero)), -1.0);
        assert_eq!(-3f32.q_scale(Scaler::new(0.5, Zero)), -1.5);
    }

    #[test]
    fn test_zero() {
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
    fn test_away() {
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
    fn test_plus_inf() {
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
    fn test_minus_inf() {
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
    fn test_even() {
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
    fn test_odd() {
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
