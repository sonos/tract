//! Integer ULP (unit in the last place) distance between floating point values.
//!
//! Absolute and relative tolerances answer "is the result roughly right?". ULP
//! distance answers a sharper question: "how many representable floats apart are
//! these two results?". That is the useful metric when two implementations of the
//! same kernel are supposed to be equivalent, because it stays meaningful across
//! the whole dynamic range and cleanly separates a one-off rounding difference
//! from a genuinely different computation.
//!
//! The distance follows the usual total-ordering convention: adjacent floats are
//! 1 apart, `+0.0` and `-0.0` are 1 apart (they are distinct representations),
//! two NaNs are 0 apart, and a NaN against anything else is [`UlpFloat::MAX_ULP`].

use half::f16;

/// Floating point types for which an integer ULP distance is defined.
pub trait UlpFloat: Copy {
    /// Largest distance representable for this type. Also the distance reported
    /// between a NaN and a non-NaN value.
    const MAX_ULP: u64;

    /// Bit pattern of the sign bit, widened to `u64`.
    const SIGN_MASK: u64;

    /// Raw bit pattern, widened to `u64`.
    ///
    /// Within a single sign, the bit patterns of finite floats are monotonic in
    /// magnitude, which is what makes the subtraction below meaningful.
    fn ulp_bits(self) -> u64;

    fn ulp_is_nan(self) -> bool;

    /// Bit pattern with the sign bit cleared, i.e. the bits of `|self|`.
    #[inline]
    fn ulp_magnitude_bits(self) -> u64 {
        self.ulp_bits() & !Self::SIGN_MASK
    }

    #[inline]
    fn ulp_is_sign_negative(self) -> bool {
        self.ulp_bits() & Self::SIGN_MASK != 0
    }
}

macro_rules! impl_ulp_float {
    ($t:ty, $bits:ty) => {
        impl UlpFloat for $t {
            const MAX_ULP: u64 = <$bits>::MAX as u64;
            const SIGN_MASK: u64 = 1 << (<$bits>::BITS - 1);

            #[inline]
            fn ulp_bits(self) -> u64 {
                self.to_bits() as u64
            }

            #[inline]
            fn ulp_is_nan(self) -> bool {
                <$t>::is_nan(self)
            }
        }
    };
}

impl_ulp_float!(f16, u16);
impl_ulp_float!(f32, u32);
impl_ulp_float!(f64, u64);

/// Integer ULP distance between two floats of the same type.
pub fn ulp_distance<T: UlpFloat>(a: T, b: T) -> u64 {
    let (a_nan, b_nan) = (a.ulp_is_nan(), b.ulp_is_nan());
    if a_nan && b_nan {
        return 0;
    }
    if a_nan || b_nan {
        return T::MAX_ULP;
    }
    if a.ulp_is_sign_negative() != b.ulp_is_sign_negative() {
        // The two values sit on opposite sides of zero, so their bit patterns are
        // not comparable directly. Measure each against its own zero, then add one
        // to bridge the gap between -0.0 and +0.0.
        return a
            .ulp_magnitude_bits()
            .saturating_add(b.ulp_magnitude_bits())
            .saturating_add(1)
            .min(T::MAX_ULP);
    }
    a.ulp_bits().abs_diff(b.ulp_bits())
}

/// Largest ULP distance over two sequences, with the index where it occurs.
///
/// Iteration stops at the shorter of the two; callers are expected to have
/// checked the shapes already. Returns `(0, None)` when nothing was compared.
pub fn max_ulp_distance<T: UlpFloat>(
    a: impl IntoIterator<Item = T>,
    b: impl IntoIterator<Item = T>,
) -> (u64, Option<usize>) {
    let mut worst = 0;
    let mut at = None;
    for (ix, (x, y)) in a.into_iter().zip(b).enumerate() {
        let d = ulp_distance(x, y);
        if d > worst || at.is_none() {
            worst = d;
            at = Some(ix);
        }
    }
    (worst, at)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adjacent_floats_are_one_ulp_apart() {
        assert_eq!(ulp_distance(1.0f32, f32::from_bits(1.0f32.to_bits() + 1)), 1);
        assert_eq!(ulp_distance(1.0f64, f64::from_bits(1.0f64.to_bits() + 1)), 1);
        assert_eq!(
            ulp_distance(f16::from_f32(1.0), f16::from_bits(f16::from_f32(1.0).to_bits() + 1)),
            1
        );
    }

    #[test]
    fn identical_values_are_zero_ulp_apart() {
        assert_eq!(ulp_distance(0.0f32, 0.0f32), 0);
        assert_eq!(ulp_distance(-3.25f32, -3.25f32), 0);
        assert_eq!(ulp_distance(f32::INFINITY, f32::INFINITY), 0);
    }

    #[test]
    fn distance_is_symmetric_and_sign_aware() {
        assert_eq!(ulp_distance(1.0f32, -1.0f32), ulp_distance(-1.0f32, 1.0f32));
        // Signed zeros are distinct representations, hence 1 apart.
        assert_eq!(ulp_distance(0.0f32, -0.0f32), 1);
        // Straddling zero costs the two magnitudes plus the zero crossing.
        let tiny = f32::from_bits(1);
        assert_eq!(ulp_distance(tiny, -tiny), 3);
    }

    #[test]
    fn nan_handling() {
        assert_eq!(ulp_distance(f32::NAN, f32::NAN), 0);
        assert_eq!(ulp_distance(f32::NAN, 1.0f32), f32::MAX_ULP);
        assert_eq!(ulp_distance(1.0f32, f32::NAN), f32::MAX_ULP);
    }

    #[test]
    fn infinity_is_adjacent_to_max_finite() {
        assert_eq!(ulp_distance(f32::MAX, f32::INFINITY), 1);
        assert_eq!(ulp_distance(-f32::MAX, f32::NEG_INFINITY), 1);
    }

    #[test]
    fn magnitude_independence() {
        // The same "one rounding step" error reads as 1 ULP at any scale, which is
        // the whole point of the metric.
        for scale in [1e-30f32, 1.0, 1e30] {
            assert_eq!(ulp_distance(scale, f32::from_bits(scale.to_bits() + 1)), 1);
        }
    }

    #[test]
    fn max_over_slice_reports_position() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [1.0f32, 2.0, f32::from_bits(3.0f32.to_bits() + 5)];
        assert_eq!(max_ulp_distance(a, b), (5, Some(2)));
        assert_eq!(max_ulp_distance::<f32>([], []), (0, None));
    }
}
