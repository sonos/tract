use crate::frame::mmm::*;

pub trait ScaleShiftAndRound {
    fn q_scale(self, mult: i32, shift: usize, policy: RoundingPolicy) -> Self;
}

impl ScaleShiftAndRound for f32 {
    fn q_scale(self, mult: i32, shift: usize, _policy: RoundingPolicy) -> Self {
        self * mult as f32 * 2. * 2f32.powi(-(shift as i32))
    }
}

impl ScaleShiftAndRound for i32 {
    fn q_scale(self, mult: i32, shift: usize, policy: RoundingPolicy) -> Self {
        use RoundingPolicy::*;
        let val = self as i64 * mult as i64;
        let shift = shift + 31;
        let half: i64 = 1 << (shift - 1);
        let nudge: i64 = match policy {
            Zero => -1,
            MinusInf => -((val >= 0) as i64),
            PlusInf => -((val <= 0) as i64),
            Away => 0,
            Even => ((val.abs() >> shift) & 0x1) - 1,
            Odd => -((val.abs() >> shift) & 0x1),
            _ => panic!(),
        };
        (val.signum() * ((val.abs() + half + nudge) >> shift)) as i32
    }
}

#[cfg(test)]
mod test {
    use super::RoundingPolicy::*;
    use super::*;

    #[test]
    fn test_zero() {
        assert_eq!(0i32.q_scale(2i32.pow(30), 0, Zero), 0);
        assert_eq!(1i32.q_scale(2i32.pow(30), 0, Zero), 0);
        assert_eq!(2i32.q_scale(2i32.pow(30), 0, Zero), 1);
        assert_eq!(3i32.q_scale(2i32.pow(30), 0, Zero), 1);
        assert_eq!((-1i32).q_scale(2i32.pow(30), 0, Zero), 0);
        assert_eq!((-2i32).q_scale(2i32.pow(30), 0, Zero), -1);
        assert_eq!((-3i32).q_scale(2i32.pow(30), 0, Zero), -1);
        assert_eq!(2i32.q_scale(2i32.pow(30), 1, Zero), 0);
        assert_eq!(3i32.q_scale(2i32.pow(30), 1, Zero), 1);
        assert_eq!(4i32.q_scale(2i32.pow(30), 1, Zero), 1);
        assert_eq!(5i32.q_scale(2i32.pow(30), 1, Zero), 1);
        assert_eq!(6i32.q_scale(2i32.pow(30), 1, Zero), 1);
        assert_eq!((-2i32).q_scale(2i32.pow(30), 1, Zero), 0);
        assert_eq!((-3i32).q_scale(2i32.pow(30), 1, Zero), -1);
        assert_eq!((-4i32).q_scale(2i32.pow(30), 1, Zero), -1);
        assert_eq!((-5i32).q_scale(2i32.pow(30), 1, Zero), -1);
        assert_eq!((-6i32).q_scale(2i32.pow(30), 1, Zero), -1);
    }

    #[test]
    fn test_away() {
        assert_eq!(0i32.q_scale(2i32.pow(30), 0, Away), 0);
        assert_eq!(1i32.q_scale(2i32.pow(30), 0, Away), 1);
        assert_eq!(2i32.q_scale(2i32.pow(30), 0, Away), 1);
        assert_eq!(3i32.q_scale(2i32.pow(30), 0, Away), 2);
        assert_eq!((-1i32).q_scale(2i32.pow(30), 0, Away), -1);
        assert_eq!((-2i32).q_scale(2i32.pow(30), 0, Away), -1);
        assert_eq!((-3i32).q_scale(2i32.pow(30), 0, Away), -2);
        assert_eq!(2i32.q_scale(2i32.pow(30), 1, Away), 1);
        assert_eq!(3i32.q_scale(2i32.pow(30), 1, Away), 1);
        assert_eq!(4i32.q_scale(2i32.pow(30), 1, Away), 1);
        assert_eq!(5i32.q_scale(2i32.pow(30), 1, Away), 1);
        assert_eq!(6i32.q_scale(2i32.pow(30), 1, Away), 2);
        assert_eq!((-2i32).q_scale(2i32.pow(30), 1, Away), -1);
        assert_eq!((-3i32).q_scale(2i32.pow(30), 1, Away), -1);
        assert_eq!((-4i32).q_scale(2i32.pow(30), 1, Away), -1);
        assert_eq!((-5i32).q_scale(2i32.pow(30), 1, Away), -1);
        assert_eq!((-6i32).q_scale(2i32.pow(30), 1, Away), -2);
    }

    #[test]
    fn test_plus_inf() {
        assert_eq!(0i32.q_scale(2i32.pow(30), 0, PlusInf), 0);
        assert_eq!(1i32.q_scale(2i32.pow(30), 0, PlusInf), 1);
        assert_eq!(2i32.q_scale(2i32.pow(30), 0, PlusInf), 1);
        assert_eq!(3i32.q_scale(2i32.pow(30), 0, PlusInf), 2);
        assert_eq!((-1i32).q_scale(2i32.pow(30), 0, PlusInf), 0);
        assert_eq!((-2i32).q_scale(2i32.pow(30), 0, PlusInf), -1);
        assert_eq!((-3i32).q_scale(2i32.pow(30), 0, PlusInf), -1);
        assert_eq!(2i32.q_scale(2i32.pow(30), 1, PlusInf), 1);
        assert_eq!(3i32.q_scale(2i32.pow(30), 1, PlusInf), 1);
        assert_eq!(4i32.q_scale(2i32.pow(30), 1, PlusInf), 1);
        assert_eq!(5i32.q_scale(2i32.pow(30), 1, PlusInf), 1);
        assert_eq!(6i32.q_scale(2i32.pow(30), 1, PlusInf), 2);
        assert_eq!((-2i32).q_scale(2i32.pow(30), 1, PlusInf), 0);
        assert_eq!((-3i32).q_scale(2i32.pow(30), 1, PlusInf), -1);
        assert_eq!((-4i32).q_scale(2i32.pow(30), 1, PlusInf), -1);
        assert_eq!((-5i32).q_scale(2i32.pow(30), 1, PlusInf), -1);
        assert_eq!((-6i32).q_scale(2i32.pow(30), 1, PlusInf), -1);
    }

    #[test]
    fn test_minus_inf() {
        assert_eq!(0i32.q_scale(2i32.pow(30), 0, MinusInf), 0);
        assert_eq!(1i32.q_scale(2i32.pow(30), 0, MinusInf), 0);
        assert_eq!(2i32.q_scale(2i32.pow(30), 0, MinusInf), 1);
        assert_eq!(3i32.q_scale(2i32.pow(30), 0, MinusInf), 1);
        assert_eq!((-1i32).q_scale(2i32.pow(30), 0, MinusInf), -1);
        assert_eq!((-2i32).q_scale(2i32.pow(30), 0, MinusInf), -1);
        assert_eq!((-3i32).q_scale(2i32.pow(30), 0, MinusInf), -2);
        assert_eq!(2i32.q_scale(2i32.pow(30), 1, MinusInf), 0);
        assert_eq!(3i32.q_scale(2i32.pow(30), 1, MinusInf), 1);
        assert_eq!(4i32.q_scale(2i32.pow(30), 1, MinusInf), 1);
        assert_eq!(5i32.q_scale(2i32.pow(30), 1, MinusInf), 1);
        assert_eq!(6i32.q_scale(2i32.pow(30), 1, MinusInf), 1);
        assert_eq!((-2i32).q_scale(2i32.pow(30), 1, MinusInf), -1);
        assert_eq!((-3i32).q_scale(2i32.pow(30), 1, MinusInf), -1);
        assert_eq!((-4i32).q_scale(2i32.pow(30), 1, MinusInf), -1);
        assert_eq!((-5i32).q_scale(2i32.pow(30), 1, MinusInf), -1);
        assert_eq!((-6i32).q_scale(2i32.pow(30), 1, MinusInf), -2);
        assert_eq!((-9i32).q_scale(2i32.pow(30), 5, MinusInf), 0);
    }

    #[test]
    fn test_even() {
        assert_eq!(0i32.q_scale(2i32.pow(30), 0, Even), 0);
        assert_eq!(1i32.q_scale(2i32.pow(30), 0, Even), 0);
        assert_eq!(2i32.q_scale(2i32.pow(30), 0, Even), 1);
        assert_eq!(3i32.q_scale(2i32.pow(30), 0, Even), 2);
        assert_eq!((-1i32).q_scale(2i32.pow(30), 0, Even), 0);
        assert_eq!((-2i32).q_scale(2i32.pow(30), 0, Even), -1);
        assert_eq!((-3i32).q_scale(2i32.pow(30), 0, Even), -2);
        assert_eq!(2i32.q_scale(2i32.pow(30), 1, Even), 0);
        assert_eq!(3i32.q_scale(2i32.pow(30), 1, Even), 1);
        assert_eq!(4i32.q_scale(2i32.pow(30), 1, Even), 1);
        assert_eq!(5i32.q_scale(2i32.pow(30), 1, Even), 1);
        assert_eq!(6i32.q_scale(2i32.pow(30), 1, Even), 2);
        assert_eq!((-2i32).q_scale(2i32.pow(30), 1, Even), 0);
        assert_eq!((-3i32).q_scale(2i32.pow(30), 1, Even), -1);
        assert_eq!((-4i32).q_scale(2i32.pow(30), 1, Even), -1);
        assert_eq!((-5i32).q_scale(2i32.pow(30), 1, Even), -1);
        assert_eq!((-6i32).q_scale(2i32.pow(30), 1, Even), -2);
    }

    #[test]
    fn test_odd() {
        assert_eq!(0i32.q_scale(2i32.pow(30), 0, Odd), 0);
        assert_eq!(1i32.q_scale(2i32.pow(30), 0, Odd), 1);
        assert_eq!(2i32.q_scale(2i32.pow(30), 0, Odd), 1);
        assert_eq!(3i32.q_scale(2i32.pow(30), 0, Odd), 1);
        assert_eq!((-1i32).q_scale(2i32.pow(30), 0, Odd), -1);
        assert_eq!((-2i32).q_scale(2i32.pow(30), 0, Odd), -1);
        assert_eq!((-3i32).q_scale(2i32.pow(30), 0, Odd), -1);
        assert_eq!(2i32.q_scale(2i32.pow(30), 1, Odd), 1);
        assert_eq!(3i32.q_scale(2i32.pow(30), 1, Odd), 1);
        assert_eq!(4i32.q_scale(2i32.pow(30), 1, Odd), 1);
        assert_eq!(5i32.q_scale(2i32.pow(30), 1, Odd), 1);
        assert_eq!(6i32.q_scale(2i32.pow(30), 1, Odd), 1);
        assert_eq!((-2i32).q_scale(2i32.pow(30), 1, Odd), -1);
        assert_eq!((-3i32).q_scale(2i32.pow(30), 1, Odd), -1);
        assert_eq!((-4i32).q_scale(2i32.pow(30), 1, Odd), -1);
        assert_eq!((-5i32).q_scale(2i32.pow(30), 1, Odd), -1);
        assert_eq!((-6i32).q_scale(2i32.pow(30), 1, Odd), -1);
    }
}
