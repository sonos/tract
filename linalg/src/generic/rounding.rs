use crate::frame::mmm::*;

pub trait PseudoRightShift {
    fn q_away(self, mult: Self, shift: usize) -> Self;
    fn q_even(self, mult: Self, shift: usize) -> Self;
    fn q_to_plus_inf(self, mult: Self, shift: usize) -> Self;
    fn shift(self, shift: usize, policy: RoundingPolicy) -> Self;
}

impl PseudoRightShift for f32 {
    fn q_even(self, mult: Self, shift: usize) -> Self {
        self * mult * 2f32.powi(-(shift as i32))
    }
    fn q_to_plus_inf(self, mult: Self, shift: usize) -> Self {
        self * mult * 2f32.powi(-(shift as i32))
    }
    fn q_away(self, mult: Self, shift: usize) -> Self {
        self * mult * 2f32.powi(-(shift as i32))
    }
    fn shift(self, shift: usize, _policy: RoundingPolicy) -> Self {
        self * 2f32.powi(-(shift as i32))
    }
}

impl PseudoRightShift for i32 {
    fn q_even(self, mult: Self, shift: usize) -> Self {
        let v = ((self as i64 * mult as i64) >> (30 + shift)) as i32;
        let truncated = v.abs();
        let nudge = ((truncated & 0x3) == 0x3) as usize as i32;
        let pos = (truncated + nudge) >> 1;
        if v.is_negative() {
            -pos
        } else {
            pos
        }
    }
    fn q_to_plus_inf(self, mult: Self, shift: usize) -> Self {
        let v = ((self as i64 * mult as i64) >> (30 + shift)) as i32;
        (v + 1) >> 1
    }
    fn q_away(self, mult: Self, shift: usize) -> Self {
        let v = ((self.abs() as i64 * mult as i64) >> (30 + shift)) as i32;
        ((v + 1) >> 1) * self.signum()
    }
    fn shift(self, shift: usize, policy: RoundingPolicy) -> Self {
        use RoundingPolicy::*;
        if shift == 0 {
            return self;
        }
        match policy {
            Zero => self.signum() * (self.abs() >> shift),
            MinusInf => self >> shift,
            PlusInf => ((self >> (shift - 1)) + 1) >> 1,
            Away => self.signum() * (((self.abs() >> (shift - 1)) + 1) >> 1),
            Even => {
                let signum = self.signum();
                let mut val = self.abs() >> (shift - 1);
                if val & 0x3 == 0x3 {
                    val += 1
                };
                (val >> 1) * signum
            }
            Odd => {
                let signum = self.signum();
                let mut val = self.abs() >> (shift - 1);
                if val & 0x3 == 0x1 {
                    val += 1
                };
                (val >> 1) * signum
            }
            _ => panic!(),
        }
    }
}

#[cfg(test)]
mod test {
    use super::RoundingPolicy::*;
    use super::*;

    #[test]
    fn test_zero() {
        assert_eq!(0i32.shift(1, Zero), 0);
        assert_eq!(1i32.shift(1, Zero), 0);
        assert_eq!(2i32.shift(1, Zero), 1);
        assert_eq!(3i32.shift(1, Zero), 1);
        assert_eq!((-1i32).shift(1, Zero), 0);
        assert_eq!((-2i32).shift(1, Zero), -1);
        assert_eq!((-3i32).shift(1, Zero), -1);
    }

    #[test]
    fn test_away() {
        assert_eq!(0i32.shift(1, Away), 0);
        assert_eq!(1i32.shift(1, Away), 1);
        assert_eq!(2i32.shift(1, Away), 1);
        assert_eq!(3i32.shift(1, Away), 2);
        assert_eq!((-1i32).shift(1, Away), -1);
        assert_eq!((-2i32).shift(1, Away), -1);
        assert_eq!((-3i32).shift(1, Away), -2);
    }

    #[test]
    fn test_plus_inf() {
        assert_eq!(0i32.shift(1, PlusInf), 0);
        assert_eq!(1i32.shift(1, PlusInf), 1);
        assert_eq!(2i32.shift(1, PlusInf), 1);
        assert_eq!(3i32.shift(1, PlusInf), 2);
        assert_eq!((-1i32).shift(1, PlusInf), 0);
        assert_eq!((-2i32).shift(1, PlusInf), -1);
        assert_eq!((-3i32).shift(1, PlusInf), -1);
    }
    
    #[test]
    fn test_minus_inf() {
        assert_eq!(0i32.shift(1, MinusInf), 0);
        assert_eq!(1i32.shift(1, MinusInf), 0);
        assert_eq!(2i32.shift(1, MinusInf), 1);
        assert_eq!(3i32.shift(1, MinusInf), 1);
        assert_eq!((-1i32).shift(1, MinusInf), -1);
        assert_eq!((-2i32).shift(1, MinusInf), -1);
        assert_eq!((-3i32).shift(1, MinusInf), -2);
    }

    #[test]
    fn test_even() {
        assert_eq!(0i32.shift(1, Even), 0);
        assert_eq!(1i32.shift(1, Even), 0);
        assert_eq!(2i32.shift(1, Even), 1);
        assert_eq!(3i32.shift(1, Even), 2);
        assert_eq!((-1i32).shift(1, Even), 0);
        assert_eq!((-2i32).shift(1, Even), -1);
        assert_eq!((-3i32).shift(1, Even), -2);
    }

    #[test]
    fn test_odd() {
        assert_eq!(0i32.shift(1, Odd), 0);
        assert_eq!(1i32.shift(1, Odd), 1);
        assert_eq!(2i32.shift(1, Odd), 1);
        assert_eq!(3i32.shift(1, Odd), 1);
        assert_eq!((-1i32).shift(1, Odd), -1);
        assert_eq!((-2i32).shift(1, Odd), -1);
        assert_eq!((-3i32).shift(1, Odd), -1);
    }
}
