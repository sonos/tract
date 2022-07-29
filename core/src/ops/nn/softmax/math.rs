use super::fixedpoint::{Q0_31, Q1_30, Q2_29, Q5_26};
use num_traits::PrimInt;

// This function convert a scale (actually the inverse of an integer 1/D)
// into an integer multiplier and a shift (the multiplier being 1/D in Q0_31).
pub fn convert_scale_to_mult_shift(scale: f32) -> Option<(i32, isize)> {
    if scale <= 0.0 {
        return None;
    }

    let scale_bits = scale.to_bits();

    // We extract the exponent value
    let current_exponent = scale_bits >> 23;

    // We extract the fractional part of the float
    let fractional_part = scale_bits & 0x007fffff;

    if fractional_part == 0 {
        let shift = 127 - current_exponent as isize;
        Some((0, shift))
    } else {
        let bumped_multi = f32::from_bits(fractional_part | 0x3f000000);
        let int_multi = (bumped_multi * (1u32 << 31) as f32).round() as i32;
        let shift = 127 - current_exponent as isize - 1;
        Some((int_multi, shift))
    }
}

// Get inverse of X with a result as Q0.31
// Here we expect X to have at least fixed point = 1 and to be >= 1 -> required to have an output in Q0_31.
// https://github.com/tensorflow/tensorflow/blob/8c6f391a2282684a25cbfec7687bd5d35261a209/tensorflow/lite/kernels/internal/common.h#L765
pub(crate) fn get_reciprocal(x: i32, fixed_point: usize) -> (i32, usize) {
    assert!(fixed_point > 0);
    //assert!(x as f32 / 2_f32.powi((31 - fixed_point) as i32) > 1.0);
    // Sounds like we compute the smallest amount of integer bits needed to represent
    // the integer part of the number.
    let headroom_plus_one = (x as u32).leading_zeros() as usize;
    let num_bits_over_unit = fixed_point - headroom_plus_one;

    let shifted_sum_minus_one = ((x as u32) << headroom_plus_one) - (1_u32 << 31);
    let shifted_scale =
        Q0_31::from_raw(shifted_sum_minus_one as i32).one_over_one_plus_x_for_x_in_0_1();
    (shifted_scale.as_raw(), num_bits_over_unit)
}

// Returns 1 / (1 + x) for x in (0, 1).
// We expect input to be in Q0_31 and output to be in Q0_31.
// https://github.com/google/gemmlowp/blob/e844ffd17118c1e17d94e1ba4354c075a4577b88/fixedpoint/fixedpoint.h#L854
pub fn one_over_one_plus_x_for_x_in_0_1(a: i32) -> i32 {
    let a_in_q0_31 = Q0_31::from_raw(a);
    let constant_48_over_17 = Q2_29::from_raw(1515870810);
    let constant_neg_32_over_17 = Q2_29::from_raw(-1010580540);

    // 1.0 cannot be represented so we set it to 0.99999999
    // which is all bits to 1.
    let one_in_q0_31 = Q0_31::one();
    let one_in_q2_29 = Q2_29::one();

    // We are in Q0.31
    // Let's call D = a + 1 do stick with the Newton–Raphson naming.
    // D is in [1, 2] so we consider D'(half_denominator) = D / 2 is in [0.5, 1]
    let half_denominator = a_in_q0_31.rounding_half_sum(one_in_q0_31);

    // We are in Q2.29 (because Q0.31 * Q2.29 is a Q2.29)
    let x_0: Q2_29 = constant_48_over_17 + half_denominator * constant_neg_32_over_17;

    // The formulae used here is:
    // Xn+1 = Xn + Xn(1 - D' * Xn)
    let mut x_n = x_0;
    for _ in 0..3 {
        // We are in Q2.29 (because Q0.31 * Q2.29 is a Q2.29)
        let half_denominator_times_x_n = half_denominator * x_n;

        // We are in Q2.29 (because Q2.29 - Q2.29 is a Q2.29)
        let one_minus_half_denominator_times_x_n = one_in_q2_29 - half_denominator_times_x_n;
        // We are in Q4.27 !! (because Q2.29 - Q2.29 is a Q4.27)
        let x_times_one_minus_half_denominator_times_x_n =
            x_n * one_minus_half_denominator_times_x_n;

        // We are in Q4.27 so we need to rescale to be in Q2.29
        let rescaled_x_n = x_times_one_minus_half_denominator_times_x_n.rescale::<2>();
        x_n = x_n + rescaled_x_n;
    }

    // We now have a value for 1/D' = 2/D = 2/(a+1)
    // Instead of doing the division by two, we just pretend that
    // this value is in Q1.30
    let half_x_n = Q1_30::from_raw(x_n.as_raw());

    // We rescale in Q0.31
    let res_in_q0_31 = half_x_n.rescale::<0>();
    res_in_q0_31.as_raw()
}

// Rescale changes the number of IntegerBits and updates the underlying raw integer value accordingly
// https://github.com/google/gemmlowp/blob/13d57703abca3005d97b19df1f2db731607a7dc2/fixedpoint/fixedpoint.h#L678
pub(crate) fn rescale(x: i32, src_integer_bits: usize, dst_integer_bits: usize) -> i32 {
    let exponent = src_integer_bits as i32 - dst_integer_bits as i32;
    saturating_rounding_multiply_by_pot(x, exponent)
}

// Here we assume the input to be a Q5.26
pub fn exp_on_negative_values(a: i32) -> i32 {
    let a_q5_36 = Q5_26::from_raw(a);
    let k_one_quarter = Q5_26::constant_pot(-2); // 1/4
    let mask = k_one_quarter - Q5_26::from_raw(1); // 1/4 - 1/2^26 in Q5.26
    let a_mod_quarter_minus_one_quarter = (a_q5_36 & mask) - k_one_quarter;

    // We need to rescale from Q5.26 to Q0.31 as we compute the exp in this range
    let rescaled_a_mod_quarter_minus_one_quarter = a_mod_quarter_minus_one_quarter.rescale::<0>();

    // We are in Q0.31
    let mut result = rescaled_a_mod_quarter_minus_one_quarter
        .exp_on_interval_between_negative_one_quarter_and_0_excl();
    let remainder = (a_mod_quarter_minus_one_quarter - a_q5_36).as_raw();

    macro_rules! exp_barrel_shifter {
        ($exponent: expr, $quantized_value: expr) => {
            if 5 > $exponent {
                // equivalent to one_in_q5_26 << exponent when exponent > 0.
                // but handle the case exponent < 0.
                let k_shift_amount = 26 + $exponent;
                let mask = mask_if_non_zero(remainder & (1 << k_shift_amount));
                result = Q0_31::select_using_mask(
                    mask,
                    result * Q0_31::from_raw($quantized_value),
                    result,
                );
            }
        };
    }

    exp_barrel_shifter!(-2, 1672461947); // exp(-1/4)
    exp_barrel_shifter!(-1, 1302514674); // exp(-1/2)
    exp_barrel_shifter!(0, 790015084); // exp(-1)
    exp_barrel_shifter!(1, 290630308); // exp(-2)
    exp_barrel_shifter!(2, 39332535); // exp(-4)
    exp_barrel_shifter!(3, 720401); // exp(-8)
    exp_barrel_shifter!(4, 242); // exp(-16)

    let mask = a_q5_36.mask_if_zero();
    let res_in_q0_31 = Q0_31::select_using_mask(mask.as_raw(), Q0_31::one(), result);
    res_in_q0_31.as_raw()
}

// Here we assume the input to be a Q0.31
// Valid for x in [-1/4, 0).
pub(crate) fn exp_on_interval_between_negative_one_quarter_and_0_excl(a: i32) -> i32 {
    let a_in_q0_31 = Q0_31::from_raw(a);
    // We are in Q0.31 and this represents exp(-1/8)
    let exp_minus_one_over_eight = Q0_31::from_raw(1895147668);
    // We are in Q0.31 and this represents 1/3
    let constant_1_over_3 = Q0_31::from_raw(715827883);

    // We estimate the value by doing a taylor expansion around -1/8
    // so we do the change of variable x = a + 1/8
    let x = a_in_q0_31 + Q0_31::constant_pot(-3);
    let x2 = x * x;
    let x3 = x2 * x;
    let x4 = x2 * x2;
    let x4_over_4 = x4.saturating_rounding_multiply_by_pot(-2);
    let x4_over_24_plus_x3_over_6_plus_x2_over_2 =
        Q0_31::from_raw(saturating_rounding_multiply_by_pot(
            (((x4_over_4 + x3) * constant_1_over_3) + x2).as_raw(),
            -1,
        ));
    let res_in_q0_31 = exp_minus_one_over_eight
        + exp_minus_one_over_eight * (x + x4_over_24_plus_x3_over_6_plus_x2_over_2);
    res_in_q0_31.as_raw()
}

// Correctly-rounded-to-nearest division by a power-of-two.
// Also known as a rounding arithmetic right shift.
// https://github.com/google/gemmlowp/blob/13d57703abca3005d97b19df1f2db731607a7dc2/fixedpoint/fixedpoint.h#L368
//
// Taken from here:
// https://github.com/ARM-software/CMSIS_5/blob/61a22cd0eac4f22aa19314125f663198736afa3f/CMSIS/NN/Include/arm_nnsupportfunctions.h#L924
pub fn rounding_divide_by_pot(x: i32, exponent: i32) -> i32 {
    // Exponent is a bit shift so it should be in [0,31[
    assert!(exponent >= 0);
    assert!(exponent <= 31);

    let mask = ((1_i64 << exponent) - 1) as i32;
    let remainder = x & mask;

    // Basic division
    let mut result = x >> exponent as usize;

    // Adjust 'result' for rounding (mid point away from zero)
    let mut threshold = mask >> 1;
    if result < 0 {
        threshold += 1;
    }
    if remainder > threshold {
        result += 1;
    }

    result
}

// https://github.com/google/gemmlowp/blob/13d57703abca3005d97b19df1f2db731607a7dc2/fixedpoint/fixedpoint.h#L385
#[allow(clippy::comparison_chain)] // nah
pub fn saturating_rounding_multiply_by_pot(x: i32, exponent: i32) -> i32 {
    if exponent == 0 {
        x
    } else if exponent < 0 {
        rounding_divide_by_pot(x, -exponent)
    } else {
        let min = i32::MIN;
        let max = i32::MAX;
        let threshold = (1 << (32 - 1 - exponent)) - 1;
        let positive_mask = mask_if_non_zero((x > threshold) as i32);
        let negative_mask = mask_if_non_zero((x < -threshold) as i32);
        let mut result = x << exponent as usize;
        result = select_using_mask(positive_mask, max, result);
        result = select_using_mask(negative_mask, min, result);
        result
    }
}

// https://github.com/google/gemmlowp/blob/e844ffd17118c1e17d94e1ba4354c075a4577b88/fixedpoint/fixedpoint.h#L237
pub fn rounding_half_sum(a: i32, b: i32) -> i32 {
    let sum = (a as i64) + (b as i64);
    let sign: i64 = if sum >= 0 { 1 } else { -1 };
    ((sum + sign) / 2) as i32
}

pub fn mask_if_non_zero(x: i32) -> i32 {
    if x != 0 {
        !0
    } else {
        0
    }
}

pub fn mask_if_zero(x: i32) -> i32 {
    if x == 0 {
        !0
    } else {
        0
    }
}

pub fn select_using_mask(mask: i32, a: i32, b: i32) -> i32 {
    (mask & a) ^ (!mask & b)
}

pub fn saturating_rounding_doubling_high_mul(a: i32, b: i32) -> i32 {
    let overflow = a == b && a == i32::MIN;
    let product = (a as i64) * (b as i64);

    let nudge = if product >= 0 { 1 << 30 } else { 1 - (1 << 30) };
    let product_x2_high32 = ((product + nudge) / (1_i64 << 31)) as i32;

    if overflow {
        i32::MAX
    } else {
        product_x2_high32
    }
}

pub fn is_signed<T: PrimInt>() -> bool {
    let mv = T::min_value();
    let z = T::zero();
    mv < z
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_rounding_divide_by_pot_1() {
        let x = 128;
        let res = rounding_divide_by_pot(x, 2);
        assert_eq!(res, 32);
    }

    #[test]
    fn test_rounding_divide_by_pot_2() {
        let x = 129;
        let res = rounding_divide_by_pot(x, 2);
        assert_eq!(res, 32);
    }

    #[test]
    fn test_rounding_half_sum_1() {
        let a = 22;
        let b = 22;
        let res = rounding_half_sum(a, b);
        assert_eq!(res, 22)
    }

    #[test]
    fn test_rounding_half_sum_2() {
        let a = 6; // 0.75 in Q28.3
        let b = 1_i32 << 3; // 1.0 in Q28.3
        let expected_res = 7; // (1.75 / 2 = 0.875) in Q28.3
        let res = rounding_half_sum(a, b);
        assert_eq!(res, expected_res)
    }

    #[test]
    fn test_rounding_half_sum_3() {
        let a = 1610612736; // 0.75 in Q0.31
        let b = i32::MAX; // 1.0 in Q0.31
        let expected_res = 1879048192; // (1.75 / 2 = 0.875) in Q0.31
        let res = rounding_half_sum(a, b);
        assert_eq!(res, expected_res)
    }

    #[test]
    fn test_saturating_rounding_doubling_high_mul() {
        let a: i32 = 1879048192; // (1.75 / 2 = 0.875) in Q0.31
        let b: i32 = -631612838; //
        let expected_res = -552661233;
        let res = saturating_rounding_doubling_high_mul(a, b);
        assert_eq!(res, expected_res);
    }
}
