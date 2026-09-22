#![allow(clippy::excessive_precision)]

/// Coefficients of the degree-6 minimax fit of `e^r` over `[-ln 2 / 2, ln 2 / 2]`, highest
/// power first.
///
/// Shared by every exp kernel whatever its width, like [`crate::generic::ln::POLY`]. The
/// softmax kernels run the same reduction and fit over a domain of their own, where the
/// argument is never positive and everything under `-103` is zero; nothing here may assume
/// either.
pub const POLY: [f32; 7] = [
    1.383684405e-03,
    8.374815793e-03,
    4.166822560e-02,
    1.666642017e-01,
    4.999999208e-01,
    1.000000036e+00,
    1.000000001e+00,
];

pub const LOG2E: f32 = 1.442_695_04;

/// `ln 2`, split so that `k * LN2_HI` is exact for every `k` the reduction produces and
/// `LN2_LO` holds what it drops, leaving `|r|` under half an ulp of `ln 2 / 2`.
pub const LN2_HI: f32 = 0.693_145_75;
pub const LN2_LO: f32 = 1.428_606_8e-6;

/// The input clamp. Above `HIGH` every result overflows f32 and below `LOW` every result
/// rounds to zero, so clamping there is what the answer already is -- and it holds `k`
/// inside the range [`SCALE_BIAS`] can build from a pair of valid exponent fields.
pub const LOW: f32 = -104.0;
pub const HIGH: f32 = 89.0;

/// What each half of `2^k` adds to its exponent field. `k` is halved and rebuilt as two
/// factors because a single `2^k` has no representation once `k` leaves `[-126, 127]`,
/// which the subnormal and overflowing tails both need.
pub const SCALE_BIAS: i32 = 127;

/// The Cody-Waite rounding constant: `x * LOG2E + MAGIC - MAGIC` is that product rounded
/// to an integer, without an integer conversion the polynomial's lane would have to leave.
pub const MAGIC: f32 = 12_582_912.0;

/// f32 exponential: a Cody-Waite reduction to `e^r * 2^k`, the [`POLY`] fit for `e^r`, and
/// `2^k` rebuilt from a pair of exponent fields.
///
/// Within two ulp of a correctly rounded `exp` over `[LOW, HIGH]`, subnormal results
/// included. Outside the clamp the answer is what the clamp gives: `+inf` above, zero
/// below, so `±inf` answer `+inf` and zero. NaN propagates.
pub fn sexp(x: f32) -> f32 {
    let x = x.clamp(LOW, HIGH);
    let kf = (x * LOG2E + MAGIC) - MAGIC;
    let r = kf.mul_add(-LN2_LO, kf.mul_add(-LN2_HI, x));
    let mut q = POLY[0];
    for c in &POLY[1..] {
        q = q.mul_add(r, *c);
    }
    let k = kf as i32;
    let low = k >> 1;
    let high = k - low;
    let scale_low = f32::from_bits(((low + SCALE_BIAS) as u32) << 23);
    let scale_high = f32::from_bits(((high + SCALE_BIAS) as u32) << 23);
    q * scale_low * scale_high
}

routine_ew_rust!(generic;
    f32,
    generic_exp_f32_4n,
    4,
    4,
    fn run(x: &mut [f32], _: ()) {
        debug_assert!(x.len() % Self::nr() == 0);
        debug_assert!(x.as_ptr() as usize % Self::alignment_bytes() == 0);
        x.iter_mut().for_each(|px| *px = sexp(*px))
    },
    func(Exp)
);
