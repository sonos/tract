#![allow(clippy::excessive_precision)]

/// Coefficients of the minimax fit of `ln(1 + f)`, less its first two Taylor terms, over
/// `[SPLIT / 2 - 1, SPLIT - 1]`, highest power first.
///
/// Shared by every ln kernel whatever its width: the fit is one decision, and a kernel
/// keeping its own copy of it could come to disagree with the rest silently.
pub const POLY: [f32; 9] = [
    7.0376836292e-2,
    -1.1514610310e-1,
    1.1676998740e-1,
    -1.2420140846e-1,
    1.4249322787e-1,
    -1.6668057665e-1,
    2.0000714765e-1,
    -2.4999993993e-1,
    3.3333331174e-1,
];

/// `ln 2`, split so that `e * LN2_HI` is exact for every exponent an f32 carries and
/// `LN2_LO` holds what it drops. One constant instead costs the mantissa its last bits
/// wherever `|e|` is large.
pub const LN2_HI: f32 = 0.693_359_375;
pub const LN2_LO: f32 = -2.121_944_4e-4;

/// Where the mantissa splits: above it the fit's argument would leave the interval
/// [`POLY`] was fitted over, so a power of two moves into the exponent instead.
pub const SPLIT: f32 = std::f32::consts::SQRT_2;

/// `2^SUBNORMAL_SHIFT`, what a subnormal input is scaled by before its exponent field is
/// read, the exponent then paying the shift back. A subnormal carries no implicit leading
/// one, so its field says nothing about its magnitude until it is normal.
pub const SUBNORMAL_SCALE: f32 = 16_777_216.0;
pub const SUBNORMAL_SHIFT: i32 = 24;

/// f32 natural logarithm: the exponent, plus the [`POLY`] fit of the mantissa reduced to
/// the interval around one where the fit holds.
///
/// Within one ulp of a correctly rounded `ln` over the whole f32 range, subnormals
/// included. The specials are IEEE's: `±0` gives `-inf`, a negative gives NaN, `+inf`
/// gives `+inf`, and a NaN input gives a NaN of the kernel's own rather than its payload
/// back.
pub fn sln(x: f32) -> f32 {
    let subnormal = x < f32::MIN_POSITIVE;
    let scaled = if subnormal { x * SUBNORMAL_SCALE } else { x };
    let bits = scaled.to_bits();
    let mut e = ((bits >> 23) & 0xff) as i32 - 127;
    if subnormal {
        e -= SUBNORMAL_SHIFT;
    }
    let mut m = f32::from_bits((bits & 0x007fffff) | 0x3f800000);
    if m > SPLIT {
        m *= 0.5;
        e += 1;
    }
    let f = m - 1.0;
    let f2 = f * f;
    let mut p = POLY[0];
    for c in &POLY[1..] {
        p = p.mul_add(f, *c);
    }
    let e = e as f32;
    let y = p * f2 * f;
    let y = e.mul_add(LN2_LO, y);
    let y = (-0.5f32).mul_add(f2, y) + f;
    let y = e.mul_add(LN2_HI, y);
    if x <= 0.0 || x.is_nan() {
        if x == 0.0 { f32::NEG_INFINITY } else { f32::NAN }
    } else if x.is_infinite() {
        f32::INFINITY
    } else {
        y
    }
}

routine_ew_rust!(generic;
    f32,
    generic_ln_f32_4n,
    4,
    4,
    fn run(x: &mut [f32], _: ()) {
        debug_assert!(x.len() % Self::nr() == 0);
        debug_assert!(x.as_ptr() as usize % Self::alignment_bytes() == 0);
        x.iter_mut().for_each(|px| *px = sln(*px))
    },
    func(Ln)
);
