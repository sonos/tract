use tract_data::internal::{tensor0, Approximation};

use crate::frame::reduce::MapReduceKer;

#[derive(Clone, Debug)]
pub struct SSoftMaxL2;

impl MapReduceKer<f32, f32> for SSoftMaxL2 {
    fn name() -> &'static str {
        "generic"
    }

    fn alignment_bytes() -> usize {
        16
    }

    fn alignment_items() -> usize {
        4
    }

    fn nr() -> usize {
        4
    }

    fn map_neutral() -> f32 {
        f32::MIN
    }

    fn reduce_neutral() -> f32 {
        0.
    }

    fn reduce_two(a: f32, b: f32) -> f32 {
        a + b
    }

    fn run(x: &mut [f32], max: f32) -> f32 {
        debug_assert!(x.len() % Self::nr() == 0);
        debug_assert!(x.as_ptr() as usize % Self::alignment_bytes() == 0);
        let mut sum = 0.;
        for v in x.iter_mut() {
            let y = *v - max;
            //            let y = y.exp();
            let y = xnnpack_loop2_exp(y);
            *v = y;
            sum += y;
        }
        sum
    }
}

// https://github.com/google/XNNPACK/blob/3bc4ef01bbdf488556c54584fc2419dd77c39c85/src/f32-raddstoreexpminusmax/scalar-rr2-p5.c.in#L131
// https://github.com/google/XNNPACK/blob/8951decff5114f70bae7cc2e23b732812e73acc7/src/microparams-init.c#L4121
#[inline]
#[allow(dead_code, non_upper_case_globals)]
fn xnnpack_loop2_exp(vx: f32) -> f32 {
    debug_assert!(vx <= 0f32);
    const log2e: f32 = hexf::hexf32!("0x1.715476p+0");
    const magic_bias: f32 = hexf::hexf32!("0x1.8000FEp23");
    const minus_ln2_hi: f32 = hexf::hexf32!("-0x1.62E400p-1");
    const minus_ln2_lo: f32 = hexf::hexf32!("-0x1.7F7D1Cp-20");
    const c5: f32 = hexf::hexf32!("0x1.0F9F9Cp-7");
    const c4: f32 = hexf::hexf32!("0x1.573A1Ap-5");
    const c3: f32 = hexf::hexf32!("0x1.555A80p-3");
    const c2: f32 = hexf::hexf32!("0x1.FFFDC6p-2");
    const c1: f32 = hexf::hexf32!("0x1.FFFFF6p-1");
    const denorm_cutoff: f32 = hexf::hexf32!("-0x1.5D589Ep6");

    // Compute reduced argument n := round(x / log(2)).
    // We do it by adding a large number (magic bias) to the product x * (1/log(2)), which cause rounding of the result
    // to an integer, then subtracing the large number back. The trick with adding large number is valid only within
    // certain bounds (|x| <= 2**22), but that's ok, because inputs outside of [-87.336540, 0.0] underflow expf(x)
    // anyway. We fixup the result for such inputs at the very end of the algorithm.
    // float vn = vx * vlog2e + vmagic_bias;
    let mut vn: f32 = vx * log2e + magic_bias;

    // Create a floating-point number s (scale) such that s == 2**n for inputs which don't cause underflow, i.e.
    // -87.33642 <= x <= 0.0, and -126 <= n <= 0 accordingly.
    // const float vs = uint32_as_float(float_as_uint32(vn) << 23);
    let vs = f32::from_bits(vn.to_bits() << 23);

    // Subtract the large number back to get final n := round(x / log(2)).
    // vn -= vmagic_bias;
    vn -= magic_bias;

    // Compute reduced argument t := x - n * log(2).
    // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
    // float vt = vn * vminus_ln2_hi + vx;
    let mut vt = vn * minus_ln2_hi + vx;
    //vt = vn * vminus_ln2_lo + vt;
    vt = vn * minus_ln2_lo + vt;

    // Compute degree-5 polynomial approximation for exp(t) on [-log(2)/2, log(2)/2].
    let mut vp = c5 * vt + c4;
    vp = vp * vt + c2;
    vp = vp * vt + c2;
    vp = vp * vt + c1;

    // Reconstruct the final f value:
    //   f = s * (1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5)))))
    //     = s + (t * s) * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))))
    //     = s + (t * s) * p
    vt *= vs;
    // float vf = vt * vp + vs;
    let mut vf = vt * vp + vs;

    // For inputs below denormal cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    if vx < denorm_cutoff {
        vf = 0.0;
    }
    vf
}

#[cfg(test)]
proptest::proptest! {
    #[test]
    fn t_xnnpack(x in -100f32..0.) {
        tensor0(xnnpack_loop2_exp(x)).close_enough(&tensor0(x.exp()), Approximation::Approximate).unwrap();
    }
}

// ported from https://github.com/gnuradio/volk/blob/master/kernels/volk/volk_32f_expfast_32f.h
#[inline]
#[allow(dead_code)]
fn very_fast_exp(v: f32) -> f32 {
    const MLN2: f32 = 0.6931471805f32;
    const A: f32 = 8388608.0f32;
    const B: f32 = 1065353216.0f32;
    const C: f32 = 60801.0f32;
    const SLOPE: f32 = A / MLN2;
    const OFFSET: f32 = B - C;
    f32::from_bits(((SLOPE * v) + OFFSET) as u32)
}

#[inline]
#[allow(dead_code)]
fn exp2_p7(v: f32) -> f32 {
    const EXP2P: [f32; 7] = [
        1.535336188319500e-4,
        1.339887440266574e-3,
        9.618437357674640e-3,
        5.550332471162809e-2,
        2.402264791363012e-1,
        6.931472028550421e-1,
        1.000000000000000,
    ];
    let v = v.min(127f32).max(-127f32);

    let ipart = (v + 0.5).floor();
    let fpart = v - ipart;

    // 2^ipart
    let two_pow_ipart = f32::from_bits((((ipart as i32) + 127) as u32) << 23);

    let mut v = EXP2P[0];
    v = v * fpart + EXP2P[1];
    v = v * fpart + EXP2P[2];
    v = v * fpart + EXP2P[3];
    v = v * fpart + EXP2P[4];
    v = v * fpart + EXP2P[5];
    v = v * fpart + EXP2P[6];
    v = v * two_pow_ipart;
    v
}

#[cfg(test)]
#[macro_use]
pub mod s {
    softmax_l2_frame_tests!(true, f32, crate::generic::softmax::SSoftMaxL2);
}
