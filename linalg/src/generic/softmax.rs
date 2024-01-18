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

    fn neutral() -> f32 {
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
            let y = y.exp();
            *v = y;
            sum += y;
        }
        sum
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
