
pub fn relu(x: f32) -> f32 {
    x.max(0f32)
}

pub fn affine(x: f32, alpha: f32, beta: f32) -> f32 {
    alpha * x + beta
}

pub fn leaky_relu(x: f32, alpha: f32) -> f32 {
    if x > 0f32 {
        x
    } else {
        alpha * x
    }
}

pub fn threshold_relu(x: f32, alpha: f32) -> f32 {
    if x >= alpha {
        x
    } else {
        0f32
    }
}

pub fn softsign(x: f32) -> f32 {
    x / (1. + x.abs())
}

pub fn hardswish(x: f32) -> f32 {
    x * 0f32.max(1f32.min((1. / 6.) * x + 0.5))
}

pub fn sigmoid(x: f32) -> f32 {
    ssigmoid(x)
}

pub fn ref_exp2f(x: f32) -> f32 {
    2f32.powf(x)
}

pub fn cm_exp2f(x: f32) -> f32 {
    exp2f(x)
}

fn ssigmoid(x: f32) -> f32 {
    const LOW: f32 = -18.6;
    const HIGH: f32 = -LOW;

    const ALPHA_13: f32 = -4.433153405e-18;
    const ALPHA_11: f32 = 1.169974371e-14;
    const ALPHA_9: f32 = -1.875289645e-11;
    const ALPHA_7: f32 = 4.257889523e-8;
    const ALPHA_5: f32 = 0.00004811817576;
    const ALPHA_3: f32 = 0.008163842030;
    const ALPHA_1: f32 = 0.2499999971;
    const BETA_6: f32 = 3.922935744e-6;
    const BETA_4: f32 = 0.001524872358;
    const BETA_2: f32 = 0.1159886749;
    const BETA_0: f32 = 1.0;

    let x = x.clamp(LOW, HIGH);

    let x2 = x * x;

    let p = ALPHA_13;
    let p = x2 * p + ALPHA_11;
    let p = x2 * p + ALPHA_9;
    let p = x2 * p + ALPHA_7;
    let p = x2 * p + ALPHA_5;
    let p = x2 * p + ALPHA_3;
    let p = x2 * p + ALPHA_1;
    let p = p * x;

    let q = BETA_6;
    let q = x2 * q + BETA_4;
    let q = x2 * q + BETA_2;
    let q = x2 * q + BETA_0;

    p / q + 0.5
}

pub fn exp2f(x: f32) -> f32 {
    const EXP2P: [f32; 7] = [
        1.535336188319500e-4,
        1.339887440266574e-3,
        9.618437357674640e-3,
        5.550332471162809e-2,
        2.402264791363012e-1,
        6.931472028550421e-1,
        1.000000000000000,
    ];

    let x = x.min(127f32).max(-127f32);

    let ipart = (x + 0.5).floor();
    let fpart = x - ipart;

    // 2^ipart
    let two_pow_ipart = f32::from_bits((((ipart as i32) + 127) as u32) << 23);

    let mut y = EXP2P[0];
    y = y * fpart + EXP2P[1];
    y = y * fpart + EXP2P[2];
    y = y * fpart + EXP2P[3];
    y = y * fpart + EXP2P[4];
    y = y * fpart + EXP2P[5];
    y = y * fpart + EXP2P[6];
    y * two_pow_ipart
}
