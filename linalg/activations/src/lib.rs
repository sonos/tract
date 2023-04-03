#[derive(Copy, Clone, Debug, PartialEq)]
pub enum RegisterId {
    A = 0,
    B = 1,
    C = 2,
    D = 3,
}

type ConstantId = usize;

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Op {
    Move(RegisterId, RegisterId),
    Load(RegisterId, ConstantId),
    Abs,
    Recip,
    Add,
    Sub,
    Mul,
    Min,
    Max,
    AddConst(ConstantId),
    SubConst(ConstantId),
    MulConst(ConstantId),
    MinConst(ConstantId),
    MaxConst(ConstantId),
    FMA(ConstantId), // a <- a * b + cst
    IfPosTE,
    SwapBC,
    Floor,
    TwoPowOfInt,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Program {
    ops: Vec<Op>,
    csts: Vec<f32>,
}

impl Program {
    pub fn compute(&self, x: f32) -> f32 {
        let mut regs = [0f32; 4];
        regs[0] = x;
        let mut constants = self.csts.clone();
        constants.insert(0, 0f32);
        constants.insert(1, 1f32);
        for op in &self.ops {
            match op {
                Op::Move(dst, src) => regs[*dst as usize] = regs[*src as usize],
                Op::Load(dst, cst) => regs[*dst as usize] = constants[*cst],
                Op::Abs => regs[0] = regs[0].abs(),
                Op::Recip => regs[0] = regs[0].recip(),
                Op::Add => regs[0] = regs[0] + regs[1],
                Op::Sub => regs[0] = regs[0] - regs[1],
                Op::Mul => regs[0] = regs[0] * regs[1],
                Op::Min => regs[0] = regs[0].min(regs[1]),
                Op::Max => regs[0] = regs[0].max(regs[1]),
                Op::AddConst(cst) => regs[0] = regs[0] + constants[*cst],
                Op::SubConst(cst) => regs[0] = regs[0] - constants[*cst],
                Op::MulConst(cst) => regs[0] = regs[0] * constants[*cst],
                Op::MinConst(cst) => regs[0] = regs[0].min(constants[*cst]),
                Op::MaxConst(cst) => regs[0] = regs[0].max(constants[*cst]),
                Op::IfPosTE => regs[0] = if regs[0] >= 0f32 { regs[1] } else { regs[2] },
                Op::FMA(cst) => regs[0] = regs[0] * regs[1] + constants[*cst],
                Op::SwapBC => regs.swap(1, 2),
                Op::Floor => regs[0] = regs[0].floor(),
                Op::TwoPowOfInt => {
                    regs[0] = f32::from_bits((((regs[0] as i32) + 127) as u32) << 23)
                }
            }
        }
        regs[0]
    }
}

pub mod definitions {
    use super::Op::*;
    use super::RegisterId::*;
    use super::*;

    pub fn relu() -> Program {
        Program { ops: vec![MaxConst(0)], csts: vec![] }
    }

    pub fn affine(alpha: f32, beta: f32) -> Program {
        Program {
            #[rustfmt::skip]
            ops: vec![
                MulConst(2),
                AddConst(3),
            ],
            csts: vec![alpha, beta],
        }
    }

    pub fn leaky_relu(alpha: f32) -> Program {
        Program {
            #[rustfmt::skip]
            ops: vec![
                Move(B,A),
                MulConst(2),
                Move(C,A),
                Move(A,B),
                IfPosTE,
            ],
            csts: vec![alpha],
        }
    }

    pub fn threshold_relu(alpha: f32) -> Program {
        Program {
            #[rustfmt::skip]
            ops: vec![
                Move(B,A),
                SubConst(2),
                Load(C,0),
                IfPosTE,
            ],
            csts: vec![alpha],
        }
    }

    pub fn softsign() -> Program {
        Program {
            #[rustfmt::skip]
            ops: vec![
                Move(B,A),
                Abs,
                AddConst(1),
                Recip,
                Mul,
            ],
            csts: vec![],
        }
    }

    pub fn hardswish() -> Program {
        Program {
            #[rustfmt::skip]
            ops: vec![
                Move(B, A),
                MulConst(2),
                AddConst(3),
                MinConst(1),
                MaxConst(0),
                Mul,
            ],
            csts: vec![1f32 / 6., 0.5],
        }
    }

    pub fn sigmoid() -> Program {
        Program {
            #[rustfmt::skip]
            ops: vec![
                MinConst(3),
                MaxConst(2),
                Move(B, A),  // b = x
                Move(C, A),  // c = x
                Mul,         // a = x2
                Move(B, A),  // b = x2
                MulConst(4),
                AddConst(5), // a = x2 * a13 + a11
                FMA(6),
                FMA(7),
                FMA(8),
                FMA(9),
                FMA(10),
                SwapBC,     // c = x2, b = x
                Mul,        // a = p(x)
                Move(B, C), // b = x2
                Move(C, A), // c = p(x)
                Move(A, B), // a = x2
                MulConst(11),
                AddConst(12),
                FMA(13),
                FMA(1),      // a = q(x)
                Recip,
                Move(B,C),   // b = p(x)
                Mul,
                AddConst(14)
                    ],
            csts: vec![
                -18.6,            // const 2
                18.6,             // const 3
                -4.433153405e-18, // const 4, also alpha_13
                1.169974371e-14,  // const 5, also a11
                -1.875289645e-11,
                4.257889523e-8,
                0.00004811817576, // const 8
                0.008163842030,
                0.2499999971,   // alpha_1
                3.922935744e-6, // beta_6
                0.001524872358, // const 12
                0.1159886749,
                0.5, //beta_0
            ],
        }
    }

    pub fn exp2f() -> Program {
        Program {
            #[rustfmt::skip]
            ops: vec![
                MinConst(2),
                MaxConst(3),
                Move(B, A),     // b = x
                AddConst(4),    // a = x + 0.5
                Floor,          // a = ipart
                Move(C, A),     // c = ipart
                Move(A, B),     // a = x
                Move(B, C),     // b = ipart
                Sub,            // a = fpart
                Move(B, A),     // b = fpart
                Load(A, 5),     // a = exp2p[0]
                FMA(6),
                FMA(7),
                FMA(8),
                FMA(9),
                FMA(10),
                FMA(1),         // a = y
                Move(B, A),
                Move(A, C),
                TwoPowOfInt,
                Mul
                    ],
            csts: vec![
                127f32,
                -127f32,
                0.5,
                1.535336188319500e-4,
                1.339887440266574e-3,
                9.618437357674640e-3,
                5.550332471162809e-2,
                2.402264791363012e-1,
                6.931472028550421e-1,
            ],
        }
    }
}

pub mod reference {
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

    pub fn subsign(x: f32) -> f32 {
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
}

#[cfg(test)]
mod test {
    use proptest::prelude::*;

    fn close_enough(a: f32, b: f32) -> bool {
        fn max(a: f32, b: f32) -> f32 {
            if a < b {
                b
            } else {
                a
            }
        }
        let rtol = 1e-05;
        let atol = 1e-06;
        let result = (a - b).abs() <= max(rtol * max(a.abs(), b.abs()), atol);
        if !result {
            dbg!(a, b);
        }
        return result
    }

    proptest! {
        #[test]
        fn test_relu(x in any::<f32>()) {
            prop_assert_eq!(super::definitions::relu().compute(x), super::reference::relu(x))
        }

        #[test]
        fn test_affine(x in any::<f32>(), alpha in any::<f32>(), beta in any::<f32>()) {
            prop_assert_eq!(super::definitions::affine(alpha, beta).compute(x),
            super::reference::affine(x, alpha, beta))
        }

        #[test]
        fn test_leaky_relu(x in any::<f32>(), alpha in any::<f32>()) {
            prop_assert_eq!(super::definitions::leaky_relu(alpha).compute(x),super::reference::leaky_relu(x, alpha))
        }

        #[test]
        fn test_threshold_relu(x in any::<f32>(), alpha in any::<f32>()) {
            prop_assert_eq!(super::definitions::threshold_relu(alpha).compute(x), super::reference::threshold_relu(x, alpha) );
        }

        #[test]
        fn test_subsign(x in any::<f32>()) {
            prop_assert!(close_enough(super::definitions::softsign().compute(x), super::reference::subsign(x)));
         }


         #[test]
         fn test_hardswish(x in any::<f32>()) {
             prop_assert!(close_enough(super::definitions::hardswish().compute(x), super::reference::hardswish(x)));
         }

         #[test]
         fn test_sigmoid(x in any::<f32>()) {
             prop_assert!(close_enough(super::definitions::sigmoid().compute(x), super::reference::sigmoid(x)));
         }

         #[test]
         fn test_cm_exp2f(x in any::<f32>()) {
             prop_assert!(close_enough(super::definitions::exp2f().compute(x), super::reference::exp2f(x)));
         }
    }
}
