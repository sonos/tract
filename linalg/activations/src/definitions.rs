
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
