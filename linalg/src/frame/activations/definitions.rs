use super::Op::*;
use super::RegisterId::*;
use super::*;

pub fn relu<T: LADatum>() -> Program<T> {
    Program { ops: vec![MaxConst(T::zero())] }
}

pub fn affine<T: LADatum>(alpha: T, beta: T) -> Program<T> {
    Program {
        #[rustfmt::skip]
        ops: vec![
            MulConst(alpha),
            AddConst(beta),
        ],
    }
}

pub fn leaky_relu<T: LADatum>(alpha: T) -> Program<T> {
    Program {
        #[rustfmt::skip]
        ops: vec![
            Move(B,A),
            MulConst(alpha),
            Move(C,A),
            Move(A,B),
            IfPosTE,
        ],
    }
}

pub fn threshold_relu<T: LADatum>(alpha: T) -> Program<T> {
    Program {
        #[rustfmt::skip]
        ops: vec![
            Move(B,A),
            SubConst(alpha),
            Load(C, T::zero()),
            IfPosTE,
        ],
    }
}

pub fn hard_sigmoid<T: LADatum>(alpha: T, beta: T) -> Program<T> {
    Program {
        #[rustfmt::skip]
        ops: vec![
            MulConst(alpha),
            AddConst(beta),
            MinConst(T::one()),
            MaxConst(T::zero()),
        ],
    }
}

pub fn softsign<T: LADatum>() -> Program<T> {
    Program {
        #[rustfmt::skip]
        ops: vec![
            Move(B,A),
            Abs,
            AddConst(T::one()),
            Recip,
            Mul,
        ],
    }
}

pub fn hard_swish<T: LADatum>() -> Program<T> {
    let one_sixth = T::one() / (T::one() + T::one() + T::one() + T::one() + T::one() + T::one());
    let one_half = T::one() / (T::one() + T::one());
    Program {
        #[rustfmt::skip]
        ops: vec![
            Move(B, A),
            MulConst(one_sixth),
            AddConst(one_half),
            MinConst(T::one()),
            MaxConst(T::zero()),
            Mul,
        ],
    }
}

pub fn sigmoid() -> Program<f32> {
    Program {
        ops: vec![
            MaxConst(-18.6),            // const 2
            MinConst(18.6),             // const 3
            Move(B, A),                 // b = x
            Move(C, A),                 // c = x
            Mul,                        // a = x2
            Move(B, A),                 // b = x2
            MulConst(-4.433153405e-18), // const 4, also alpha_13
            AddConst(1.169974371e-14),  // const 5, also a11
            FMA(-1.875289645e-11),
            FMA(4.257889523e-8),
            FMA(0.00004811817576),      // const 8
            FMA(0.008163842030),
            FMA(0.2499999971),        // alpha_1
            SwapBC,                   // c = x2, b = x
            Mul,                      // a = p(x)
            Move(B, C),               // b = x2
            Move(C, A),               // c = p(x)
            Move(A, B),               // a = x2
            MulConst(3.922935744e-6), // beta_6
            AddConst(0.001524872358), // const 12
            FMA(0.1159886749),
            FMA(1.0), // a = q(x)
            Recip,
            Move(B, C), // b = p(x)
            Mul,
            AddConst(0.5), //beta_0
        ],
    }
}

/*
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
*/
