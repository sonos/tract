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
    IfPosTE,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Program {
    ops: Vec<Op>,
    csts: Vec<f32>,
}

impl Program {
    fn compute(&self, x: f32) -> f32 {
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
            }
        }
        regs[0]
    }
}

mod definitions {
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
        let atol = 1e-08;
        return (a - b).abs() <= max(rtol * max(a.abs(), b.abs()), atol);
    }

    proptest! {
        #[test]
        fn test_relu(x in any::<f32>()) {
            prop_assert_eq!(super::definitions::relu().compute(x), x.max(0f32))
        }

        #[test]
        fn test_affine(x in any::<f32>(), alpha in any::<f32>(), beta in any::<f32>()) {
            prop_assert_eq!(super::definitions::affine(alpha, beta).compute(x), alpha * x + beta)
        }

        #[test]
        fn test_leaky_relu(x in any::<f32>(), alpha in any::<f32>()) {
            prop_assert_eq!(super::definitions::leaky_relu(alpha).compute(x), if x > 0f32 { x }  else { alpha * x });
        }

        #[test]
        fn test_threshold_relu(x in any::<f32>(), alpha in any::<f32>()) {
            prop_assert_eq!(super::definitions::threshold_relu(alpha).compute(x), if x >= alpha { x } else { 0f32 } );
        }

        #[test]
        fn test_subsign(x in any::<f32>()) {
            prop_assert!(close_enough(super::definitions::softsign().compute(x), x / (1.+x.abs())));
        }

        #[test]
        fn test_hardswish(x in any::<f32>()) {
            prop_assert!(close_enough(super::definitions::hardswish().compute(x), x * 0f32.max( 1f32.min((1./6.) * x + 0.5))));
        }
    }
}
