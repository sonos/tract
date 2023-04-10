pub mod definitions;
pub mod reference;

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum RegisterId {
    A = 0,
    B = 1,
    C = 2,
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
    pub fn compute_slice(&self, xs: &mut [f32]) {
        let mut a = xs.to_vec();
        let mut b = vec![0.0f32; a.len()];
        let mut c = vec![0.0f32; a.len()];
        let mut constants = self.csts.clone();
        constants.insert(0, 0f32);
        constants.insert(1, 1f32);
        for op in &self.ops {
            match op {
                Op::Move(dst, src) => {
                    let mut regs = [&mut a, &mut b, &mut c];
                    let dst = *dst as usize;
                    let src = *src as usize;
                    if dst < src {
                        let (left, right) = regs.split_at_mut(src);
                        let d = &mut **left[dst];
                        let s = &**right[0];
                        d.copy_from_slice(s)
                    } else {
                        let (left, right) = regs.split_at_mut(dst);
                        let s = &**left[src];
                        let d = &mut **right[0];
                        d.copy_from_slice(s)
                    }
                }
                Op::Load(dst, cst) if *dst == RegisterId::A => {
                    a.iter_mut().for_each(|x| *x = constants[*cst])
                }
                Op::Load(dst, cst) if *dst == RegisterId::B => {
                    b.iter_mut().for_each(|x| *x = constants[*cst])
                }
                Op::Load(_dst, cst) => c.iter_mut().for_each(|x| *x = constants[*cst]),
                Op::Abs => a.iter_mut().for_each(|x| *x = x.abs()),
                Op::Recip => a.iter_mut().for_each(|x| *x = x.recip()),
                Op::Add => a.iter_mut().zip(&b).for_each(|(x, y)| *x += *y),
                Op::Sub => a.iter_mut().zip(&b).for_each(|(x, y)| *x -= *y),
                Op::Mul => a.iter_mut().zip(&b).for_each(|(x, y)| *x *= *y),
                Op::Min => a.iter_mut().zip(&b).for_each(|(x, y)| *x = x.min(*y)),
                Op::Max => a.iter_mut().zip(&b).for_each(|(x, y)| *x = x.max(*y)),
                Op::AddConst(cst) => a.iter_mut().for_each(|x| *x += constants[*cst]),
                Op::SubConst(cst) => a.iter_mut().for_each(|x| *x -= constants[*cst]),
                Op::MulConst(cst) => a.iter_mut().for_each(|x| *x *= constants[*cst]),
                Op::MinConst(cst) => a.iter_mut().for_each(|x| *x = x.min(constants[*cst])),
                Op::MaxConst(cst) => a.iter_mut().for_each(|x| *x = x.max(constants[*cst])),
                Op::IfPosTE => a
                    .iter_mut()
                    .zip(&b)
                    .zip(&c)
                    .for_each(|((x, y), z)| *x = if *x >= 0f32 { *y } else { *z }),
                Op::FMA(cst) => {
                    a.iter_mut().zip(&b).for_each(|(x, y)| *x = *x * *y + constants[*cst])
                }
                Op::SwapBC => {
                    b.iter_mut().zip(c.iter_mut()).for_each(|(b, c)| std::mem::swap(b, c))
                }
                Op::Floor => a.iter_mut().for_each(|x| *x = x.floor()),
                Op::TwoPowOfInt => a
                    .iter_mut()
                    .for_each(|x| *x = f32::from_bits((((*x as i32) + 127) as u32) << 23)),
            }
        }
        xs.copy_from_slice(&a)
    }

    pub fn compute(&self, x: f32) -> f32 {
        let mut regs = [0f32; 3];
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

#[cfg(test)]
mod test {

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
        let result = (a.is_infinite() && b.is_infinite() && a.signum() == b.signum())
            || ((a - b).abs() <= max(rtol * max(a.abs(), b.abs()), atol));
        if !result {
            dbg!(a, b);
        }
        return result;
    }

    mod scalar {
        use super::close_enough;
        use proptest::prelude::*;

        macro_rules! prop_activation {
            ($name: ident ( $($param:ident),* )) => {
                proptest! {
                    #[test]
                    fn $name(x in any::<f32>(), $($param in any::<f32>()),*) {
                        prop_assert!(close_enough(crate::definitions::$name($($param),*).compute(x),crate::reference::$name(x, $($param),*)))
                    }
                }
            }
        }

        prop_activation!(relu());
        prop_activation!(affine(alpha, beta));
        prop_activation!(leaky_relu(alpha));
        prop_activation!(threshold_relu(alpha));
        prop_activation!(softsign());
        prop_activation!(hardswish());
        prop_activation!(sigmoid());
        prop_activation!(exp2f());
    }

    mod vector {
        use super::close_enough;
        use proptest::prelude::*;

        macro_rules! prop_activation {
            ($name: ident ( $($param:ident),* )) => {
                proptest! {
                    #[test]
                    fn $name(x in any::<f32>(), $($param in any::<f32>()),*) {
                        let mut slice = [x];
                        crate::definitions::$name($($param),*).compute_slice(&mut slice);
                        prop_assert!(close_enough(slice[0], crate::reference::$name(x, $($param),*)))
                    }
                }
            }
        }

        prop_activation!(relu());
        prop_activation!(affine(alpha, beta));
        prop_activation!(leaky_relu(alpha));
        prop_activation!(threshold_relu(alpha));
        prop_activation!(softsign());
        prop_activation!(hardswish());
        prop_activation!(sigmoid());
        prop_activation!(exp2f());
    }
}
