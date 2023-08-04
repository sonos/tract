use crate::frame::activations::{ActivationKer, KerOp, OpOrConst, RegisterId};

// TODO make the inner loop tighter
unsafe fn compute_slice(ops: *const OpOrConst<f32>, xs: *mut f32, len: usize) {
    let mut a = std::slice::from_raw_parts_mut(xs, len);
    let mut b = vec![0.0f32; a.len()];
    let mut c = vec![0.0f32; a.len()];
    let mut pc = ops;
    loop {
        let op = (*pc).op;
        match op {
            KerOp::Done => break,
            KerOp::Move(dst, src) => {
                let mut regs = [&mut a, &mut *b, &mut c];
                let dst = dst as usize;
                let src = src as usize;
                if dst < src {
                    let (left, right) = regs.split_at_mut(src);
                    let d = &mut *left[dst];
                    let s = &*right[0];
                    d.copy_from_slice(s)
                } else {
                    let (left, right) = regs.split_at_mut(dst);
                    let s = &*left[src];
                    let d = &mut *right[0];
                    d.copy_from_slice(s)
                }
            }
            KerOp::Load(dst) => {
                pc = pc.add(1);
                let t = (*pc).t;
                match dst {
                    RegisterId::A => a.iter_mut().for_each(|x| *x = t),
                    RegisterId::B => b.iter_mut().for_each(|x| *x = t),
                    RegisterId::C => c.iter_mut().for_each(|x| *x = t),
                }
            }
            KerOp::Abs => a.iter_mut().for_each(|x| *x = x.abs()),
            KerOp::Recip => a.iter_mut().for_each(|x| *x = x.recip()),
            KerOp::Add => a.iter_mut().zip(&b).for_each(|(x, y)| *x += *y),
            KerOp::Sub => a.iter_mut().zip(&b).for_each(|(x, y)| *x -= *y),
            KerOp::Mul => a.iter_mut().zip(&b).for_each(|(x, y)| *x *= *y),
            KerOp::Min => a.iter_mut().zip(&b).for_each(|(x, y)| *x = x.min(*y)),
            KerOp::Max => a.iter_mut().zip(&b).for_each(|(x, y)| *x = x.max(*y)),
            KerOp::AddConst
            | KerOp::SubConst
            | KerOp::MulConst
            | KerOp::MinConst
            | KerOp::MaxConst
            | KerOp::FMA => {
                pc = pc.add(1);
                let t = (*pc).t;
                match op {
                    KerOp::AddConst => a.iter_mut().for_each(|x| *x += t),
                    KerOp::SubConst => a.iter_mut().for_each(|x| *x -= t),
                    KerOp::MulConst => a.iter_mut().for_each(|x| *x *= t),
                    KerOp::MinConst => a.iter_mut().for_each(|x| *x = x.min(t)),
                    KerOp::MaxConst => a.iter_mut().for_each(|x| *x = x.max(t)),
                    KerOp::FMA => a.iter_mut().zip(&b).for_each(|(x, y)| *x = *x * *y + t),
                    _ => unreachable!(),
                }
            }
            KerOp::IfPosTE => a
                .iter_mut()
                .zip(&b)
                .zip(&c)
                .for_each(|((x, y), z)| *x = if *x >= 0f32 { *y } else { *z }),
            KerOp::SwapBC => b.iter_mut().zip(c.iter_mut()).for_each(|(b, c)| std::mem::swap(b, c)),
            KerOp::Floor => a.iter_mut().for_each(|x| *x = x.floor()),
            KerOp::TwoPowOfInt => {
                a.iter_mut().for_each(|x| *x = f32::from_bits((((*x as i32) + 127) as u32) << 23))
            }
            KerOp::Noop => {},
        }
        pc = pc.add(1);
    }
}

#[derive(Clone, Debug)]
pub struct SActivations;

impl ActivationKer<f32> for SActivations {
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

    fn run(ops: &[OpOrConst<f32>], xs: &mut [f32]) {
        debug_assert!(xs.len() % Self::nr() == 0);
        debug_assert!(xs.as_ptr() as usize % Self::alignment_bytes() == 0);
        unsafe { compute_slice(ops.as_ptr(), xs.as_mut_ptr(), xs.len()) };
    }
}

#[cfg(test)]
mod tests {
    use crate::frame::activations::ActivationKer;
    use crate::frame::activations::KerOp;
    use crate::frame::activations::OpOrConst;
    use super::SActivations;

    act_tests!(true, crate::generic::activations::SActivations, f32);

    #[test]
    fn act_noop() {
        let mut xs = vec![1f32; SActivations::nr()];
        let expect = xs.clone();
        SActivations::run(&[OpOrConst { op: KerOp::Done }], &mut *xs);
        assert_eq!(expect, xs);
    }
}
