use crate::frame::activations::{ActivationKer, Op, RegisterId};

// TODO make the inner loop tighter
unsafe fn compute_slice(ops: *const Op, constants: *const f32, xs: *mut f32, len: usize) {
    let mut a = std::slice::from_raw_parts_mut(xs, len);
    let mut b = vec![0.0f32; a.len()];
    let mut c = vec![0.0f32; a.len()];
    let mut pc = ops;
    loop {
        match *pc {
            Op::Done => break,
            Op::Move(dst, src) => {
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
            Op::Load(dst, cst) if dst == RegisterId::A => {
                a.iter_mut().for_each(|x| *x = *constants.add(cst as usize))
            }
            Op::Load(dst, cst) if dst == RegisterId::B => {
                b.iter_mut().for_each(|x| *x = *constants.add(cst as usize))
            }
            Op::Load(_dst, cst) => c.iter_mut().for_each(|x| *x = *constants.add(cst as usize)),
            Op::Abs => a.iter_mut().for_each(|x| *x = x.abs()),
            Op::Recip => a.iter_mut().for_each(|x| *x = x.recip()),
            Op::Add => a.iter_mut().zip(&b).for_each(|(x, y)| *x += *y),
            Op::Sub => a.iter_mut().zip(&b).for_each(|(x, y)| *x -= *y),
            Op::Mul => a.iter_mut().zip(&b).for_each(|(x, y)| *x *= *y),
            Op::Min => a.iter_mut().zip(&b).for_each(|(x, y)| *x = x.min(*y)),
            Op::Max => a.iter_mut().zip(&b).for_each(|(x, y)| *x = x.max(*y)),
            Op::AddConst(cst) => a.iter_mut().for_each(|x| *x += *constants.add(cst as usize)),
            Op::SubConst(cst) => a.iter_mut().for_each(|x| *x -= *constants.add(cst as usize)),
            Op::MulConst(cst) => a.iter_mut().for_each(|x| *x *= *constants.add(cst as usize)),
            Op::MinConst(cst) => {
                a.iter_mut().for_each(|x| *x = x.min(*constants.add(cst as usize)))
            }
            Op::MaxConst(cst) => {
                a.iter_mut().for_each(|x| *x = x.max(*constants.add(cst as usize)))
            }
            Op::IfPosTE => a
                .iter_mut()
                .zip(&b)
                .zip(&c)
                .for_each(|((x, y), z)| *x = if *x >= 0f32 { *y } else { *z }),
            Op::FMA(cst) => {
                a.iter_mut().zip(&b).for_each(|(x, y)| *x = *x * *y + *constants.add(cst as usize))
            }
            Op::SwapBC => b.iter_mut().zip(c.iter_mut()).for_each(|(b, c)| std::mem::swap(b, c)),
            Op::Floor => a.iter_mut().for_each(|x| *x = x.floor()),
            Op::TwoPowOfInt => {
                a.iter_mut().for_each(|x| *x = f32::from_bits((((*x as i32) + 127) as u32) << 23))
            }
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

    fn run(ops: &[Op], csts: &[f32], xs: &mut [f32]) {
        debug_assert!(xs.len() % Self::nr() == 0);
        debug_assert!(xs.as_ptr() as usize % Self::alignment_bytes() == 0);
        unsafe { compute_slice(ops.as_ptr(), csts.as_ptr(), xs.as_mut_ptr(), xs.len()) };
    }
}

#[cfg(test)]
act_frame_tests!(true, SActivations, f32);

