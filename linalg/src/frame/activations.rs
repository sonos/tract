use std::fmt::Debug;
use std::marker::PhantomData;

use tract_data::TractResult;

use crate::LADatum;

use super::element_wise_helper::run_over_slice_with_alignment;

pub mod definitions;
pub mod reference;
#[macro_use]
#[cfg(test)]
pub mod tests;

#[derive(Clone, Debug, PartialEq)]
pub struct Program<T: LADatum> {
    pub ops: Vec<Op<T>>,
}

#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(u8)]
pub enum RegisterId {
    A = 0,
    B = 1,
    C = 2,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Op<T: LADatum> {
    Move(RegisterId, RegisterId),
    Load(RegisterId, T),
    Abs, // 3
    Recip,
    Add,
    Sub, // 6
    Mul,
    Min,
    Max,         // 9
    AddConst(T), // 10
    SubConst(T),
    MulConst(T),
    MinConst(T),
    MaxConst(T), // 14
    FMA(T),      // a <- a * b + cst
    IfPosTE,
    SwapBC,
    Floor,
    TwoPowOfInt,
    Noop,
}

impl<T: LADatum> Program<T> {
    pub fn translate(&self) -> KerProgram<T> {
        let mut ops: Vec<OpOrConst<T>> = vec![];
        for op in &self.ops {
            match op {
                Op::Move(a, b) => ops.push(OpOrConst { op: KerOp::Move(*a, *b) }),
                Op::Load(a, t) => {
                    ops.push(OpOrConst { op: KerOp::Load(*a) });
                    ops.push(OpOrConst { t: *t });
                }
                Op::Abs => ops.push(OpOrConst { op: KerOp::Abs }),
                Op::Recip => ops.push(OpOrConst { op: KerOp::Recip }),
                Op::Add => ops.push(OpOrConst { op: KerOp::Add }),
                Op::Sub => ops.push(OpOrConst { op: KerOp::Sub }), // 6
                Op::Mul => ops.push(OpOrConst { op: KerOp::Mul }),
                Op::Min => ops.push(OpOrConst { op: KerOp::Min }),
                Op::Max => ops.push(OpOrConst { op: KerOp::Max }), // 9
                Op::AddConst(t) => {
                    ops.push(OpOrConst { op: KerOp::AddConst });
                    ops.push(OpOrConst { t: *t });
                }
                Op::SubConst(t) => {
                    ops.push(OpOrConst { op: KerOp::SubConst });
                    ops.push(OpOrConst { t: *t });
                }
                Op::MulConst(t) => {
                    ops.push(OpOrConst { op: KerOp::MulConst });
                    ops.push(OpOrConst { t: *t });
                }
                Op::MinConst(t) => {
                    ops.push(OpOrConst { op: KerOp::MinConst });
                    ops.push(OpOrConst { t: *t });
                }
                Op::MaxConst(t) => {
                    ops.push(OpOrConst { op: KerOp::MaxConst });
                    ops.push(OpOrConst { t: *t });
                }
                Op::FMA(t) => {
                    ops.push(OpOrConst { op: KerOp::FMA });
                    ops.push(OpOrConst { t: *t });
                }
                Op::IfPosTE => ops.push(OpOrConst { op: KerOp::IfPosTE }),
                Op::SwapBC => ops.push(OpOrConst { op: KerOp::SwapBC }),
                Op::Floor => ops.push(OpOrConst { op: KerOp::Floor }),
                Op::TwoPowOfInt => ops.push(OpOrConst { op: KerOp::TwoPowOfInt }),
                Op::Noop => ops.push(OpOrConst { op: KerOp::Noop }),
            }
        }
        ops.push(OpOrConst { op: KerOp::Done });
        ops.push(OpOrConst { op: KerOp::Done }); // add a second one to help with pair load
        KerProgram { ops }
    }
}

#[repr(C, u16)]
#[derive(Copy, Clone, Debug, PartialEq)]
#[rustfmt::skip]
pub enum KerOp {
    Done,                           // jump_to:done
    Move(RegisterId, RegisterId),   // jump_to:move
    Load(RegisterId),               // jump_to:load
    Abs,                            // jump_to:abs
    Recip,                          // jump_to:recip
    Add,                            // jump_to:add
    Sub,                            // jump_to:sub
    Mul,                            // jump_to:mul
    Min,                            // jump_to:min
    Max,                            // jump_to:max
    AddConst,                       // jump_to:add_const
    SubConst,                       // jump_to:sub_const
    MulConst,                       // jump_to:mul_const
    MinConst,                       // jump_to:min_const
    MaxConst,                       // jump_to:max_const
    // a <- a * b + cst
    FMA,                            // jump_to:fma
    IfPosTE,                        // jump_to:if_pos_then_else
    SwapBC,                         // jump_to:swap_b_c
    Floor,                          // jump_to:floor
    TwoPowOfInt,                    // jump_to:two_pow_of_int
    Noop                            // jump_to:noop
}

#[derive(Clone)]
pub struct KerProgram<T: LADatum> {
    pub ops: Vec<OpOrConst<T>>,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub union OpOrConst<T: LADatum> {
    pub op: KerOp,
    pub t: T,
}

pub trait Activation<T: LADatum>: Send + Sync + Debug + dyn_clone::DynClone {
    fn name(&self) -> &'static str;
    fn run(&self, prog: &Program<T>, vec: &mut [T]) -> TractResult<()>;
}

#[derive(Debug, Clone, new)]
pub struct ActivationImpl<K, T>
where
    T: LADatum,
    K: ActivationKer<T> + Clone,
{
    phantom: PhantomData<(K, T)>,
}

impl<K, T> Activation<T> for ActivationImpl<K, T>
where
    T: LADatum,
    K: ActivationKer<T> + Clone,
{
    fn name(&self) -> &'static str {
        K::name()
    }

    fn run(&self, program: &Program<T>, vec: &mut [T]) -> TractResult<()> {
        let ker_program = program.translate();
        run_over_slice_with_alignment(
            vec,
            |slice| K::run(&ker_program.ops, slice),
            K::nr(),
            K::alignment_bytes(),
        )
    }
}

pub trait ActivationKer<T>: Send + Sync + Debug + dyn_clone::DynClone + Clone + 'static
where
    T: LADatum,
{
    fn name() -> &'static str;
    fn alignment_bytes() -> usize;
    fn alignment_items() -> usize;
    fn nr() -> usize;
    fn run(ops: &[OpOrConst<T>], vec: &mut [T]);
    fn act() -> Box<dyn Activation<T>> {
        Box::new(ActivationImpl::<Self, T>::new())
    }
}

#[allow(unused_macros)]
macro_rules! act_impl {
    ($ti: ident, $func: ident, $nr: expr, $alignment_items: expr, $cond: expr) => {
        paste! {
            mod [<sys_ $func>] {
                #[allow(unused_imports)]
                use tract_data::prelude::f16;
                use $crate::frame::activations::OpOrConst;
                extern_kernel!(fn $func(ops: *const OpOrConst<$ti>, xs: *mut $ti, len: usize) -> usize);
            }

            #[derive(Copy, Clone, Debug)]
            #[allow(non_camel_case_types)]
            pub struct $func;

            impl $crate::frame::activations::ActivationKer<$ti> for $func {
                #[inline(always)]
                fn name() -> &'static str {
                    stringify!($func)
                }
                #[inline(always)]
                fn nr() -> usize {
                    $nr
                }
                #[inline(always)]
                fn alignment_items() -> usize {
                    $alignment_items
                }
                #[inline(always)]
                fn alignment_bytes() -> usize {
                    $alignment_items * std::mem::size_of::<$ti>()
                }
                #[inline(never)]
                fn run(ops: &[$crate::frame::activations::OpOrConst<$ti>], buf: &mut [$ti]) {
                    let err = unsafe { [<sys_ $func>]::$func(ops.as_ptr(), buf.as_mut_ptr(), buf.len()) };
                    assert_eq!(err, 0, "Kernel function return non zero {}", err);
                }
            }

            #[cfg(test)]
            mod [<test_ $func>] {
                pub use super::*;
                act_tests!($cond, $func, $ti);
            }
        }
    };
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn size_of_op() {
        assert_eq!(std::mem::size_of::<OpOrConst<f32>>(), 4);
    }
}
