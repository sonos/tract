use std::fmt::Debug;
use std::marker::PhantomData;

use tract_data::TractResult;

use crate::LADatum;

use super::element_wise_helper::run_over_slice_with_alignment;

pub mod definitions;
pub mod reference;
#[macro_use]
pub mod tests;

#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(u8)]
pub enum RegisterId {
    A = 0,
    B = 1,
    C = 2,
}

type ConstantId = u8;

#[repr(C, u16)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Op {
    Done,
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
pub struct Program<T: LADatum> {
    pub ops: Vec<Op>,
    pub csts: Vec<T>,
}

pub trait Activation<T: LADatum>: Send + Sync + Debug + dyn_clone::DynClone {
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
    fn run(&self, program: &Program<T>, vec: &mut [T]) -> TractResult<()> {
        run_over_slice_with_alignment(
            vec,
            |slice| K::run(&program.ops, &*program.csts, slice),
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
    fn run(ops: &[Op], csts: &[T], vec: &mut [T]);
    fn act() -> Box<dyn Activation<T>> {
        Box::new(ActivationImpl::<Self, T>::new())
    }
}

macro_rules! act_impl {
    ($ti: ident, $func: ident, $nr: expr, $alignment_items: expr, $cond: expr) => {
        paste! {
            mod [<sys_ $func>] {
                #[allow(unused_imports)]
                use tract_data::prelude::f16;
                use crate::frame::activations::Op;
                extern_kernel!(fn $func(ops: *const Op, constants: *const $ti, xs: *mut $ti, len: usize) -> usize);
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
                fn run(ops: &[$crate::frame::activations::Op], csts:&[$ti], buf: &mut [$ti]) {
                    let err = unsafe { [<sys_ $func>]::$func(ops.as_ptr(), csts.as_ptr(), buf.as_mut_ptr(), buf.len()) };
                    assert_eq!(err, 0);
                }
            }

            mod [<test_ $func>] {
                use super::*;

                #[cfg(test)]
                act_tests!($cond, $func, $ti);
            }
        }
    };
}

#[cfg(test)]
mod test {
    #[test]
    fn size_of_op() {
        assert_eq!(std::mem::size_of::<super::Op>(), 4);
    }
}
