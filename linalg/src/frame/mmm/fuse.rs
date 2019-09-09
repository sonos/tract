use num_traits::Zero;
use std::fmt;
use std::fmt::Debug;
use std::ops::{Add, Mul};

#[derive(PartialEq, Clone)]
pub enum FusedSpec<T>
where
    T: Copy + Clone + Debug + Add + Mul + Zero,
{
    Min(T),
    Max(T),
    AddC,
    PerRowMul(Vec<T>),
    PerRowAdd(Vec<T>),
    PerColMul(Vec<T>),
    PerColAdd(Vec<T>),
}

impl<T> Debug for FusedSpec<T>
where
    T: Copy + Clone + Debug + Add + Mul + Zero,
{
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match self {
            FusedSpec::Min(t) => write!(fmt, "Min({:?})", t),
            FusedSpec::Max(t) => write!(fmt, "Max({:?})", t),
            FusedSpec::AddC => write!(fmt, "AddC"),
            FusedSpec::PerRowMul(_) => write!(fmt, "PerRowMul"),
            FusedSpec::PerRowAdd(_) => write!(fmt, "PerRowAdd"),
            FusedSpec::PerColMul(_) => write!(fmt, "PerColMul"),
            FusedSpec::PerColAdd(_) => write!(fmt, "PerColAdd"),
        }
    }
}

#[repr(C, usize)]
#[derive(PartialEq, Copy, Clone, Debug)]
pub enum FusedKerSpec<T>
where
    T: Copy + Clone + Debug + Add + Mul + Zero,
{
    Done,
    Min(T),
    Max(T),
    AddC,
    PerRowMul(*const T),
    PerRowAdd(*const T),
    PerColMul(*const T),
    PerColAdd(*const T),
}
