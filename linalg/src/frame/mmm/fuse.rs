use num_traits::Zero;
use std::fmt;
use std::fmt::Debug;
use std::ops::{Add, Mul};

#[derive(PartialEq, Clone)]
pub enum FusedSpec<TI>
where
    TI: Copy + Add + Mul + Zero + Debug + fmt::Display + PartialEq + Send + Sync,
{
    Min(TI),
    Max(TI),
    AddC,
    PerRowMul(Vec<TI>),
    PerRowAdd(Vec<TI>),
    PerColMul(Vec<TI>),
    PerColAdd(Vec<TI>),
}

impl<TI> Debug for FusedSpec<TI>
where
    TI: Copy + Add + Mul + Zero + Debug + fmt::Display + PartialEq + Send + Sync,
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
pub enum FusedKerSpec<TI>
where
    TI: Copy + Add + Mul + Zero + Debug + fmt::Display + PartialEq + Send + Sync,
{
    Done,
    Min(TI),
    Max(TI),
    AddC,
    PerRowMul(*const TI),
    PerRowAdd(*const TI),
    PerColMul(*const TI),
    PerColAdd(*const TI),
}
