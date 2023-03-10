use crate::internal::*;
use std::fmt;
use tract_itertools::Itertools;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub enum DataFormat {
    #[default]
    NCHW,
    NHWC,
    CHW,
    HWC,
}

impl DataFormat {
    pub fn dispose_n_axis(&self) -> DataFormat {
        match self {
            DataFormat::NCHW => DataFormat::CHW,
            DataFormat::NHWC => DataFormat::HWC,
            _ => panic!("Attempt at removing N axis on {self:?}"),
        }
    }

    pub fn shape<D, S>(&self, shape: S) -> TractResult<BaseDataShape<D, S>>
    where
        D: DimLike,
        S: AsRef<[D]> + fmt::Debug,
    {
        let mut strides: TVec<D> = tvec![D::one()];
        for dim in shape.as_ref().iter().skip(1).rev() {
            let previous = strides.last().unwrap().clone();
            strides.push(previous * dim)
        }
        strides.reverse();
        Ok(BaseDataShape { fmt: *self, shape, strides })
    }

    pub fn from_n_c_hw<D, S>(&self, n: D, c: D, shape: S) -> TractResult<BaseDataShape<D, TVec<D>>>
    where
        D: DimLike,
        S: AsRef<[D]> + fmt::Debug,
    {
        let mut me = tvec!();
        if *self == DataFormat::NCHW || *self == DataFormat::NHWC {
            me.push(n);
        }
        if *self == DataFormat::NCHW || *self == DataFormat::CHW {
            me.push(c.clone());
        }
        me.extend(shape.as_ref().iter().cloned());
        if *self == DataFormat::NHWC || *self == DataFormat::HWC {
            me.push(c);
        }
        self.shape(me)
    }

    pub fn has_n(&self) -> bool {
        *self == DataFormat::NHWC || *self == DataFormat::NCHW
    }

    pub fn c_is_last(&self) -> bool {
        *self == DataFormat::NHWC || *self == DataFormat::HWC
    }

    pub fn h_axis(&self) -> usize {
        self.has_n() as usize + !self.c_is_last() as usize
    }

    pub fn with_n(&self) -> DataFormat {
        match self {
            DataFormat::CHW => DataFormat::NCHW,
            DataFormat::HWC => DataFormat::NHWC,
            _ => *self,
        }
    }
}

pub type SymDataShape = BaseDataShape<TDim, TVec<TDim>>;
pub type DataShape = BaseDataShape<usize, TVec<usize>>;

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct BaseDataShape<D, S>
where
    D: DimLike,
    S: AsRef<[D]> + fmt::Debug,
{
    pub fmt: DataFormat,
    pub shape: S,
    pub strides: TVec<D>,
}

impl<D, S> fmt::Debug for BaseDataShape<D, S>
where
    D: DimLike,
    S: AsRef<[D]> + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:?} {} (strides: {})",
            self.fmt,
            self.shape.as_ref().iter().join(","),
            self.strides.iter().join(",")
        )
    }
}

impl<D, S> BaseDataShape<D, S>
where
    D: DimLike,
    S: AsRef<[D]> + fmt::Debug,
{
    #[inline]
    pub fn rank(&self) -> usize {
        self.shape.as_ref().len()
    }

    #[inline]
    pub fn hw_rank(&self) -> usize {
        self.rank() - 1 - self.n_axis().is_some() as usize
    }

    #[inline]
    pub fn n_axis(&self) -> Option<usize> {
        match self.fmt {
            DataFormat::NHWC | DataFormat::NCHW => Some(0),
            DataFormat::HWC | DataFormat::CHW => None,
        }
    }

    #[inline]
    pub fn c_axis(&self) -> usize {
        match self.fmt {
            DataFormat::NHWC | DataFormat::HWC => self.shape.as_ref().len() - 1,
            DataFormat::NCHW => 1,
            DataFormat::CHW => 0,
        }
    }

    #[inline]
    pub fn h_axis(&self) -> usize {
        match self.fmt {
            DataFormat::HWC => 0,
            DataFormat::NHWC | DataFormat::CHW => 1,
            DataFormat::NCHW => 2,
        }
    }

    #[inline]
    pub fn hw_axes(&self) -> ::std::ops::Range<usize> {
        self.h_axis()..self.h_axis() + self.hw_rank()
    }

    #[inline]
    pub fn n_dim(&self) -> Option<&D> {
        self.n()
    }

    #[inline]
    pub fn c_dim(&self) -> &D {
        self.c()
    }

    #[inline]
    pub fn hw_dims(&self) -> &[D] {
        unsafe { self.shape.as_ref().get_unchecked(self.hw_axes()) }
    }

    #[inline]
    pub fn n(&self) -> Option<&D> {
        unsafe { self.n_axis().map(|axis| self.shape.as_ref().get_unchecked(axis)) }
    }

    #[inline]
    pub fn c(&self) -> &D {
        unsafe { self.shape.as_ref().get_unchecked(self.c_axis()) }
    }

    #[inline]
    pub fn n_stride(&self) -> Option<&D> {
        unsafe { self.n_axis().map(|axis| self.strides.get_unchecked(axis)) }
    }

    #[inline]
    pub fn h_stride(&self) -> &D {
        unsafe { self.hw_strides().get_unchecked(0) }
    }

    #[inline]
    pub fn hw_strides(&self) -> &[D] {
        unsafe { self.strides.get_unchecked(self.hw_axes()) }
    }

    #[inline]
    pub fn w_stride(&self) -> &D {
        unsafe { self.hw_strides().get_unchecked(self.hw_rank() - 1) }
    }

    #[inline]
    pub fn c_stride(&self) -> &D {
        unsafe { self.strides.get_unchecked(self.c_axis()) }
    }
}
