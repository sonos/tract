use crate::dim::DimLike;
use std::fmt;
use std::marker::PhantomData;

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum DataFormat {
    NCHW,
    NHWC,
}

impl Default for DataFormat {
    fn default() -> DataFormat {
        DataFormat::NCHW
    }
}

impl DataFormat {
    pub fn shape<D, S>(&self, shape: S) -> DataShape<D, S>
    where
        D: DimLike,
        S: AsRef<[D]> + fmt::Debug,
    {
        DataShape { fmt: *self, shape, _phantom: PhantomData }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct DataShape<D, S>
where
    D: DimLike,
    S: AsRef<[D]> + fmt::Debug,
{
    pub fmt: DataFormat,
    pub shape: S,
    _phantom: PhantomData<D>,
}

impl<D, S> DataShape<D, S>
where
    D: DimLike,
    S: AsRef<[D]> + fmt::Debug,
{
    pub fn rank(&self) -> usize {
        self.shape.as_ref().len()
    }

    pub fn hw_rank(&self) -> usize {
        self.shape.as_ref().len() - 2
    }

    pub fn n_axis(&self) -> usize {
        0
    }

    pub fn c_axis(&self) -> usize {
        match self.fmt {
            DataFormat::NHWC => self.shape.as_ref().len() - 1,
            DataFormat::NCHW => 1,
        }
    }

    pub fn h_axis(&self) -> usize {
        match self.fmt {
            DataFormat::NHWC => 1,
            DataFormat::NCHW => 2,
        }
    }

    pub fn hw_axes(&self) -> ::std::ops::Range<usize> {
        self.h_axis()..self.h_axis() + self.hw_rank()
    }

    pub fn n_dim(&self) -> D {
        unsafe { *self.shape.as_ref().get_unchecked(self.n_axis()) }
    }

    pub fn c_dim(&self) -> D {
        unsafe { *self.shape.as_ref().get_unchecked(self.c_axis()) }
    }

    pub fn hw_dims(&self) -> &[D] {
        unsafe { self.shape.as_ref().get_unchecked(self.hw_axes()) }
    }

    pub fn n(&self) -> D {
        unsafe { *self.shape.as_ref().get_unchecked(self.n_axis()) }
    }

    pub fn c(&self) -> D {
        unsafe { *self.shape.as_ref().get_unchecked(self.c_axis()) }
    }
}
