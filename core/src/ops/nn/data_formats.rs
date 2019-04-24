use crate::model::TVec;
use crate::dim::DimLike;
use std::fmt;

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
    pub fn shape<D, S>(&self, shape: S) -> BaseDataShape<D, S>
    where
        D: DimLike,
        S: AsRef<[D]> + fmt::Debug,
    {
        let mut strides: TVec<D> = tvec![1.into()];
        for dim in shape.as_ref().iter().skip(1).rev() {
            let previous = strides.last().unwrap();
            strides.push(*previous * *dim);
        }
        strides.reverse();
        BaseDataShape { fmt: *self, shape, strides }
    }

    pub fn from_n_c_hw<D, S>(&self, n: D, c: D, shape: S) -> BaseDataShape<D, TVec<D>>
    where
        D: DimLike,
        S: AsRef<[D]> + fmt::Debug,
    {
        let mut me = tvec!(n);
        if *self == DataFormat::NCHW {
            me.push(c);
        }
        me.extend(shape.as_ref().iter().cloned());
        if *self == DataFormat::NHWC {
            me.push(c);
        }
        self.shape(me)
    }
}

pub type DataShape = BaseDataShape<usize, TVec<usize>>;

#[derive(Clone, Debug, PartialEq)]
pub struct BaseDataShape<D, S>
where
    D: DimLike,
    S: AsRef<[D]> + fmt::Debug,
{
    pub fmt: DataFormat,
    pub shape: S,
    pub strides: TVec<D>,
}

impl<D, S> BaseDataShape<D, S>
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

    pub fn n_stride(&self) -> D {
        unsafe { *self.strides.get_unchecked(self.n_axis()) }
    }

    pub fn hw_strides(&self) -> &[D] {
        unsafe { self.strides.as_ref().get_unchecked(self.hw_axes()) }
    }

    pub fn w_stride(&self) -> D {
        unsafe { *self.hw_strides().get_unchecked(self.hw_rank()-1) }
    }

    pub fn c_stride(&self) -> D {
        unsafe { *self.strides.get_unchecked(self.c_axis()) }
    }

}
