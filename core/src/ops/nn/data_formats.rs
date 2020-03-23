use crate::dim::DimLike;
use crate::model::TVec;
use std::fmt;

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum DataFormat {
    NCHW,
    NHWC,
    CHW,
    HWC,
}

impl Default for DataFormat {
    fn default() -> DataFormat {
        DataFormat::NCHW
    }
}

impl DataFormat {
    pub fn dispose_n_axis(&self) -> DataFormat {
        match self {
            &DataFormat::NCHW => DataFormat::CHW,
            &DataFormat::NHWC => DataFormat::HWC,
            _ => panic!("Attempt at removing N axis on {:?}", self),
        }
    }

    pub fn shape<D, S>(&self, shape: S) -> BaseDataShape<D, S>
    where
        D: DimLike,
        S: AsRef<[D]> + fmt::Debug,
    {
        if shape.as_ref().iter().filter(|d| d.to_integer().is_err()).count() > 1 {
            panic!("Can not work out a data format with two actual symbolic dim")
        }
        let mut strides: Vec<i32> = vec![1];
        for dim in shape.as_ref().iter().skip(1).rev() {
            let previous = strides.last().unwrap().clone();
            strides.push(previous * dim.to_integer().unwrap_or(1));
        }
        strides.reverse();
        let stream_index = shape.as_ref().iter().position(|d| d.to_integer().is_err()).unwrap_or(0);
        let strides = strides
            .iter()
            .enumerate()
            .map(|(ix, d)| {
                if ix < stream_index {
                    shape.as_ref()[stream_index].clone() * *d as usize
                } else {
                    (*d as usize).into()
                }
            })
            .collect();
        BaseDataShape { fmt: *self, shape, strides }
    }

    pub fn from_n_c_hw<D, S>(&self, n: D, c: D, shape: S) -> BaseDataShape<D, TVec<D>>
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
            me.push(c.clone());
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
    pub strides: Vec<D>,
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
            DataFormat::NHWC => self.shape.as_ref().len() - 1,
            DataFormat::HWC => self.shape.as_ref().len() - 1,
            DataFormat::NCHW => 1,
            DataFormat::CHW => 0,
        }
    }

    #[inline]
    pub fn h_axis(&self) -> usize {
        match self.fmt {
            DataFormat::NHWC => 1,
            DataFormat::HWC => 0,
            DataFormat::NCHW => 2,
            DataFormat::CHW => 1,
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
        unsafe { &self.hw_strides().get_unchecked(0) }
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
