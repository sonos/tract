use crate::tensor::*;
use anyhow::*;
use std::marker::PhantomData;

#[derive(Clone, Debug)]
enum Indexing<'a> {
    Prefix(usize),
    Custom { shape: &'a [usize], strides: &'a [isize] },
}

#[derive(Clone, Debug)]
pub struct TensorView<'a> {
    pub tensor: &'a Tensor,
    offset_bytes: isize,
    indexing: Indexing<'a>,
    phantom: PhantomData<&'a ()>,
}

impl<'a> TensorView<'a> {
    pub unsafe fn from_bytes(
        tensor: &'a Tensor,
        offset_bytes: isize,
        shape: &'a [usize],
        strides: &'a [isize],
    ) -> TensorView<'a> {
        TensorView {
            tensor,
            offset_bytes,
            indexing: Indexing::Custom { shape, strides },
            phantom: PhantomData,
        }
    }

    pub fn at_prefix(tensor: &'a Tensor, prefix: &[usize]) -> anyhow::Result<TensorView<'a>> {
        ensure!(
            prefix.len() <= tensor.rank() && prefix.iter().zip(tensor.shape()).all(|(p, d)| p < d),
            "Invalid prefix {:?} for shape {:?}",
            prefix,
            tensor.shape()
        );
        unsafe { Ok(Self::at_prefix_unchecked(tensor, prefix)) }
    }

    pub unsafe fn at_prefix_unchecked(tensor: &'a Tensor, prefix: &[usize]) -> TensorView<'a> {
        let offset_bytes =
            prefix.iter().zip(tensor.strides()).map(|(a, b)| *a as isize * b).sum::<isize>()
                * tensor.datum_type().size_of() as isize;
        TensorView {
            tensor,
            offset_bytes,
            indexing: Indexing::Prefix(prefix.len()),
            phantom: PhantomData,
        }
    }

    pub fn datum_type(&self) -> DatumType {
        self.tensor.datum_type()
    }

    pub fn shape(&self) -> &[usize] {
        match &self.indexing {
            Indexing::Prefix(i) => &self.tensor.shape()[*i..],
            Indexing::Custom { shape, .. } => &*shape,
        }
    }

    pub fn strides(&self) -> &[isize] {
        match &self.indexing {
            Indexing::Prefix(i) => &self.tensor.strides()[*i..],
            Indexing::Custom { strides, .. } => &*strides,
        }
    }

    pub fn len(&self) -> usize {
        self.shape().iter().product::<usize>()
    }

    pub fn rank(&self) -> usize {
        self.shape().len()
    }

    fn check_dt<D: Datum>(&self) -> anyhow::Result<()> {
        if self.datum_type() != D::datum_type() {
            anyhow::bail!(
                "TensorView datum type error: tensor is {:?}, accessed as {:?}",
                self.datum_type(),
                D::datum_type(),
            );
        }
        Ok(())
    }

    fn check_coords(&self, coords: &[usize]) -> anyhow::Result<()> {
        ensure!(
            coords.len() == self.rank()
                && coords.iter().zip(self.shape()).all(|(&x, &dim)| x < dim),
            "Can't access coordinates {:?} of TensorView of shape {:?}",
            coords,
            self.shape(),
        );
        Ok(())
    }

    /// Access the data as a pointer.
    pub fn as_ptr<D: Datum>(&self) -> anyhow::Result<*const D> {
        self.check_dt::<D>()?;
        Ok(unsafe { self.as_ptr_unchecked() })
    }

    /// Access the data as a pointer.
    pub unsafe fn as_ptr_unchecked<D: Datum>(&self) -> *const D {
        self.tensor.as_bytes().as_ptr().offset(self.offset_bytes) as *const D
    }

    /// Access the data as a pointer.
    pub unsafe fn as_ptr_mut_unchecked<D: Datum>(&mut self) -> *mut D {
        self.as_ptr_unchecked::<D>() as *mut D
    }

    /// Access the data as a mutable pointer.
    pub fn as_ptr_mut<D: Datum>(&mut self) -> anyhow::Result<*mut D> {
        Ok(self.as_ptr::<D>()? as *mut D)
    }

    /// Access the data as a slice.
    pub unsafe fn as_slice_unchecked<D: Datum>(&self) -> &[D] {
        std::slice::from_raw_parts::<D>(self.as_ptr_unchecked(), self.len())
    }

    /// Access the data as a slice.
    pub fn as_slice<D: Datum>(&self) -> anyhow::Result<&[D]> {
        self.check_dt::<D>()?;
        unsafe { Ok(self.as_slice_unchecked()) }
    }

    /// Access the data as a mutable slice.
    pub unsafe fn as_slice_mut_unchecked<D: Datum>(&mut self) -> &mut [D] {
        std::slice::from_raw_parts_mut::<D>(self.as_ptr_mut_unchecked(), self.len())
    }

    /// Access the data as a mutable slice.
    pub fn as_slice_mut<D: Datum>(&mut self) -> anyhow::Result<&mut [D]> {
        self.check_dt::<D>()?;
        unsafe { Ok(self.as_slice_mut_unchecked()) }
    }

    pub unsafe fn offset_bytes(&mut self, offset: isize) {
        self.offset_bytes += offset
    }

    pub unsafe fn offset_axis_unchecked(&mut self, axis: usize, pos: isize) {
        let stride = self.strides()[axis] * self.datum_type().size_of() as isize;
        self.offset_bytes(stride * pos)
    }

    pub unsafe fn offset_axis(&mut self, axis: usize, pos: isize) {
        let stride = self.strides()[axis] * self.datum_type().size_of() as isize;
        self.offset_bytes(stride * pos)
    }

    fn offset_for_coords(&self, coords: &[usize]) -> isize {
        self.strides()
            .iter()
            .zip(coords.as_ref())
            .map(|(s, c)| *s as isize * *c as isize)
            .sum::<isize>()
    }

    pub unsafe fn at_unchecked<T: Datum>(&self, coords: impl AsRef<[usize]>) -> &T {
        self.as_ptr_unchecked::<T>()
            .offset(self.offset_for_coords(coords.as_ref()))
            .as_ref()
            .unwrap()
    }

    pub unsafe fn at_mut_unchecked<T: Datum>(&mut self, coords: impl AsRef<[usize]>) -> &mut T {
        self.as_ptr_mut_unchecked::<T>()
            .offset(self.offset_for_coords(coords.as_ref()))
            .as_mut()
            .unwrap()
    }

    pub fn at<T: Datum>(&self, coords: impl AsRef<[usize]>) -> anyhow::Result<&T> {
        self.check_dt::<T>()?;
        let coords = coords.as_ref();
        self.check_coords(coords)?;
        unsafe { Ok(self.at_unchecked(coords)) }
    }

    pub fn at_mut<T: Datum>(&mut self, coords: impl AsRef<[usize]>) -> anyhow::Result<&mut T> {
        self.check_dt::<T>()?;
        let coords = coords.as_ref();
        self.check_coords(coords)?;
        unsafe { Ok(self.at_mut_unchecked(coords)) }
    }

    /*
      pub unsafe fn reshaped(&self, shape: impl AsRef<[usize]>) -> TensorView<'a> {
      let shape = shape.as_ref();
      let mut strides: TVec<isize> = shape
      .iter()
      .rev()
      .scan(1, |state, d| {
      let old = *state;
    *state = *state * d;
    Some(old as isize)
    })
    .collect();
    strides.reverse();
    TensorView { shape: shape.into(), strides, ..*self }
    }
    */
}
