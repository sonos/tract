use super::*;
use crate::internal::*;

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
        }
    }

    pub fn offsetting(tensor: &'a Tensor, coords: &[usize]) -> TractResult<TensorView<'a>> {
        ensure!(
            coords.len() == tensor.rank() && coords.iter().zip(tensor.shape()).all(|(p, d)| p < d),
            "Invalid coords {:?} for shape {:?}",
            coords,
            tensor.shape()
        );
        unsafe { Ok(Self::offsetting_unchecked(tensor, coords)) }
    }

    pub unsafe fn offsetting_unchecked(tensor: &'a Tensor, coords: &[usize]) -> TensorView<'a> {
        let offset_bytes =
            coords.iter().zip(tensor.strides()).map(|(a, b)| *a as isize * b).sum::<isize>()
                * tensor.datum_type().size_of() as isize;
        TensorView {
            tensor,
            offset_bytes,
            indexing: Indexing::Custom {
                shape: &tensor.shape,
                strides: &tensor.strides,
            },
        }
    }

    pub fn at_prefix(tensor: &'a Tensor, prefix: &[usize]) -> TractResult<TensorView<'a>> {
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
        TensorView { tensor, offset_bytes, indexing: Indexing::Prefix(prefix.len()) }
    }

    #[inline]
    pub unsafe fn view(tensor: &'a Tensor) -> TensorView<'a> {
        TensorView { tensor, offset_bytes: 0, indexing: Indexing::Prefix(0) }
    }

    #[inline]
    pub fn datum_type(&self) -> DatumType {
        self.tensor.datum_type()
    }

    #[inline]
    pub fn shape(&self) -> &[usize] {
        match &self.indexing {
            Indexing::Prefix(i) => &self.tensor.shape()[*i..],
            Indexing::Custom { shape, .. } => shape,
        }
    }

    #[inline]
    pub fn strides(&self) -> &[isize] {
        match &self.indexing {
            Indexing::Prefix(i) => &self.tensor.strides()[*i..],
            Indexing::Custom { strides, .. } => strides,
        }
    }

    #[inline]
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        match &self.indexing {
            Indexing::Prefix(i) => {
                if *i == 0 {
                    self.tensor.len()
                } else {
                    self.tensor.strides[*i - 1] as usize
                }
            }
            Indexing::Custom { shape, .. } => shape.iter().product(),
        }
    }

    #[inline]
    #[allow(clippy::len_without_is_empty)]
    pub fn valid_bytes(&self) -> usize {
        self.tensor.data.layout().size() - self.offset_bytes as usize
    }

    #[inline]
    pub fn rank(&self) -> usize {
        match &self.indexing {
            Indexing::Prefix(i) => self.tensor.rank() - i,
            Indexing::Custom { shape, .. } => shape.len(),
        }
    }

    fn check_dt<D: Datum>(&self) -> TractResult<()> {
        self.tensor.check_for_access::<D>()
    }

    fn check_coords(&self, coords: &[usize]) -> TractResult<()> {
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
    #[inline]
    pub fn as_ptr<D: Datum>(&self) -> TractResult<*const D> {
        self.check_dt::<D>()?;
        Ok(unsafe { self.as_ptr_unchecked() })
    }

    /// Access the data as a pointer.
    #[inline]
    pub unsafe fn as_ptr_unchecked<D: Datum>(&self) -> *const D {
        self.tensor.as_ptr_unchecked::<u8>().offset(self.offset_bytes) as *const D
    }

    /// Access the data as a pointer.
    #[inline]
    pub unsafe fn as_ptr_mut_unchecked<D: Datum>(&mut self) -> *mut D {
        self.as_ptr_unchecked::<D>() as *mut D
    }

    /// Access the data as a mutable pointer.
    #[inline]
    pub fn as_ptr_mut<D: Datum>(&mut self) -> TractResult<*mut D> {
        Ok(self.as_ptr::<D>()? as *mut D)
    }

    /// Access the data as a slice.
    #[inline]
    pub unsafe fn as_slice_unchecked<D: Datum>(&self) -> &'a [D] {
        std::slice::from_raw_parts::<D>(self.as_ptr_unchecked(), self.len())
    }

    /// Access the data as a slice.
    #[inline]
    pub fn as_slice<D: Datum>(&self) -> TractResult<&'a [D]> {
        self.check_dt::<D>()?;
        unsafe { Ok(self.as_slice_unchecked()) }
    }

    /// Access the data as a mutable slice.
    #[inline]
    pub unsafe fn as_slice_mut_unchecked<D: Datum>(&mut self) -> &mut [D] {
        std::slice::from_raw_parts_mut::<D>(self.as_ptr_mut_unchecked(), self.len())
    }

    /// Access the data as a mutable slice.
    #[inline]
    pub fn as_slice_mut<D: Datum>(&mut self) -> TractResult<&mut [D]> {
        self.check_dt::<D>()?;
        unsafe { Ok(self.as_slice_mut_unchecked()) }
    }

    #[inline]
    pub unsafe fn offset_bytes(&mut self, offset: isize) {
        self.offset_bytes += offset
    }

    #[inline]
    pub unsafe fn offset_axis_unchecked(&mut self, axis: usize, pos: isize) {
        let stride = self.strides()[axis] * self.datum_type().size_of() as isize;
        self.offset_bytes(stride * pos)
    }

    #[inline]
    pub unsafe fn offset_axis(&mut self, axis: usize, pos: isize) {
        let stride = self.strides()[axis] * self.datum_type().size_of() as isize;
        self.offset_bytes(stride * pos)
    }

    #[inline]
    fn offset_for_coords(&self, coords: &[usize]) -> isize {
        self.strides().iter().zip(coords.as_ref()).map(|(s, c)| *s * *c as isize).sum::<isize>()
    }

    #[inline]
    pub unsafe fn at_unchecked<T: Datum>(&self, coords: impl AsRef<[usize]>) -> &T {
        self.as_ptr_unchecked::<T>()
            .offset(self.offset_for_coords(coords.as_ref()))
            .as_ref()
            .unwrap()
    }

    #[inline]
    pub unsafe fn at_mut_unchecked<T: Datum>(&mut self, coords: impl AsRef<[usize]>) -> &mut T {
        self.as_ptr_mut_unchecked::<T>()
            .offset(self.offset_for_coords(coords.as_ref()))
            .as_mut()
            .unwrap()
    }

    #[inline]
    pub fn at<T: Datum>(&self, coords: impl AsRef<[usize]>) -> TractResult<&T> {
        self.check_dt::<T>()?;
        let coords = coords.as_ref();
        self.check_coords(coords)?;
        unsafe { Ok(self.at_unchecked(coords)) }
    }

    #[inline]
    pub fn at_mut<T: Datum>(&mut self, coords: impl AsRef<[usize]>) -> TractResult<&mut T> {
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

#[cfg(test)]
mod test {
    use crate::prelude::Tensor;
    use super::TensorView;

    #[test]
    fn test_at_prefix() {
        let a = Tensor::from_shape(&[2, 2], &[1, 2, 3, 4]).unwrap();
        let a_view = TensorView::at_prefix(&a, &[1]).unwrap();
        assert_eq!(a_view.shape(), &[2]);
        assert_eq!(a_view.as_slice::<i32>().unwrap(), &[3, 4]);


    }
}
