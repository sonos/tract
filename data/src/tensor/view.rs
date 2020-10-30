use crate::tensor::*;

#[derive(Debug)]
pub struct TensorView<'a> {
    tensor: &'a Tensor,
    offset: usize,
    prefix_len: usize,
}

impl<'a> TensorView<'a> {
    pub fn at_prefix(tensor: &'a Tensor, prefix: &[usize]) -> anyhow::Result<TensorView<'a>> {
        anyhow::ensure!(prefix.len() <= tensor.rank(), "prefix longer than tensor shape");
        anyhow::ensure!(prefix.iter().zip(tensor.shape()).all(|(p, d)| p < d), "prefix invalid");
        let offset = prefix.iter().zip(tensor.strides()).map(|(a, b)| a * b).sum();
        Ok(TensorView { tensor, prefix_len: prefix.len(), offset })
    }

    pub fn len(&self) -> usize {
        self.tensor.shape.iter().skip(self.prefix_len).product::<usize>()
    }

    /// Transform the data as a `ndarray::Array`.
    pub fn to_array_view<D: Datum>(&'a self) -> anyhow::Result<ArrayViewD<'a, D>> {
        self.tensor.check_for_access::<D>()?;
        unsafe { Ok(self.to_array_view_unchecked()) }
    }

    /// Transform the data as a `ndarray::Array`.
    pub unsafe fn to_array_view_unchecked<D: Datum>(&'a self) -> ArrayViewD<'a, D> {
        if self.len() != 0 {
            ArrayViewD::from_shape_ptr(&*self.shape(), self.as_ptr_unchecked())
        } else {
            ArrayViewD::from_shape(&*self.shape(), &[]).unwrap()
        }
    }

    pub fn datum_type(&self) -> DatumType {
        self.tensor.datum_type()
    }

    pub fn rank(&self) -> usize {
        self.tensor.rank() - self.prefix_len
    }

    pub fn shape(&self) -> &[usize] {
        &self.tensor.shape()[self.prefix_len..]
    }

    /// Access the data as a pointer.
    pub fn as_ptr<D: Datum>(&self) -> anyhow::Result<*const D> {
        self.tensor.check_for_access::<D>()?;
        Ok(unsafe { self.as_ptr_unchecked() })
    }

    /// Access the data as a pointer.
    pub unsafe fn as_ptr_unchecked<D: Datum>(&self) -> *const D {
        (self.tensor.data as *const u8)
            .offset(self.offset as isize * self.tensor.datum_type().size_of() as isize)
            as *const D
    }

    /// Access the data as a slice.
    pub unsafe fn as_slice_unchecked<D: Datum>(&self) -> &[D] {
        std::slice::from_raw_parts::<D>(self.as_ptr_unchecked(), self.len())
    }

    /// Access the data as a slice.
    pub fn as_slice<D: Datum>(&self) -> anyhow::Result<&[D]> {
        self.tensor.check_for_access::<D>()?;
        unsafe { Ok(self.as_slice_unchecked()) }
    }
}

#[derive(Debug)]
pub struct TensorViewMut<'a> {
    tensor: &'a mut Tensor,
    offset: usize,
    prefix_len: usize,
}

impl<'a> TensorViewMut<'a> {
    pub fn at_prefix(
        tensor: &'a mut Tensor,
        prefix: &[usize],
    ) -> anyhow::Result<TensorViewMut<'a>> {
        anyhow::ensure!(prefix.len() <= tensor.rank(), "prefix longer than tensor shape");
        anyhow::ensure!(prefix.iter().zip(tensor.shape()).all(|(p, d)| p < d), "prefix invalid");
        let offset = prefix.iter().zip(tensor.strides()).map(|(a, b)| a * b).sum();
        Ok(TensorViewMut { tensor, prefix_len: prefix.len(), offset })
    }

    pub fn datum_type(&self) -> DatumType {
        self.tensor.datum_type()
    }

    pub fn shape(&self) -> &[usize] {
        &self.tensor.shape()[self.prefix_len..]
    }

    pub fn len(&self) -> usize {
        self.tensor.shape.iter().skip(self.prefix_len).product::<usize>()
    }

    pub fn rank(&self) -> usize {
        self.shape().len()
    }

    /// Access the data as a pointer.
    pub fn as_ptr<D: Datum>(&self) -> anyhow::Result<*const D> {
        self.tensor.check_for_access::<D>()?;
        Ok(unsafe { self.as_ptr_unchecked() })
    }

    /// Access the data as a pointer.
    pub unsafe fn as_ptr_unchecked<D: Datum>(&self) -> *const D {
        (self.tensor.data as *const u8)
            .offset(self.offset as isize * self.tensor.datum_type().size_of() as isize)
            as *const D
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
        let len = self.shape().iter().product();
        std::slice::from_raw_parts::<D>(self.as_ptr_unchecked(), len)
    }

    /// Access the data as a slice.
    pub fn as_slice<D: Datum>(&self) -> anyhow::Result<&[D]> {
        self.tensor.check_for_access::<D>()?;
        unsafe { Ok(self.as_slice_unchecked()) }
    }

    /// Access the data as a mutable slice.
    pub unsafe fn as_slice_mut_unchecked<D: Datum>(&mut self) -> &mut [D] {
        let len = self.shape().iter().product();
        std::slice::from_raw_parts_mut::<D>(self.as_ptr_mut_unchecked(), len)
    }

    /// Access the data as a mutable slice.
    pub fn as_slice_mut<D: Datum>(&mut self) -> anyhow::Result<&mut [D]> {
        self.tensor.check_for_access::<D>()?;
        unsafe { Ok(self.as_slice_mut_unchecked()) }
    }
}
