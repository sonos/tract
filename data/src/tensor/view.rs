use crate::tensor::*;

pub struct TensorView<'a> {
    tensor: &'a Tensor,
    offset: usize,
    prefix_len: usize,
}

impl<'a> TensorView<'a> {
    pub unsafe fn at_prefix(tensor: &'a Tensor, prefix: &[usize]) -> TensorView<'a> {
        let offset = prefix.iter().zip(tensor.strides()).map(|(a, b)| a * b).sum();
        TensorView { tensor, prefix_len: prefix.len(), offset }
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
}

pub struct TensorViewMut<'a> {
    tensor: &'a mut Tensor,
    offset: usize,
    prefix_len: usize,
}

impl<'a> TensorViewMut<'a> {
    pub unsafe fn at_prefix(tensor: &'a mut Tensor, prefix: &[usize]) -> TensorViewMut<'a> {
        let offset = prefix.iter().zip(tensor.strides()).map(|(a, b)| a * b).sum();
        TensorViewMut { tensor, prefix_len: prefix.len(), offset }
    }

    pub fn datum_type(&self) -> DatumType {
        self.tensor.datum_type()
    }

    pub fn shape(&self) -> &[usize] {
        &self.tensor.shape()[self.prefix_len..]
    }

    pub fn rank(&self) -> usize {
        self.tensor.rank() - self.prefix_len
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
        self.as_ptr::<D>().map(|p| p as *mut D)
    }
}
