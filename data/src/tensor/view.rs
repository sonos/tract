use crate::tensor::*;
use std::marker::PhantomData;

#[derive(Debug)]
pub struct TensorView<'a> {
    datum_type: DatumType,
    data: *const u8,
    shape: TVec<usize>,
    strides: TVec<isize>,
    boo: PhantomData<&'a ()>,
}

impl<'a> TensorView<'a> {
    pub unsafe fn from_bytes(
        datum_type: DatumType,
        data: &'a mut [u8],
        shape: &[usize],
        strides: &[isize],
    ) -> TensorView<'a> {
        TensorView {
            datum_type,
            data: data.as_ptr(),
            shape: shape.into(),
            strides: strides.into(),
            boo: PhantomData,
        }
    }

    pub fn at_prefix(
        tensor: &'a Tensor,
        prefix: &[usize],
    ) -> anyhow::Result<TensorView<'a>> {
        anyhow::ensure!(prefix.len() <= tensor.rank(), "prefix longer than tensor shape");
        anyhow::ensure!(prefix.iter().zip(tensor.shape()).all(|(p, d)| p < d), "prefix invalid");
        unsafe {
            let datum_type = tensor.datum_type();
            let offset = prefix.iter().zip(tensor.strides()).map(|(a, b)| a * b).sum::<usize>()
                * datum_type.size_of();
            let data = (tensor.as_ptr_unchecked() as *const u8).offset(offset as isize);
            let shape = tensor.shape().iter().skip(prefix.len()).copied().collect();
            let strides = tensor.strides().iter().skip(prefix.len()).map(|&d| d as isize).collect();
            Ok(TensorView { datum_type, data, shape, strides, boo: PhantomData })
        }
    }

    pub fn datum_type(&self) -> DatumType {
        self.datum_type
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn len(&self) -> usize {
        self.shape.iter().product::<usize>()
    }

    pub fn rank(&self) -> usize {
        self.shape().len()
    }

    fn check_for_access<D: Datum>(&self) -> anyhow::Result<()> {
        if self.datum_type() != D::datum_type() {
            anyhow::bail!(
                "TensorView datum type error: tensor is {:?}, accessed as {:?}",
                self.datum_type(),
                D::datum_type(),
            );
        }
        Ok(())
    }

    /// Access the data as a pointer.
    pub fn as_ptr<D: Datum>(&self) -> anyhow::Result<*const D> {
        self.check_for_access::<D>()?;
        Ok(unsafe { self.as_ptr_unchecked() })
    }

    /// Access the data as a pointer.
    pub unsafe fn as_ptr_unchecked<D: Datum>(&self) -> *const D {
        self.data as *const D
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
        self.check_for_access::<D>()?;
        unsafe { Ok(self.as_slice_unchecked()) }
    }

    /// Access the data as a mutable slice.
    pub unsafe fn as_slice_mut_unchecked<D: Datum>(&mut self) -> &mut [D] {
        std::slice::from_raw_parts_mut::<D>(self.as_ptr_mut_unchecked(), self.len())
    }

    /// Access the data as a mutable slice.
    pub fn as_slice_mut<D: Datum>(&mut self) -> anyhow::Result<&mut [D]> {
        self.check_for_access::<D>()?;
        unsafe { Ok(self.as_slice_mut_unchecked()) }
    }

    pub unsafe fn offset_bytes(&mut self, offset: isize) {
        self.data = self.data.offset(offset)
    }

    pub unsafe fn offset_axis(&mut self, axis: usize, pos: isize) {
        let stride = self.strides[axis] * self.datum_type().size_of() as isize;
        self.offset_bytes(stride * pos)
    }
}
