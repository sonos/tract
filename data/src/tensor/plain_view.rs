use std::alloc::Layout;

use ndarray::prelude::*;

use crate::datum::{Datum, DatumType};
use crate::internal::*;
use crate::tensor::Tensor;

use super::storage::PlainStorage;

fn check_for_access<D: Datum>(dt: DatumType) -> TractResult<()> {
    ensure!(
        dt.unquantized() == D::datum_type().unquantized(),
        "Tensor datum type error: tensor is {:?}, accessed as {:?}",
        dt,
        D::datum_type(),
    );
    Ok(())
}

/// Immutable view into a [`Tensor`] verified to have plain storage.
///
/// Construction is the single point of failure (`Tensor::as_plain()` returns
/// `Option`). Once constructed, all data access is infallible with no
/// `unwrap()`/`expect()` on the plain codepath.
pub struct PlainView<'a> {
    tensor: &'a Tensor,
    storage: &'a PlainStorage,
}

impl<'a> PlainView<'a> {
    /// Private constructor used by `Tensor::as_plain()`.
    #[inline]
    pub(crate) fn new(tensor: &'a Tensor, storage: &'a PlainStorage) -> Self {
        PlainView { tensor, storage }
    }

    // -- Metadata (delegated to tensor) --

    #[inline]
    pub fn tensor(&self) -> &Tensor {
        self.tensor
    }

    #[inline]
    pub fn datum_type(&self) -> DatumType {
        self.tensor.datum_type()
    }

    #[inline]
    pub fn shape(&self) -> &[usize] {
        self.tensor.shape()
    }

    #[inline]
    pub fn strides(&self) -> &[isize] {
        self.tensor.strides()
    }

    #[inline]
    pub fn rank(&self) -> usize {
        self.tensor.rank()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.tensor.len()
    }

    // -- Plain-specific (direct storage access, no dispatch) --

    #[inline]
    pub fn as_bytes(&self) -> &'a [u8] {
        self.storage.as_bytes()
    }

    #[inline]
    pub fn layout(&self) -> &Layout {
        self.storage.layout()
    }

    // -- Typed access --
    // TractResult is for datum-type check only, NOT plain check.

    #[inline]
    pub fn as_ptr<D: Datum>(&self) -> TractResult<*const D> {
        check_for_access::<D>(self.datum_type())?;
        unsafe { Ok(self.as_ptr_unchecked()) }
    }

    #[inline]
    pub unsafe fn as_ptr_unchecked<D: Datum>(&self) -> *const D {
        self.storage.as_ptr() as *const D
    }

    #[inline]
    pub fn as_slice<D: Datum>(&self) -> TractResult<&'a [D]> {
        check_for_access::<D>(self.datum_type())?;
        unsafe { Ok(self.as_slice_unchecked()) }
    }

    #[inline]
    pub unsafe fn as_slice_unchecked<D: Datum>(&self) -> &'a [D] {
        if self.storage.is_empty() {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(self.as_ptr_unchecked(), self.len()) }
        }
    }

    #[inline]
    pub fn to_scalar<D: Datum>(&self) -> TractResult<&'a D> {
        check_for_access::<D>(self.datum_type())?;
        unsafe { Ok(self.to_scalar_unchecked()) }
    }

    #[inline]
    pub unsafe fn to_scalar_unchecked<D: Datum>(&self) -> &'a D {
        unsafe { &*(self.storage.as_ptr() as *const D) }
    }

    #[inline]
    pub fn to_array_view<D: Datum>(&self) -> TractResult<ArrayViewD<'a, D>> {
        check_for_access::<D>(self.datum_type())?;
        unsafe { Ok(self.to_array_view_unchecked()) }
    }

    #[inline]
    pub unsafe fn to_array_view_unchecked<D: Datum>(&self) -> ArrayViewD<'a, D> {
        if self.len() != 0 {
            unsafe { ArrayViewD::from_shape_ptr(self.shape(), self.storage.as_ptr() as *const D) }
        } else {
            ArrayViewD::from_shape(self.shape(), &[]).unwrap()
        }
    }
}

/// Mutable view into a [`Tensor`] verified to have plain storage.
///
/// Fields are split to satisfy the borrow checker: mutable storage +
/// immutable metadata borrowed from the same Tensor.
pub struct PlainViewMut<'a> {
    dt: DatumType,
    shape: &'a [usize],
    strides: &'a [isize],
    len: usize,
    storage: &'a mut PlainStorage,
}

impl<'a> PlainViewMut<'a> {
    /// Private constructor used by `Tensor::as_plain_mut()`.
    #[inline]
    pub(crate) fn new(
        dt: DatumType,
        shape: &'a [usize],
        strides: &'a [isize],
        len: usize,
        storage: &'a mut PlainStorage,
    ) -> Self {
        PlainViewMut { dt, shape, strides, len, storage }
    }

    // -- Metadata --

    #[inline]
    pub fn datum_type(&self) -> DatumType {
        self.dt
    }

    #[inline]
    pub fn shape(&self) -> &[usize] {
        self.shape
    }

    #[inline]
    pub fn strides(&self) -> &[isize] {
        self.strides
    }

    #[inline]
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    // -- Read access (same as PlainView, self.storage reborrows as &PlainStorage) --

    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        self.storage.as_bytes()
    }

    #[inline]
    pub fn layout(&self) -> &Layout {
        self.storage.layout()
    }

    #[inline]
    pub fn as_ptr<D: Datum>(&self) -> TractResult<*const D> {
        check_for_access::<D>(self.dt)?;
        unsafe { Ok(self.as_ptr_unchecked()) }
    }

    #[inline]
    pub unsafe fn as_ptr_unchecked<D: Datum>(&self) -> *const D {
        self.storage.as_ptr() as *const D
    }

    #[inline]
    pub fn as_slice<D: Datum>(&self) -> TractResult<&[D]> {
        check_for_access::<D>(self.dt)?;
        unsafe { Ok(self.as_slice_unchecked()) }
    }

    #[inline]
    pub unsafe fn as_slice_unchecked<D: Datum>(&self) -> &[D] {
        if self.storage.is_empty() {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(self.as_ptr_unchecked(), self.len) }
        }
    }

    #[inline]
    pub fn to_scalar<D: Datum>(&self) -> TractResult<&D> {
        check_for_access::<D>(self.dt)?;
        unsafe { Ok(self.to_scalar_unchecked()) }
    }

    #[inline]
    pub unsafe fn to_scalar_unchecked<D: Datum>(&self) -> &D {
        unsafe { &*(self.storage.as_ptr() as *const D) }
    }

    #[inline]
    pub fn to_array_view<D: Datum>(&self) -> TractResult<ArrayViewD<'_, D>> {
        check_for_access::<D>(self.dt)?;
        unsafe { Ok(self.to_array_view_unchecked()) }
    }

    #[inline]
    pub unsafe fn to_array_view_unchecked<D: Datum>(&self) -> ArrayViewD<'_, D> {
        if self.len != 0 {
            unsafe { ArrayViewD::from_shape_ptr(self.shape, self.storage.as_ptr() as *const D) }
        } else {
            ArrayViewD::from_shape(self.shape, &[]).unwrap()
        }
    }

    // -- Mutable access --

    #[inline]
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        self.storage.as_bytes_mut()
    }

    #[inline]
    pub fn as_ptr_mut<D: Datum>(&mut self) -> TractResult<*mut D> {
        check_for_access::<D>(self.dt)?;
        unsafe { Ok(self.as_ptr_mut_unchecked()) }
    }

    #[inline]
    pub unsafe fn as_ptr_mut_unchecked<D: Datum>(&mut self) -> *mut D {
        self.storage.as_mut_ptr() as *mut D
    }

    #[inline]
    pub fn as_slice_mut<D: Datum>(&mut self) -> TractResult<&mut [D]> {
        check_for_access::<D>(self.dt)?;
        unsafe { Ok(self.as_slice_mut_unchecked()) }
    }

    #[inline]
    pub unsafe fn as_slice_mut_unchecked<D: Datum>(&mut self) -> &mut [D] {
        if self.storage.is_empty() {
            &mut []
        } else {
            let len = self.len;
            unsafe { std::slice::from_raw_parts_mut(self.as_ptr_mut_unchecked(), len) }
        }
    }

    #[inline]
    pub fn to_scalar_mut<D: Datum>(&mut self) -> TractResult<&mut D> {
        check_for_access::<D>(self.dt)?;
        unsafe { Ok(self.to_scalar_mut_unchecked()) }
    }

    #[inline]
    pub unsafe fn to_scalar_mut_unchecked<D: Datum>(&mut self) -> &mut D {
        unsafe { &mut *(self.storage.as_mut_ptr() as *mut D) }
    }

    #[inline]
    pub fn to_array_view_mut<D: Datum>(&mut self) -> TractResult<ArrayViewMutD<'_, D>> {
        check_for_access::<D>(self.dt)?;
        unsafe { Ok(self.to_array_view_mut_unchecked()) }
    }

    #[inline]
    pub unsafe fn to_array_view_mut_unchecked<D: Datum>(&mut self) -> ArrayViewMutD<'_, D> {
        if self.len != 0 {
            unsafe {
                ArrayViewMutD::from_shape_ptr(self.shape, self.storage.as_mut_ptr() as *mut D)
            }
        } else {
            ArrayViewMutD::from_shape(self.shape, &mut []).unwrap()
        }
    }
}
