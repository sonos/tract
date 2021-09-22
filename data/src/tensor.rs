//! `Tensor`, tract main data object of interest.
use crate::datum::{round_ties_to_even, scale_by, Blob, ClampCast, Datum, DatumType, QParams};
use crate::dim::TDim;
use crate::f16::f16;
use crate::TVec;
use itertools::Itertools;
use ndarray::prelude::*;
use num_complex::Complex;
#[cfg(feature = "serialize")]
use serde::ser::{Serialize, Serializer};
use std::alloc;
use std::borrow::Cow;
use std::fmt;
use std::hash::Hash;
use std::mem::{align_of, size_of};
use std::ops::Range;
use std::sync::Arc;

pub mod litteral;
pub mod view;

/// Tensor is a concrete tensor in tract.
pub struct Tensor {
    dt: DatumType,
    shape: TVec<usize>,
    strides: TVec<isize>,
    len: usize,
    layout: alloc::Layout,
    data: *mut u8,
}

unsafe impl Send for Tensor {}
unsafe impl Sync for Tensor {}

impl Hash for Tensor {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        use DatumType::*;
        self.dt.hash(state);
        self.shape.hash(state);
        self.layout.align().hash(state);
        unsafe {
            match self.dt {
                Bool => self.as_slice_unchecked::<bool>().hash(state),
                I8 => self.as_slice_unchecked::<i8>().hash(state),
                I16 => self.as_slice_unchecked::<i16>().hash(state),
                I32 => self.as_slice_unchecked::<i32>().hash(state),
                I64 => self.as_slice_unchecked::<i64>().hash(state),
                U8 => self.as_slice_unchecked::<u8>().hash(state),
                U16 => self.as_slice_unchecked::<u16>().hash(state),
                U32 => self.as_slice_unchecked::<u32>().hash(state),
                U64 => self.as_slice_unchecked::<u64>().hash(state),
                F16 => self.as_slice_unchecked::<i16>().hash(state),
                F32 => self.as_slice_unchecked::<i32>().hash(state),
                F64 => self.as_slice_unchecked::<i64>().hash(state),
                TDim => self.as_slice_unchecked::<crate::dim::TDim>().hash(state),
                String => self.as_slice_unchecked::<std::string::String>().hash(state),
                Blob => self.as_slice_unchecked::<crate::datum::Blob>().hash(state),
                QI8(_) => self.as_slice_unchecked::<i8>().hash(state),
                QU8(_) => self.as_slice_unchecked::<u8>().hash(state),
                ComplexI16 => self.as_slice_unchecked::<Complex<i16>>().hash(state),
                ComplexI32 => self.as_slice_unchecked::<Complex<i32>>().hash(state),
                ComplexI64 => self.as_slice_unchecked::<Complex<i64>>().hash(state),
                ComplexF16 => self.as_slice_unchecked::<Complex<i16>>().hash(state),
                ComplexF32 => self.as_slice_unchecked::<Complex<i32>>().hash(state),
                ComplexF64 => self.as_slice_unchecked::<Complex<i64>>().hash(state),
            }
        }
    }
}

impl Clone for Tensor {
    fn clone(&self) -> Tensor {
        self.deep_clone()
    }
}

impl Default for Tensor {
    fn default() -> Tensor {
        litteral::tensor0(0f32)
    }
}

impl Drop for Tensor {
    fn drop(&mut self) {
        if self.dt == DatumType::Blob {
            unsafe {
                self.as_slice_mut::<Blob>()
                    .unwrap()
                    .iter_mut()
                    .for_each(|s| std::ptr::drop_in_place(s as *mut Blob));
            }
        }
        if self.dt == DatumType::String {
            unsafe {
                self.as_slice_mut::<String>()
                    .unwrap()
                    .iter_mut()
                    .for_each(|s| std::ptr::drop_in_place(s as *mut String));
            }
        }
        if self.dt == DatumType::TDim {
            unsafe {
                self.as_slice_mut::<TDim>()
                    .unwrap()
                    .iter_mut()
                    .for_each(|s| std::ptr::drop_in_place(s as *mut TDim));
            }
        }
        if !self.data.is_null() && self.layout.size() > 0 {
            unsafe { alloc::dealloc(self.data, self.layout) }
        }
    }
}

impl Tensor {
    /// Create an uninitialized tensor (dt as type paramater).
    pub unsafe fn uninitialized<T: Datum>(shape: &[usize]) -> anyhow::Result<Tensor> {
        Self::uninitialized_dt(T::datum_type(), shape)
    }

    /// Create an uninitialized tensor (dt as regular parameter).
    pub unsafe fn uninitialized_dt(dt: DatumType, shape: &[usize]) -> anyhow::Result<Tensor> {
        Self::uninitialized_aligned_dt(dt, shape, dt.alignment())
    }

    /// Create an uninitialized tensor with a given alignment (in bytes).
    pub unsafe fn uninitialized_aligned<T: Datum>(
        shape: &[usize],
        alignment: usize,
    ) -> anyhow::Result<Tensor> {
        Self::uninitialized_aligned_dt(T::datum_type(), shape, alignment)
    }

    /// Create an uninitialized tensor with a given alignment (in bytes).
    pub unsafe fn uninitialized_aligned_dt(
        dt: DatumType,
        shape: &[usize],
        alignment: usize,
    ) -> anyhow::Result<Tensor> {
        if dt == String::datum_type() {
            return Ok(ndarray::ArrayD::<String>::default(shape).into());
        } else if dt == Blob::datum_type() {
            return Ok(ndarray::ArrayD::<Blob>::default(shape).into());
        } else if dt == TDim::datum_type() {
            return Ok(ndarray::ArrayD::<TDim>::default(shape).into());
        }
        assert!(dt.is_copy());
        let bytes = shape.iter().cloned().product::<usize>() * dt.size_of();
        let layout = alloc::Layout::from_size_align(bytes, alignment)?;
        let data = if bytes == 0 {
            std::ptr::null()
        } else {
            let ptr = alloc::alloc(layout);
            assert!(!ptr.is_null());
            ptr
        } as *mut u8;
        let mut tensor = Tensor { strides: tvec!(), layout, dt, shape: shape.into(), data, len: 0 };
        #[cfg(debug_assertions)]
        {
            if dt == DatumType::F32 {
                tensor.as_slice_mut_unchecked::<f32>().iter_mut().for_each(|f| *f = std::f32::NAN)
            } else if dt == DatumType::I32 {
                tensor.as_slice_mut_unchecked::<i32>().iter_mut().for_each(|f| *f = std::i32::MIN)
            }
        }
        tensor.update_strides_and_len();
        Ok(tensor)
    }

    pub fn stack_tensors(
        axis: usize,
        tensors: &[impl std::borrow::Borrow<Tensor>],
    ) -> anyhow::Result<Tensor> {
        use crate::datum::ArrayDatum;
        let dt = tensors[0].borrow().datum_type();
        if tensors.iter().any(|t| t.borrow().datum_type() != dt) {
            anyhow::bail!("Inconsistent datum type in stack.")
        }
        // map all copy types to the i* of the same size
        let mut tensor = unsafe {
            match dt {
                DatumType::F16 => i16::stack_tensors(axis, &tensors),
                DatumType::F32 => i32::stack_tensors(axis, &tensors),
                DatumType::F64 => i64::stack_tensors(axis, &tensors),
                DatumType::Bool => i8::stack_tensors(axis, &tensors),
                DatumType::U8 => i8::stack_tensors(axis, &tensors),
                DatumType::U16 => i16::stack_tensors(axis, &tensors),
                DatumType::U32 => i32::stack_tensors(axis, &tensors),
                DatumType::U64 => i64::stack_tensors(axis, &tensors),
                DatumType::I8 => i8::stack_tensors(axis, &tensors),
                DatumType::I16 => i16::stack_tensors(axis, &tensors),
                DatumType::I32 => i32::stack_tensors(axis, &tensors),
                DatumType::I64 => i64::stack_tensors(axis, &tensors),
                DatumType::TDim => TDim::stack_tensors(axis, &tensors),
                DatumType::Blob => Blob::stack_tensors(axis, &tensors),
                DatumType::String => String::stack_tensors(axis, &tensors),
                DatumType::QI8(_) => i8::stack_tensors(axis, &tensors),
                DatumType::QU8(_) => i8::stack_tensors(axis, &tensors),
                DatumType::ComplexI16 => Complex::<i16>::stack_tensors(axis, &tensors),
                DatumType::ComplexI32 => Complex::<i32>::stack_tensors(axis, &tensors),
                DatumType::ComplexI64 => Complex::<i64>::stack_tensors(axis, &tensors),
                DatumType::ComplexF16 => Complex::<i16>::stack_tensors(axis, &tensors),
                DatumType::ComplexF32 => Complex::<i32>::stack_tensors(axis, &tensors),
                DatumType::ComplexF64 => Complex::<i64>::stack_tensors(axis, &tensors),
            }
        }?;
        tensor.dt = dt;
        Ok(tensor)
    }

    pub unsafe fn clear<T: Datum + num_traits::Zero>(&mut self) {
        self.as_slice_mut_unchecked::<T>().iter_mut().for_each(|item| *item = T::zero());
    }
    //FIXME : zero for quantised dt ?
    pub fn zero<T: Datum + num_traits::Zero>(shape: &[usize]) -> anyhow::Result<Tensor> {
        unsafe {
            let mut t = Tensor::uninitialized::<T>(shape)?;
            t.clear::<T>();
            Ok(t)
        }
    }

    pub fn zero_scalar<T: Datum + num_traits::Zero>() -> anyhow::Result<Tensor> {
        Tensor::zero::<T>(&[])
    }

    pub fn zero_scalar_dt(dt: DatumType) -> anyhow::Result<Tensor> {
        dispatch_numbers!(Self::zero_scalar(dt)())
    }

    pub fn zero_dt(dt: DatumType, shape: &[usize]) -> anyhow::Result<Tensor> {
        dispatch_numbers!(Self::zero(dt)(shape))
    }

    pub fn zero_aligned_dt(
        dt: DatumType,
        shape: &[usize],
        alignment: usize,
    ) -> anyhow::Result<Tensor> {
        dispatch_numbers!(Self::zero_aligned(dt)(shape, alignment))
    }

    pub fn zero_aligned<T: Datum + num_traits::Zero>(
        shape: &[usize],
        alignment: usize,
    ) -> anyhow::Result<Tensor> {
        unsafe {
            let mut tensor = Self::uninitialized_aligned::<T>(shape, alignment)?;
            tensor.clear::<T>();
            Ok(tensor)
        }
    }

    /// Create a tensor with a given shape and a slice of elements. 
    /// The data is copied and aligned to size of T. 
    pub fn from_shape<T: Datum + Copy>(shape: &[usize], data: &[T]) -> anyhow::Result<Tensor> {
        let dt = T::datum_type();
        Self::from_shape_align(shape, data, dt.alignment())
    }

    /// Create a tensor with a given shape and a slice of elements. 
    /// The data is copied and aligned to given alignment. 
    pub fn from_shape_align<T: Datum + Copy>(shape: &[usize], data: &[T], align: usize) -> anyhow::Result<Tensor> {
        anyhow::ensure!(data.len() == shape.iter().product::<usize>(), "Shape product must be equal to data length");
        unsafe {
            let bytes = std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                data.len() * T::datum_type().size_of(),
            );
            let dt = T::datum_type();
            Self::from_raw_dt_align(dt, shape, bytes, align)
        }
    }

    /// Create a tensor from raw data.
    ///
    /// It copies the data, aligning it to the size of T.
    pub unsafe fn from_raw<T: Datum>(shape: &[usize], content: &[u8]) -> anyhow::Result<Tensor> {
        Tensor::from_raw_dt(T::datum_type(), shape, content)
    }

    pub unsafe fn from_raw_aligned<T: Datum>(
        shape: &[usize],
        content: &[u8],
        align: usize,
    ) -> anyhow::Result<Tensor> {
        Tensor::from_raw_dt_align(T::datum_type(), shape, content, align)
    }

    pub unsafe fn from_raw_dt(
        dt: DatumType,
        shape: &[usize],
        content: &[u8],
    ) -> anyhow::Result<Tensor> {
        Self::from_raw_dt_align(dt, shape, content, dt.alignment())
    }

    pub unsafe fn from_raw_dt_align(
        dt: DatumType,
        shape: &[usize],
        content: &[u8],
        align: usize,
    ) -> anyhow::Result<Tensor> {
        let mut tensor = Tensor::uninitialized_aligned_dt(dt, shape, align)?;
        tensor.as_bytes_mut().copy_from_slice(content);
        Ok(tensor)
    }

    pub unsafe fn from_slice_align<T: Datum>(
        content: &[T],
        align: usize,
    ) -> anyhow::Result<Tensor> {
        let bytes = std::slice::from_raw_parts(
            content.as_ptr() as *const u8,
            content.len() * T::datum_type().size_of(),
        );
        Self::from_raw_dt_align(T::datum_type(), &[content.len()], bytes, align)
    }

    /// Get the number of dimensions (or axes) of the tensor.
    #[inline]
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Get the shape of the tensor.
    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the number of valeus in the tensor.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Get the shape of the tensor.
    #[inline]
    pub fn strides(&self) -> &[isize] {
        &self.strides
    }

    fn update_strides_and_len(&mut self) {
        self.strides.clear();
        compute_natural_stride_to(&mut self.strides, &self.shape);
        self.len = if self.rank() == 0 {
            1
        } else {
            unsafe { *self.strides.get_unchecked(0) as usize * self.shape.get_unchecked(0) }
        }
    }

    /// Force the tensor shape, no consistency check.
    pub unsafe fn set_shape_unchecked(&mut self, shape: &[usize]) {
        if shape != &*self.shape {
            self.shape.clear();
            self.shape.extend_from_slice(shape);
            self.update_strides_and_len();
        }
    }

    /// Force the tensor shape.
    pub fn set_shape(&mut self, shape: &[usize]) -> anyhow::Result<()> {
        if self.len() != shape.iter().product::<usize>() {
            anyhow::bail!("Invalid reshape {:?} to {:?}", self.shape, shape);
        }
        unsafe { self.set_shape_unchecked(shape) }
        Ok(())
    }

    pub fn permute_axes(self, axes: &[usize]) -> anyhow::Result<Tensor> {
        unsafe {
            #[inline]
            unsafe fn permute<T: Datum>(axes: &[usize], input: Tensor) -> Tensor {
                input.into_array_unchecked::<T>().permuted_axes(axes).into_tensor()
            }
            let dt = self.datum_type();
            let mut t = dispatch_datum_by_size!(permute(self.datum_type())(axes, self));
            t.set_datum_type(dt);
            Ok(t)
        }
    }

    pub fn move_axis(self, from: usize, to: usize) -> anyhow::Result<Tensor> {
        let mut permutation: Vec<usize> = (0..self.rank()).collect();
        permutation.remove(from);
        permutation.insert(to, from);
        self.permute_axes(&permutation)
    }

    pub fn collapse_axis_with_next(mut self, axis: usize) -> Tensor {
        let removed = self.shape.remove(axis + 1);
        self.shape[axis] *= removed;
        self.update_strides_and_len();
        self
    }

    pub fn split_axis(mut self, axis: usize, outer_dim: usize) -> anyhow::Result<Tensor> {
        if self.shape[axis] % outer_dim != 0 {
            anyhow::bail!(
                "Invalid axis split, shape is {:?}, axis split at {}, outer {}",
                self.shape,
                axis,
                outer_dim
            );
        }
        self.shape.insert(axis + 1, self.shape[axis] / outer_dim);
        self.shape[axis] = outer_dim;
        self.update_strides_and_len();
        Ok(self)
    }

    /// Reshape the tensor to `shape`.
    pub fn into_shape(mut self, shape: &[usize]) -> anyhow::Result<Tensor> {
        self.set_shape(shape)?;
        Ok(self)
    }

    pub fn insert_axis(&mut self, axis: usize) -> anyhow::Result<()> {
        self.shape.insert(axis, 1);
        self.strides.insert(axis, self.strides.get(axis).copied().unwrap_or(1));
        Ok(())
    }

    pub fn remove_axis(&mut self, axis: usize) -> anyhow::Result<()> {
        self.shape.remove(axis);
        self.strides.remove(axis);
        Ok(())
    }

    pub fn broadcast_into_rank(mut self, rank: usize) -> anyhow::Result<Tensor> {
        self.broadcast_to_rank(rank)?;
        self.update_strides_and_len();
        Ok(self)
    }

    pub fn broadcast_to_rank(&mut self, rank: usize) -> anyhow::Result<()> {
        if rank < self.rank() {
            anyhow::bail!("Can only broadcast to higher rank")
        }
        while self.shape.len() < rank {
            self.shape.insert(0, 1)
        }
        self.update_strides_and_len();
        Ok(())
    }

    pub fn broadcast_scalar_to_shape(&self, shape: &[usize]) -> anyhow::Result<Tensor> {
        if self.rank() > 0 {
            anyhow::bail!("broadcast_scalar_to_shape called on {:?}, which is not a salar", self);
        }
        unsafe fn make<T: Datum>(src: &Tensor, dst: &mut Tensor) {
            let value: &T = src.to_scalar_unchecked::<T>();
            dst.as_slice_mut_unchecked::<T>().iter_mut().for_each(|item| *item = value.clone());
        }
        unsafe {
            let mut t = Tensor::uninitialized_dt(self.datum_type(), shape)?;
            dispatch_datum_by_size!(make(self.datum_type())(self, &mut t));
            Ok(t)
        }
    }

    fn clip_range_bounds(
        &self,
        axis: usize,
        range: impl std::ops::RangeBounds<usize>,
    ) -> Range<usize> {
        use std::ops::Bound;
        let start = match range.start_bound() {
            Bound::Included(ix) => *ix,
            Bound::Excluded(ix) => ix + 1,
            Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            Bound::Included(ix) => *ix + 1,
            Bound::Excluded(ix) => *ix,
            Bound::Unbounded => self.shape()[axis],
        };
        start..end
    }

    pub fn assign_slice(
        &mut self,
        range: impl std::ops::RangeBounds<usize>,
        src: &Tensor,
        src_range: impl std::ops::RangeBounds<usize>,
        axis: usize,
    ) -> anyhow::Result<()> {
        let range = self.clip_range_bounds(axis, range);
        let src_range = src.clip_range_bounds(axis, src_range);
        anyhow::ensure!(
            src.datum_type() == self.datum_type(),
            "Attempt to assign into {:?} from {:?}, datum type mismatch",
            self.datum_type(),
            src.datum_type()
        );
        anyhow::ensure!(
            src_range.len() == range.len(),
            "Attempt to assign a range of {:?} from a range of {:?}",
            range,
            src_range,
        );
        anyhow::ensure!(
            self.rank() == src.rank()
                && itertools::izip!(0.., self.shape(), src.shape())
                    .all(|(ix, dst, src)| ix == axis || src == dst),
            "Attempt to assign a {}-axis range of {:?} from a range of {:?}",
            axis,
            self,
            src
        );
        anyhow::ensure!(
            src_range.end <= src.shape()[axis],
            "Assigning from invalid slice (axis {}, {:?}) of {:?}",
            axis,
            src_range,
            src
        );
        anyhow::ensure!(
            range.end <= self.shape()[axis],
            "Assigning to invalid slice (axis {}, {:?}) of {:?}",
            axis,
            range,
            self
        );
        self.assign_slice_from_resolved(range, src, src_range, axis);
        Ok(())
    }

    pub unsafe fn assign_slice_unchecked(
        &mut self,
        range: impl std::ops::RangeBounds<usize>,
        src: &Tensor,
        src_range: impl std::ops::RangeBounds<usize>,
        axis: usize,
    ) {
        let range = self.clip_range_bounds(axis, range);
        let src_range = src.clip_range_bounds(axis, src_range);
        self.assign_slice_from_resolved(range, src, src_range, axis);
    }

    fn assign_slice_from_resolved(
        &mut self,
        range: std::ops::Range<usize>,
        src: &Tensor,
        src_range: std::ops::Range<usize>,
        axis: usize,
    ) {
        use ndarray::Slice;
        unsafe fn assign_slice_t<T: Datum>(
            to: &mut Tensor,
            to_range: Range<usize>,
            from: &Tensor,
            from_range: Range<usize>,
            axis: usize,
        ) {
            to.to_array_view_mut_unchecked::<T>()
                .slice_axis_mut(Axis(axis), Slice::from(to_range))
                .assign(
                    &from
                        .to_array_view_unchecked::<T>()
                        .slice_axis(Axis(axis), Slice::from(from_range)),
                )
        }
        unsafe {
            if axis == 0 && self.datum_type().is_copy() {
                let stride = self.strides[0] as usize * self.datum_type().size_of();
                let dst_start = (stride * range.start) as isize;
                let src_start = (stride * src_range.start) as isize;
                let len = stride * range.len();
                if self.data != src.data {
                    std::ptr::copy_nonoverlapping(
                        src.data.offset(src_start),
                        self.data.offset(dst_start),
                        len,
                    );
                } else {
                    std::ptr::copy(src.data.offset(src_start), self.data.offset(dst_start), len);
                }
            } else {
                dispatch_datum!(assign_slice_t(self.datum_type())(
                    self, range, src, src_range, axis
                ));
            }
        }
    }

    /// Get the datum type of the tensor.
    #[inline]
    pub fn datum_type(&self) -> DatumType {
        self.dt
    }

    /// Set the datum type of the tensor.
    #[inline]
    pub unsafe fn set_datum_type(&mut self, dt: DatumType) {
        self.dt = dt
    }

    /// Dump the tensor in a human readable form.
    ///
    /// `force_full` will force the tensor to be dump in full even if it is big.
    pub fn dump(&self, force_full: bool) -> anyhow::Result<String> {
        unsafe fn dump_t<D: Datum>(tensor: &Tensor, n: usize) -> String {
            tensor.as_slice_unchecked::<D>()[0..n].iter().join(", ")
        }
        unsafe {
            let trunc = self.len() > 12 && !force_full;
            let data = dispatch_datum!(dump_t(self.datum_type())(
                self,
                if trunc { 12 } else { self.len() }
            ));
            Ok(format!(
                "{},{:?} {}{}",
                self.shape.iter().join(","),
                self.dt,
                data,
                if trunc { "..." } else { "" }
            ))
        }
    }

    /// Compare two tensors, allowing for rounding errors.
    pub fn close_enough(&self, other: &Self, approx: bool) -> anyhow::Result<()> {
        if self.shape() != other.shape() {
            anyhow::bail!("Shape mismatch {:?} != {:?}", self.shape(), other.shape())
        }
        if approx {
            let atol = 5e-4;
            let rtol = 1e-4;
            let ma = self.cast_to::<f32>()?;
            let ma = ma.to_array_view::<f32>()?;
            let mb = other.cast_to::<f32>()?;
            let mb = mb.to_array_view::<f32>()?;
            ndarray::indices_of(&ma).into_iter().try_for_each(|indices| {
                let a = ma[&indices];
                let b = mb[&indices];
                if !((a.is_nan() && b.is_nan())
                    || (a.is_infinite() && b.is_infinite() && a.signum() == b.signum())
                    || (a - b).abs() <= atol + rtol * b.abs())
                {
                    anyhow::bail!("Mismatch at {:?} {} != {}", indices.slice(), a, b)
                }
                Ok(())
            })
        } else {
            if self.eq(other) {
                Ok(())
            } else {
                anyhow::bail!("Mismatch")
            }
        }
    }

    /// Transform the tensor into a `ndarray::Array`.
    pub fn into_array<D: Datum>(self) -> anyhow::Result<ArrayD<D>> {
        Ok(self.to_array_view::<D>()?.to_owned())
    }

    /// Transform the tensor into a `ndarray::Array`.
    pub unsafe fn into_array_unchecked<D: Datum>(self) -> ArrayD<D> {
        self.to_array_view_unchecked::<D>().to_owned()
    }

    fn check_for_access<D: Datum>(&self) -> anyhow::Result<()> {
        if self.datum_type().unquantized() != D::datum_type().unquantized() {
            anyhow::bail!(
                "Tensor datum type error: tensor is {:?}, accessed as {:?}",
                self.datum_type(),
                D::datum_type(),
            );
        }
        Ok(())
    }

    /// Transform the data as a `ndarray::Array`.
    pub fn to_array_view<'a, D: Datum>(&'a self) -> anyhow::Result<ArrayViewD<'a, D>> {
        self.check_for_access::<D>()?;
        unsafe { Ok(self.to_array_view_unchecked()) }
    }

    /// Transform the data as a mutable `ndarray::Array`.
    pub fn to_array_view_mut<'a, D: Datum>(&'a mut self) -> anyhow::Result<ArrayViewMutD<'a, D>> {
        self.check_for_access::<D>()?;
        unsafe { Ok(self.to_array_view_mut_unchecked()) }
    }

    /// Transform the data as a `ndarray::Array`.
    pub unsafe fn to_array_view_unchecked<'a, D: Datum>(&'a self) -> ArrayViewD<'a, D> {
        if self.len() != 0 {
            ArrayViewD::from_shape_ptr(&*self.shape, self.data as *const D)
        } else {
            ArrayViewD::from_shape(&*self.shape, &[]).unwrap()
        }
    }

    /// Transform the data as a mutable `ndarray::Array`.
    pub unsafe fn to_array_view_mut_unchecked<'a, D: Datum>(&'a mut self) -> ArrayViewMutD<'a, D> {
        if self.len() != 0 {
            ArrayViewMutD::from_shape_ptr(&*self.shape, self.data as *mut D)
        } else {
            ArrayViewMutD::from_shape(&*self.shape, &mut []).unwrap()
        }
    }

    /// Access the data as a pointer.
    pub fn as_ptr<D: Datum>(&self) -> anyhow::Result<*const D> {
        self.check_for_access::<D>()?;
        Ok(self.data as *const D)
    }

    /// Access the data as a pointer.
    pub unsafe fn as_ptr_unchecked<D: Datum>(&self) -> *const D {
        self.data as *const D
    }

    /// Access the data as a pointer.
    pub unsafe fn as_ptr_mut_unchecked<D: Datum>(&mut self) -> *mut D {
        self.data as *mut D
    }

    /// Access the data as a mutable pointer.
    pub fn as_ptr_mut<D: Datum>(&mut self) -> anyhow::Result<*mut D> {
        self.as_ptr::<D>().map(|p| p as *mut D)
    }

    /// Access the data as a slice.
    pub fn as_slice<D: Datum>(&self) -> anyhow::Result<&[D]> {
        unsafe { Ok(std::slice::from_raw_parts::<D>(self.as_ptr()?, self.len())) }
    }

    /// Access the data as a mutable slice.
    pub fn as_slice_mut<D: Datum>(&mut self) -> anyhow::Result<&mut [D]> {
        unsafe { Ok(std::slice::from_raw_parts_mut::<D>(self.as_ptr_mut()?, self.len())) }
    }

    /// Access the data as a slice.
    pub unsafe fn as_slice_unchecked<D: Datum>(&self) -> &[D] {
        std::slice::from_raw_parts::<D>(self.data as *const D, self.len())
    }

    /// Access the data as a mutable slice.
    pub unsafe fn as_slice_mut_unchecked<D: Datum>(&mut self) -> &mut [D] {
        std::slice::from_raw_parts_mut::<D>(self.data as *mut D, self.len())
    }

    /// Access the data as a scalar.
    pub fn to_scalar<'a, D: Datum>(&'a self) -> anyhow::Result<&D> {
        self.check_for_access::<D>()?;
        if self.len() == 0 {
            anyhow::bail!("to_scalar called on empty tensor ({:?})", self)
        }
        unsafe { Ok(self.to_scalar_unchecked()) }
    }

    /// Access the data as a scalar.
    pub unsafe fn to_scalar_unchecked<'a, D: Datum>(&'a self) -> &D {
        &*(self.data as *mut D)
    }

    pub unsafe fn as_bytes(&self) -> &[u8] {
        std::slice::from_raw_parts(self.data, self.layout.size())
    }

    pub unsafe fn as_bytes_mut(&mut self) -> &mut [u8] {
        std::slice::from_raw_parts_mut(self.data, self.layout.size())
    }

    unsafe fn is_uniform_t<T: Datum>(&self) -> bool {
        let slice = self.as_slice_unchecked::<T>();
        slice[1..].iter().all(|x| x == &slice[0])
    }

    pub fn is_uniform(&self) -> bool {
        if self.len() <= 1 {
            return true;
        }
        unsafe { dispatch_datum!(Tensor::is_uniform_t(self.datum_type())(self)) }
    }

    unsafe fn as_uniform_t<T: Datum>(&self) -> Tensor {
        let v: T = self.as_slice_unchecked::<T>()[0].clone();
        litteral::tensor0(v)
    }

    pub fn as_uniform(&self) -> Option<Tensor> {
        if self.len() >= 1 && self.is_uniform() {
            unsafe { Some(dispatch_datum!(Tensor::as_uniform_t(self.datum_type())(self))) }
        } else {
            None
        }
    }

    unsafe fn natural_cast<
        Source: Datum + num_traits::AsPrimitive<Target>,
        Target: Datum + Copy,
    >(
        &self,
        other: &mut Tensor,
    ) {
        self.as_slice_unchecked::<Source>()
            .iter()
            .zip(other.as_slice_mut_unchecked::<Target>().iter_mut())
            .for_each(|(s, d)| *d = s.as_());
    }

    unsafe fn cast_number_to_bool<Source: Datum + num_traits::Zero>(&self, other: &mut Tensor) {
        self.as_slice_unchecked::<Source>()
            .iter()
            .zip(other.as_slice_mut_unchecked::<bool>().iter_mut())
            .for_each(|(s, d)| *d = !s.is_zero());
    }

    unsafe fn cast_from_string<Target: Datum + core::str::FromStr>(
        &self,
        other: &mut Tensor,
    ) -> anyhow::Result<()> {
        for (s, d) in self
            .as_slice_unchecked::<String>()
            .iter()
            .zip(other.as_slice_mut_unchecked::<Target>().iter_mut())
        {
            *d = s.parse().map_err(|_| {
                anyhow::format_err!("Could not parse {} as {:?}", s, Target::datum_type())
            })?
        }
        Ok(())
    }

    unsafe fn cast_to_string<Source: Datum>(&self, other: &mut Tensor) {
        for (s, d) in self
            .as_slice_unchecked::<Source>()
            .iter()
            .zip(other.as_slice_mut_unchecked::<String>().iter_mut())
        {
            *d = s.to_string()
        }
    }

    /// Optionnaly convert data to a tensor for a new DatumType.
    pub fn cast_to<D: Datum>(&self) -> anyhow::Result<Cow<Tensor>> {
        self.cast_to_dt(D::datum_type())
    }

    /// Optionnaly convert data to a tensor for a new DatumType.
    pub fn cast_to_dt(&self, dst_dt: DatumType) -> anyhow::Result<Cow<Tensor>> {
        unsafe {
            if self.dt == dst_dt {
                return Ok(Cow::Borrowed(self));
            }
            if self.dt == TDim::datum_type() && (dst_dt.is_integer() || dst_dt.is_float()) {
                let slice = self.as_slice_unchecked::<TDim>();
                let mut ints = Self::uninitialized::<i64>(&self.shape)?;
                let ints_slice = ints.as_slice_mut_unchecked::<i64>();
                for i in 0..self.len() {
                    ints_slice[i] = slice[i].to_i64()?;
                }
                return Ok(Cow::Owned(ints.cast_to_dt(dst_dt)?.into_owned()));
            }
            if self.dt == bool::datum_type() && (dst_dt.is_integer() || dst_dt.is_float()) {
                let slice = self.as_slice_unchecked::<bool>();
                let mut ints = Self::uninitialized::<i8>(&self.shape)?;
                let ints_slice = ints.as_slice_mut_unchecked::<i8>();
                for i in 0..self.len() {
                    ints_slice[i] = slice[i] as usize as i8;
                }
                return Ok(Cow::Owned(ints.cast_to_dt(dst_dt)?.into_owned()));
            }
            let mut result = Self::uninitialized_dt(dst_dt, &self.shape)?;
            if self.dt == DatumType::String {
                dispatch_datum!(Self::cast_from_string(dst_dt)(self, &mut result))?;
                return Ok(Cow::Owned(result));
            }
            if dst_dt == DatumType::String {
                dispatch_datum!(Self::cast_to_string(self.dt)(self, &mut result));
                return Ok(Cow::Owned(result));
            }
            macro_rules! n {
                ($source:ty) => {
                    if <$source>::datum_type() == self.datum_type() {
                        match dst_dt {
                            DatumType::I8 => self.natural_cast::<$source, i8>(&mut result),
                            DatumType::I16 => self.natural_cast::<$source, i16>(&mut result),
                            DatumType::I32 => self.natural_cast::<$source, i32>(&mut result),
                            DatumType::I64 => self.natural_cast::<$source, i64>(&mut result),
                            DatumType::U8 => self.natural_cast::<$source, u8>(&mut result),
                            DatumType::U16 => self.natural_cast::<$source, u16>(&mut result),
                            DatumType::U32 => self.natural_cast::<$source, u32>(&mut result),
                            DatumType::U64 => self.natural_cast::<$source, u64>(&mut result),
                            DatumType::F16 => self.natural_cast::<$source, f16>(&mut result),
                            DatumType::F32 => self.natural_cast::<$source, f32>(&mut result),
                            DatumType::F64 => self.natural_cast::<$source, f64>(&mut result),
                            DatumType::TDim => {
                                let ints = self.cast_to::<i32>()?;
                                let slice = ints.as_slice_unchecked::<i32>();
                                let result = result.as_slice_mut_unchecked::<TDim>();
                                for i in 0..self.len() {
                                    result[i] = slice[i].into();
                                }
                            }
                            DatumType::Bool => self.cast_number_to_bool::<$source>(&mut result),
                            _ => todo!(),
                        }
                        return Ok(Cow::Owned(result));
                    };
                };
            }
            //If there is no quantization
            if !dst_dt.is_quantized() && !self.datum_type().is_quantized() {
                n!(u8);
                n!(u16);
                n!(u32);
                n!(u64);
                n!(i8);
                n!(i16);
                n!(i32);
                n!(i64);
                n!(f16);
                n!(f32);
                n!(f64);
            } else {
                let (s_zp, s_scale) = self.datum_type().zp_scale();
                let (d_zp, d_scale) = dst_dt.zp_scale();
                //TODO: optimize scale_by
                macro_rules! q8_to_q8 {
                    ($typ:ty) => {
                        if dst_dt.unquantized() == <$typ>::datum_type() {
                            self.as_slice_unchecked::<$typ>()
                                .iter()
                                .zip(result.as_slice_mut_unchecked::<$typ>().iter_mut())
                                .for_each(|(&s, d)| {
                                    *d = (d_zp as i16
                                        + scale_by(s as i16 - s_zp as i16, s_scale / d_scale))
                                    .clamp_cast()
                                });
                            return Ok(Cow::Owned(result));
                        }
                    };
                }

                macro_rules! q_f {
                    ($source:ty, $dest:ty, $round:expr) => {
                        if <$source>::datum_type().unquantized() == self.datum_type().unquantized()
                            && <$dest>::datum_type().unquantized() == dst_dt.unquantized()
                        {
                            self.as_slice_unchecked::<$source>()
                                .iter()
                                .zip(result.as_slice_mut_unchecked::<$dest>().iter_mut())
                                .for_each(|(&s, d)| {
                                    let s_float = (s as f32 - s_zp as f32) * s_scale as f32;
                                    let d_float = s_float as f32 / d_scale as f32 + d_zp as f32;
                                    *d = $round(d_float).clamp_cast();
                                });
                            return Ok(Cow::Owned(result));
                        }
                    };
                }

                macro_rules! q_n {
                    (clamp $source:ty, $dest:ty) => {{
                        if <$source>::datum_type().unquantized() == self.datum_type().unquantized()
                            && <$dest>::datum_type().unquantized() == dst_dt.unquantized()
                        {
                            self.as_slice_unchecked::<$source>()
                                .iter()
                                .zip(result.as_slice_mut_unchecked::<$dest>().iter_mut())
                                .for_each(|(&s, d)| {
                                    *d = s.clamp_cast();
                                });
                            return Ok(Cow::Owned(result));
                        }
                    }};
                    ($source:ty, $dest:ty) => {{
                        if <$source>::datum_type().unquantized() == self.datum_type().unquantized()
                            && <$dest>::datum_type().unquantized() == dst_dt.unquantized()
                        {
                            self.as_slice_unchecked::<$source>()
                                .iter()
                                .zip(result.as_slice_mut_unchecked::<$dest>().iter_mut())
                                .for_each(|(&s, d)| {
                                    *d = s as $dest;
                                });
                            return Ok(Cow::Owned(result));
                        }
                    }};
                }

                if dst_dt.unquantized() == self.datum_type().unquantized()
                    && dst_dt.is_quantized()
                    && self.datum_type().is_quantized()
                {
                    q8_to_q8!(i8);
                    q8_to_q8!(u8);
                }

                q_f!(f32, i8, round_ties_to_even);
                q_f!(f32, u8, round_ties_to_even);
                q_f!(i8, f32, |f| f);
                q_f!(u8, f32, |f| f);

                if dst_dt.is_quantized() && self.datum_type().is_quantized() {
                    q_f!(u8, i8, round_ties_to_even);
                    q_f!(i8, u8, round_ties_to_even);
                }

                q_n!(i8, i32);
                q_n!(i8, u32);
                q_n!(u8, i32);
                q_n!(u8, u32);
                q_n!(clamp i32, i8);
                q_n!(clamp i32, u8);
                q_n!(clamp u32, i8);
                q_n!(clamp u32, u8);
                q_n!(i8, i8);
                q_n!(u8, u8);
                q_n!(i32, i32);
                q_n!(u32, u32);
            }

            anyhow::bail!("Unsupported cast from {:?} to {:?}", self.dt, dst_dt)
        }
    }

    /// Access the data as a scalar, after a cast.
    pub fn cast_to_scalar<D: Datum + Copy>(&self) -> anyhow::Result<D> {
        let casted = self.cast_to::<D>()?;
        casted.to_scalar::<D>().map(|&x| x)
    }

    /// Access the nth element of the tensor, returned as a 0-rank Tensor
    pub fn nth(&self, nth: usize) -> anyhow::Result<Tensor> {
        if nth >= self.len() {
            anyhow::bail!(
                "nth called with {}th element on a tensor of len {} ({:?}",
                nth,
                self.len(),
                self
            );
        }
        unsafe fn nth_t<T: Datum>(me: &Tensor, nth: usize, output: &mut Tensor) {
            let value = me.as_slice_unchecked::<T>()[nth].clone();
            output.as_slice_mut_unchecked::<T>()[0] = value;
        }
        unsafe {
            let mut output = Tensor::uninitialized_dt(self.datum_type(), &[])?;
            dispatch_datum_by_size!(nth_t(self.datum_type())(self, nth, &mut output));
            Ok(output)
        }
    }

    /// Strict equality test on tensors.
    fn eq_dt(&self, other: &Tensor) -> anyhow::Result<bool> {
        unsafe fn eq_t<D: Datum>(me: &Tensor, other: &Tensor) -> bool {
            me.as_slice_unchecked::<D>() == other.as_slice_unchecked::<D>()
        }

        unsafe {
            Ok(self.datum_type() == other.datum_type()
                && self.shape() == other.shape()
                && dispatch_datum!(eq_t(self.dt)(self, other)))
        }
    }

    fn from_datum<T: Datum>(it: ArrayD<T>) -> Tensor {
        if it.as_slice().is_some() {
            let layout =
                alloc::Layout::from_size_align(it.len() * size_of::<T>(), align_of::<T>()).unwrap();
            let shape = it.shape().into();
            let vec = it.into_raw_vec().into_boxed_slice();
            let data = Box::into_raw(vec) as *mut u8;
            let mut t =
                Tensor { dt: T::datum_type(), shape, layout, data, strides: tvec!(), len: 0 };
            t.update_strides_and_len();
            return t;
        }
        unsafe {
            let mut t = Self::uninitialized::<T>(it.shape()).unwrap();
            if it.strides().iter().all(|&s| s > 0) {
                let mut len_and_strides: TVec<(usize, usize)> = tvec!();
                for (len, stride) in itertools::izip!(it.shape(), it.strides(), t.strides())
                    .sorted_by_key(|(_, src, _)| *src)
                    .map(|(l, _, dst)| (*l as isize, *dst))
                {
                    if !len_and_strides.is_empty()
                        && len_and_strides.last().unwrap().1 * len_and_strides.last().unwrap().0
                            == stride as usize
                    {
                        len_and_strides.last_mut().unwrap().0 *= len as usize;
                    } else {
                        len_and_strides.push((len as usize, stride as usize));
                    }
                }
                len_and_strides.reverse();
                crate::scatter::scatter_contig_data(
                    it.as_ptr(),
                    t.as_ptr_mut_unchecked(),
                    &len_and_strides,
                );
                return t;
            }
            // finally use ndarray into_iter()
            t.as_slice_mut_unchecked()
                .iter_mut()
                .zip(it.into_iter())
                .for_each(|(t, a)| *t = a.clone());
            t
        }
    }

    pub fn deep_clone(&self) -> Tensor {
        if self.dt == DatumType::String {
            let data: Vec<String> = self.as_slice::<String>().unwrap().to_vec();
            let t = Tensor {
                data: data.as_ptr() as *mut u8,
                shape: self.shape.clone(),
                strides: self.strides.clone(),
                ..*self
            };
            std::mem::forget(data);
            t
        } else if self.dt == DatumType::TDim {
            let data: Vec<TDim> = self.as_slice::<TDim>().unwrap().to_vec();
            let t = Tensor {
                data: data.as_ptr() as *mut u8,
                shape: self.shape.clone(),
                strides: self.strides.clone(),
                ..*self
            };
            std::mem::forget(data);
            t
        } else {
            unsafe {
                let tensor = Tensor::uninitialized_dt(self.datum_type(), self.shape()).unwrap();
                self.data
                    .copy_to_nonoverlapping(tensor.data, self.len() * self.datum_type().size_of());
                tensor
            }
        }
    }

    pub fn slice(&self, axis: usize, start: usize, end: usize) -> anyhow::Result<Tensor> {
        if axis >= self.rank() {
            anyhow::bail!("Can not slice at axis {} tensor {:?}", axis, self);
        }
        fn slice_t<T: Datum>(
            t: &Tensor,
            axis: usize,
            start: usize,
            end: usize,
        ) -> anyhow::Result<Tensor> {
            Ok(t.to_array_view::<T>()?
                .slice_axis(ndarray::Axis(axis), (start..end).into())
                .into_owned()
                .into_tensor())
        }
        dispatch_datum!(slice_t(self.datum_type())(&self, axis, start, end))
    }

    pub fn view(&self) -> view::TensorView {
        unsafe { view::TensorView::at_prefix_unchecked(self, &[]) }
    }

    pub fn view_at_prefix(&self, prefix: &[usize]) -> anyhow::Result<view::TensorView> {
        view::TensorView::at_prefix(self, prefix)
    }

    pub fn view_mut(&mut self) -> view::TensorView {
        unsafe { view::TensorView::at_prefix_unchecked(self, &[]) }
    }

    pub fn view_at_prefix_mut(&mut self, prefix: &[usize]) -> anyhow::Result<view::TensorView> {
        view::TensorView::at_prefix(self, prefix)
    }

    /// Offsets the tensor as an i8 type if it's an u8 type, otherwise passes it unchanged.
    pub fn offset_u8_as_i8(self: &Arc<Self>) -> Arc<Self> {
        let mut t = if let DatumType::U8 = self.dt.unquantized() {
            self.to_array_view::<u8>().unwrap().mapv(|v| v.wrapping_sub(128) as i8).into_tensor()
        } else {
            return self.clone();
        };

        if let DatumType::QU8(qp) = self.dt {
            if let QParams::ZpScale { zero_point, scale } = qp {
                t.dt = DatumType::QI8(QParams::ZpScale { zero_point: zero_point - 128, scale });
            } else {
                t.dt = DatumType::QI8(qp);
            }
        }

        t.into_arc_tensor()
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Tensor) -> bool {
        if self.dt != other.dt || self.shape != other.shape {
            return false;
        }
        self.eq_dt(other).unwrap_or(false)
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let content = self.dump(false).unwrap_or_else(|e| format!("Error : {:?}", e));
        write!(formatter, "{}", content)
    }
}

pub fn natural_strides(shape: &[usize]) -> TVec<isize> {
    let mut strides = tvec!();
    compute_natural_stride_to(&mut strides, shape);
    strides
}

fn compute_natural_stride_to(strides: &mut TVec<isize>, shape: &[usize]) {
    match shape.len() {
        0 => (),
        1 => strides.push(1),
        2 => strides.extend_from_slice(&[shape[1] as isize, 1]),
        3 => strides.extend_from_slice(&[(shape[1] * shape[2]) as isize, shape[2] as _, 1]),
        4 => strides.extend_from_slice(&[
            (shape[1] * shape[2] * shape[3]) as isize,
            (shape[2] * shape[3]) as _,
            shape[3] as _,
            1,
        ]),
        _ => {
            strides.push(1);
            for dim in shape.as_ref().iter().skip(1).rev() {
                let previous = strides.last().unwrap().clone();
                strides.push(previous * *dim as isize)
            }
            strides.reverse();
        }
    }
}

#[cfg(feature = "serialize")]
impl Serialize for Tensor {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        macro_rules! serialize_inner {
            ($type:ident, $m:ident) => {{
                let data =
                    (stringify!($type), self.shape(), $m.iter().cloned().collect::<Vec<_>>());
                data.serialize(serializer)
            }};
        };

        use Tensor::*;
        match self {
            Bool(m) => serialize_inner!(bool, m),
            U8(m) => serialize_inner!(u8, m),
            U16(m) => serialize_inner!(u16, m),
            I8(m) => serialize_inner!(i8, m),
            I16(m) => serialize_inner!(i16, m),
            I32(m) => serialize_inner!(i32, m),
            I64(m) => serialize_inner!(i64, m),
            F16(m) => serialize_inner!(f16, m),
            F32(m) => serialize_inner!(f32, m),
            F64(m) => serialize_inner!(f64, m),
            TDim(m) => serialize_inner!(TDim, m),
            String(m) => serialize_inner!(str, m),
        }
    }
}

impl<D: ::ndarray::Dimension, T: Datum> From<Array<T, D>> for Tensor {
    fn from(it: Array<T, D>) -> Tensor {
        Tensor::from_datum(it.into_dyn())
    }
}

/// Convenient conversion to Tensor.
pub trait IntoTensor: Sized {
    /// Convert Self to a Tensor.
    ///
    /// May perform a copy
    fn into_tensor(self) -> Tensor;
}

/// Convenient conversion to Arc<Tensor>.
pub trait IntoArcTensor: Sized {
    /// Convert Self to a Arc<Tensor>.
    ///
    /// May perform a copy
    fn into_arc_tensor(self) -> Arc<Tensor>;
}

impl<D: ::ndarray::Dimension, T: Datum> IntoTensor for Array<T, D> {
    fn into_tensor(self) -> Tensor {
        Tensor::from(self)
    }
}

impl<D: ::ndarray::Dimension, T: Datum> IntoArcTensor for Array<T, D> {
    fn into_arc_tensor(self) -> Arc<Tensor> {
        Arc::new(Tensor::from(self))
    }
}

impl IntoTensor for Arc<Tensor> {
    fn into_tensor(self) -> Tensor {
        Arc::try_unwrap(self).unwrap_or_else(|t| (*t).clone())
    }
}

impl IntoArcTensor for Tensor {
    fn into_arc_tensor(self) -> Arc<Tensor> {
        Arc::new(self)
    }
}

impl IntoArcTensor for Arc<Tensor> {
    fn into_arc_tensor(self) -> Arc<Tensor> {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[derive(Debug)]
    struct PermuteAxisProblem {
        shape: Vec<usize>,
        permutation: Vec<usize>,
    }

    impl Arbitrary for PermuteAxisProblem {
        type Strategy = BoxedStrategy<PermuteAxisProblem>;
        type Parameters = ();

        fn arbitrary_with(_: Self::Parameters) -> Self::Strategy {
            (0..8usize)
                .prop_flat_map(|rank| {
                    let permute: Vec<usize> = (0..rank).collect();
                    (proptest::collection::vec(1..5usize, rank), Just(permute).prop_shuffle())
                })
                .prop_map(|(shape, permutation)| PermuteAxisProblem { shape, permutation })
                .boxed()
        }
    }

    impl PermuteAxisProblem {
        fn input(&self) -> ArrayD<i32> {
            let mut i = 0;
            ArrayD::from_shape_simple_fn(&*self.shape, || {
                i += 1;
                i
            })
            .permuted_axes(&*self.permutation)
        }

        fn reference(&self) -> Tensor {
            let values: Vec<i32> = self.input().iter().copied().collect();
            let shape = self.permutation.iter().map(|ix| self.shape[*ix]).collect::<TVec<usize>>();
            super::litteral::tensor1(&values).into_shape(&shape).unwrap()
        }

        fn tract(&self) -> Tensor {
            Tensor::from(self.input())
        }

        fn check(&self) -> proptest::test_runner::TestCaseResult {
            prop_assert_eq!(self.tract(), self.reference());
            Ok(())
        }
    }

    proptest::proptest! {
        #[test]
        fn prop(pb: PermuteAxisProblem) {
            pb.check().unwrap();
        }
    }

    #[test]
    fn t_1_2() {
        PermuteAxisProblem { shape: vec![2, 1], permutation: vec![1, 0] }.check().unwrap();
    }

    #[test]
    fn t_2_2() {
        PermuteAxisProblem { shape: vec![2, 2], permutation: vec![1, 0] }.check().unwrap();
    }
}
