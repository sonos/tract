//! `Tensor`, tract main data object of interest.
use crate::dim::TDim;
use crate::internal::*;
use ndarray::prelude::*;
use std::alloc;
use std::fmt;
use std::mem::{align_of, size_of};

use tract_linalg::f16::f16;

#[cfg(feature = "serialize")]
use serde::ser::{Serialize, Serializer};
use std::sync::Arc;

pub mod litteral;

/// Tensor is a concrete tensor in tract.
pub struct Tensor {
    dt: DatumType,
    shape: TVec<usize>,
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
        Tensor::from(arr0(0f32))
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
    pub unsafe fn uninitialized<T: Datum>(shape: &[usize]) -> TractResult<Tensor> {
        Self::uninitialized_dt(T::datum_type(), shape)
    }

    /// Create an uninitialized tensor (dt as regular parameter).
    pub unsafe fn uninitialized_dt(dt: DatumType, shape: &[usize]) -> TractResult<Tensor> {
        Self::uninitialized_aligned_dt(dt, shape, dt.alignment())
    }

    /// Create an uninitialized tensor with a given alignment (in bytes).
    pub unsafe fn uninitialized_aligned<T: Datum>(
        shape: &[usize],
        alignment: usize,
    ) -> TractResult<Tensor> {
        Self::uninitialized_aligned_dt(T::datum_type(), shape, alignment)
    }

    /// Create an uninitialized tensor with a given alignment (in bytes).
    pub unsafe fn uninitialized_aligned_dt(
        dt: DatumType,
        shape: &[usize],
        alignment: usize,
    ) -> TractResult<Tensor> {
        if dt == String::datum_type() {
            return Ok(ndarray::ArrayD::<String>::default(shape).into());
        } else if dt == TDim::datum_type() {
            return Ok(ndarray::ArrayD::<TDim>::default(shape).into());
        }
        let bytes = shape.iter().cloned().product::<usize>() * dt.size_of();
        let layout = alloc::Layout::from_size_align(bytes, alignment)?;
        let data = if bytes == 0 {
            std::ptr::null()
        } else {
            let ptr = alloc::alloc(layout);
            assert!(!ptr.is_null());
            ptr
        } as *mut u8;
        Ok(Tensor { layout, dt, shape: shape.into(), data })
    }

    pub fn stack_tensors(
        axis: usize,
        tensors: &[impl std::borrow::Borrow<Tensor>],
    ) -> TractResult<Tensor> {
        use crate::datum::ArrayDatum;
        let dt = tensors[0].borrow().datum_type();
        if tensors.iter().any(|t| t.borrow().datum_type() != dt) {
            bail!("Inconsistent datum type in stack.")
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
            }
        }?;
        tensor.dt = dt;
        Ok(tensor)
    }

    pub fn zero<T: Datum + num_traits::Zero>(shape: &[usize]) -> TractResult<Tensor> {
        let mut t = unsafe { Tensor::uninitialized::<T>(shape)? };
        t.as_slice_mut::<T>().unwrap().iter_mut().for_each(|item| *item = T::zero());
        Ok(t)
    }

    pub fn zero_dt(dt: DatumType, shape: &[usize]) -> TractResult<Tensor> {
        dispatch_numbers!(Self::zero(dt)(shape))
    }

    /// Create an tensor from raw data.
    ///
    /// It copies the data, aligning it to the size of T.
    pub unsafe fn from_raw<T: Datum>(shape: &[usize], content: &[u8]) -> TractResult<Tensor> {
        Tensor::from_raw_dt(T::datum_type(), shape, content)
    }

    pub unsafe fn from_raw_dt(
        dt: DatumType,
        shape: &[usize],
        content: &[u8],
    ) -> TractResult<Tensor> {
        let bytes = shape.iter().cloned().product::<usize>() * dt.size_of();
        let layout = alloc::Layout::from_size_align(bytes, dt.alignment())?;
        let data = alloc::alloc(layout);
        content.as_ptr().copy_to_nonoverlapping(data, bytes);
        Ok(Tensor { dt, shape: shape.into(), data, layout })
    }

    /// Get the number of dimensions (or axes) of the tensor.
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Get the shape of the tensor.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the number of valeus in the tensor.
    pub fn len(&self) -> usize {
        self.shape.iter().cloned().product::<usize>()
    }

    /// Force the tensor shape, no consistency check.
    pub unsafe fn set_shape_unchecked(&mut self, shape: &[usize]) {
        self.shape = shape.into();
    }

    /// Force the tensor shape.
    pub fn set_shape(&mut self, shape: &[usize]) -> TractResult<()> {
        if self.len() != shape.iter().product() {
            bail!("Invalid reshape {:?} to {:?}", self.shape, shape);
        }
        self.shape = shape.into();
        Ok(())
    }

    pub fn permute_axes(self, axes: &[usize]) -> TractResult<Tensor> {
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

    /// Reshape the tensor to `shape`.
    pub fn into_shape(mut self, shape: &[usize]) -> TractResult<Tensor> {
        self.set_shape(shape)?;
        Ok(self)
    }

    pub fn insert_axis(&mut self, axis: usize) -> TractResult<()> {
        self.shape.insert(axis, 1);
        Ok(())
    }

    pub fn remove_axis(&mut self, axis: usize) -> TractResult<()> {
        self.shape.remove(axis);
        Ok(())
    }

    pub fn broadcast_into_rank(mut self, rank: usize) -> TractResult<Tensor> {
        self.broadcast_to_rank(rank)?;
        Ok(self)
    }

    pub fn broadcast_to_rank(&mut self, rank: usize) -> TractResult<()> {
        if rank < self.rank() {
            bail!("Can only broadcast to higher rank")
        }
        while self.shape.len() < rank {
            self.shape.insert(0, 1)
        }
        Ok(())
    }

    /// Get the datum type of the tensor.
    pub fn datum_type(&self) -> DatumType {
        self.dt
    }

    /// Set the datum type of the tensor.
    pub unsafe fn set_datum_type(&mut self, dt: DatumType) {
        self.dt = dt
    }

    /// Dump the tensor in a human readable form.
    ///
    /// `force_full` will force the tensor to be dump in full even if it is big.
    pub fn dump(&self, force_full: bool) -> TractResult<String> {
        let spec = TypedFact::dt_shape(self.datum_type(), &*self.shape)?;
        unsafe fn dump_t<D: Datum>(tensor: &Tensor, n: usize) -> String {
            use itertools::Itertools;
            tensor.as_slice_unchecked::<D>()[0..n].iter().join(", ")
        }
        unsafe {
            if force_full || self.len() <= 12 {
                let data = dispatch_datum!(dump_t(self.datum_type())(self, self.len()));
                Ok(format!("{} {}", spec.format_dt_shape(), data))
            } else {
                let data = dispatch_datum!(dump_t(self.datum_type())(self, 12));
                Ok(format!("{} {}...", spec.format_dt_shape(), data))
            }
        }
    }

    /// Compare two tensors, allowing for rounding errors.
    pub fn close_enough(&self, other: &Self, approx: bool) -> TractResult<()> {
        if self.shape() != other.shape() {
            bail!("Shape mismatch {:?} != {:?}", self.shape(), other.shape())
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
                    bail!("Mismatch at {:?} {} != {}", indices.slice(), a, b)
                }
                Ok(())
            })
        } else {
            if self.eq(other) {
                Ok(())
            } else {
                bail!("Mismatch")
            }
        }
    }

    /// Transform the tensor into a `ndarray::Array`.
    pub fn into_array<D: Datum>(self) -> TractResult<ArrayD<D>> {
        Ok(self.to_array_view::<D>()?.to_owned())
    }

    /// Transform the tensor into a `ndarray::Array`.
    pub unsafe fn into_array_unchecked<D: Datum>(self) -> ArrayD<D> {
        self.to_array_view_unchecked::<D>().to_owned()
    }

    fn check_for_access<D: Datum>(&self) -> TractResult<()> {
        if self.datum_type() != D::datum_type() {
            bail!(
                "Tensor datum type error: tensor is {:?}, accessed as {:?}",
                self.datum_type(),
                D::datum_type(),
            );
        }
        Ok(())
    }

    /// Transform the data as a `ndarray::Array`.
    pub fn to_array_view<'a, D: Datum>(&'a self) -> TractResult<ArrayViewD<'a, D>> {
        self.check_for_access::<D>()?;
        unsafe { Ok(self.to_array_view_unchecked()) }
    }

    /// Transform the data as a mutable `ndarray::Array`.
    pub fn to_array_view_mut<'a, D: Datum>(&'a mut self) -> TractResult<ArrayViewMutD<'a, D>> {
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
    pub fn as_ptr<D: Datum>(&self) -> TractResult<*const D> {
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
    pub fn as_ptr_mut<D: Datum>(&mut self) -> TractResult<*mut D> {
        self.as_ptr::<D>().map(|p| p as *mut D)
    }

    /// Access the data as a slice.
    pub fn as_slice<D: Datum>(&self) -> TractResult<&[D]> {
        unsafe { Ok(std::slice::from_raw_parts::<D>(self.as_ptr()?, self.len())) }
    }

    /// Access the data as a mutable slice.
    pub fn as_slice_mut<D: Datum>(&mut self) -> TractResult<&mut [D]> {
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
    pub fn to_scalar<'a, D: Datum>(&'a self) -> TractResult<&D> {
        unsafe { Ok(&*(self.as_ptr::<D>()?)) }
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

    fn is_uniform_t<T: Datum>(&self) -> TractResult<bool> {
        let slice = self.as_slice::<T>()?;
        Ok(slice[1..].iter().all(|x| x == &slice[0]))
    }

    pub fn is_uniform(&self) -> TractResult<bool> {
        if self.len() <= 1 {
            return Ok(true);
        }
        dispatch_datum!(Tensor::is_uniform_t(self.datum_type())(self))
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
    ) -> TractResult<()> {
        for (s, d) in self
            .as_slice_unchecked::<String>()
            .iter()
            .zip(other.as_slice_mut_unchecked::<Target>().iter_mut())
        {
            *d = s
                .parse()
                .map_err(|_| format!("Could not parse {} as {:?}", s, Target::datum_type()))?
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
    pub fn cast_to<D: Datum>(&self) -> TractResult<Cow<Tensor>> {
        self.cast_to_dt(D::datum_type())
    }

    /// Optionnaly convert data to a tensor for a new DatumType.
    pub fn cast_to_dt(&self, dt: DatumType) -> TractResult<Cow<Tensor>> {
        unsafe {
            if self.dt == dt {
                return Ok(Cow::Borrowed(self));
            }
            if self.dt == TDim::datum_type() && (dt.is_integer() || dt.is_float()) {
                let slice = self.as_slice_unchecked::<TDim>();
                let mut ints = Self::uninitialized::<i64>(&self.shape)?;
                let ints_slice = ints.as_slice_mut_unchecked::<i64>();
                for i in 0..self.len() {
                    ints_slice[i] = slice[i].to_integer()?;
                }
                return Ok(Cow::Owned(ints.cast_to_dt(dt)?.into_owned()));
            }
            if self.dt == bool::datum_type() && (dt.is_integer() || dt.is_float()) {
                let slice = self.as_slice_unchecked::<bool>();
                let mut ints = Self::uninitialized::<i8>(&self.shape)?;
                let ints_slice = ints.as_slice_mut_unchecked::<i8>();
                for i in 0..self.len() {
                    ints_slice[i] = slice[i] as usize as i8;
                }
                return Ok(Cow::Owned(ints.cast_to_dt(dt)?.into_owned()));
            }
            let mut result = Self::uninitialized_dt(dt, &self.shape)?;
            if self.dt == DatumType::String {
                dispatch_datum!(Self::cast_from_string(dt)(self, &mut result))?;
                return Ok(Cow::Owned(result));
            }
            if dt == DatumType::String {
                dispatch_datum!(Self::cast_to_string(self.dt)(self, &mut result));
                return Ok(Cow::Owned(result));
            }
            macro_rules! n {
                ($source:ty) => {
                    if <$source>::datum_type() == self.datum_type() {
                        match dt {
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
            };
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
            bail!("Unsupported cast from {:?} to {:?}", self.dt, dt)
        }
    }

    /// Access the data as a scalar, after a cast.
    pub fn cast_to_scalar<D: Datum + Copy>(&self) -> TractResult<D> {
        let casted = self.cast_to::<D>()?;
        casted.to_scalar::<D>().map(|&x| x)
    }

    /// Strict equality test on tensors.
    fn eq_dt(&self, other: &Tensor) -> TractResult<bool> {
        unsafe fn eq_t<D: Datum>(me: &Tensor, other: &Tensor) -> bool {
            me.as_slice_unchecked::<D>() == other.as_slice_unchecked::<D>()
        }

        unsafe {
            Ok(self.datum_type() == other.datum_type()
                && self.shape() == other.shape()
                && dispatch_datum!(eq_t(self.dt)(self, other)))
        }
    }

    fn from_copy_datum<D: ::ndarray::Dimension, T: Datum>(it: Array<T, D>) -> Tensor {
        let shape = it.shape().into();
        let vec = if it.as_slice().is_some() {
            it.into_raw_vec().into_boxed_slice()
        } else {
            it.into_owned().into_iter().cloned().collect::<Box<[T]>>()
        };
        let layout =
            alloc::Layout::from_size_align(vec.len() * size_of::<T>(), align_of::<T>()).unwrap();
        let data = Box::into_raw(vec) as *mut u8;
        Tensor { dt: T::datum_type(), shape, layout, data }
    }

    pub fn deep_clone(&self) -> Tensor {
        if self.dt == DatumType::String {
            let data: Vec<String> = self.as_slice::<String>().unwrap().to_vec();
            let t = Tensor { data: data.as_ptr() as *mut u8, shape: self.shape.clone(), ..*self };
            std::mem::forget(data);
            t
        } else if self.dt == DatumType::TDim {
            let data: Vec<TDim> = self.as_slice::<TDim>().unwrap().to_vec();
            let t = Tensor { data: data.as_ptr() as *mut u8, shape: self.shape.clone(), ..*self };
            std::mem::forget(data);
            t
        } else {
            unsafe {
                let data = alloc::alloc(self.layout) as *mut u8;
                self.data.copy_to_nonoverlapping(data, self.layout.size());
                Tensor { data, shape: self.shape.clone(), ..*self }
            }
        }
    }

    pub fn slice(&self, axis: usize, start: usize, end: usize) -> TractResult<Tensor> {
        if axis >= self.rank() {
            bail!("Can not slice at axis {} tensor {:?}", axis, self);
        }
        fn slice_t<T: Datum>(
            t: &Tensor,
            axis: usize,
            start: usize,
            end: usize,
        ) -> TractResult<Tensor> {
            Ok(t.to_array_view::<T>()?
                .slice_axis(ndarray::Axis(axis), (start..end).into())
                .into_owned()
                .into_tensor())
        }
        dispatch_datum!(slice_t(self.datum_type())(&self, axis, start, end))
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
        Tensor::from_copy_datum(it)
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
