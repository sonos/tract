//! `Tensor`, tract main data object of interest.
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
    null: bool,
    dt: DatumType,
    shape: TVec<usize>,
    layout: alloc::Layout,
    data: *mut u8,
}

unsafe impl Send for Tensor {}
unsafe impl Sync for Tensor {}

impl Clone for Tensor {
    fn clone(&self) -> Tensor {
        if self.dt == DatumType::String {
            let data = self.as_slice::<String>().unwrap().to_vec();
            let t = Tensor { data: data.as_ptr() as *mut u8, shape: self.shape.clone(), ..*self };
            std::mem::forget(data);
            t
        } else if self.null {
            Tensor { shape: self.shape.clone(), ..*self }
        } else {
            unsafe {
                let data = alloc::alloc(self.layout) as *mut u8;
                self.data.copy_to_nonoverlapping(data, self.layout.size());
                Tensor { data, shape: self.shape.clone(), ..*self }
            }
        }
    }
}

impl Default for Tensor {
    fn default() -> Tensor {
        Tensor::from(arr0(0f32))
    }
}

impl Drop for Tensor {
    fn drop(&mut self) {
        if self.dt == DatumType::String {
            unsafe {
                self.as_slice_mut::<String>()
                    .unwrap()
                    .iter_mut()
                    .for_each(|s| std::ptr::drop_in_place(s as *mut String));
            }
        }
        if !self.data.is_null() && self.layout.size() > 0 {
            unsafe { alloc::dealloc(self.data, self.layout) }
        }
    }
}

impl Tensor {
    /// Create an uninitialized tensor with a given alignment (in bytes).
    pub unsafe fn uninitialized_aligned<T: Datum>(
        shape: &[usize],
        alignment: usize,
    ) -> TractResult<Tensor> {
        let bytes = shape.iter().cloned().product::<usize>() * size_of::<T>();
        let layout = alloc::Layout::from_size_align(bytes, alignment)?;
        let data = if bytes == 0 { std::ptr::null() } else { alloc::alloc(layout) } as *mut u8;
        Ok(Tensor { null: false, layout, dt: T::datum_type(), shape: shape.into(), data })
    }

    /// Create an tensor from raw data.
    ///
    /// It copies the data, aligning it to the size of T.
    pub unsafe fn from_raw<T: Datum>(shape: &[usize], content: &[u8]) -> TractResult<Tensor> {
        let bytes = shape.iter().cloned().product::<usize>() * size_of::<T>();
        let layout = alloc::Layout::from_size_align(bytes, size_of::<T>())?;
        let data = alloc::alloc(layout);
        content.as_ptr().copy_to_nonoverlapping(data, bytes);
        Ok(Tensor { null: false, dt: T::datum_type(), shape: shape.into(), data, layout })
    }

    /// Creates a null tensor (this is rare, and should stay that way).
    pub unsafe fn null<T: Datum>(shape: &[usize]) -> TractResult<Tensor> {
        Self::null_dt(T::datum_type(), shape)
    }

    /// Creates a null tensor (this is rare, and should stay that way).
    pub unsafe fn null_dt(dt: DatumType, shape: &[usize]) -> TractResult<Tensor> {
        Ok(Tensor {
            null: true,
            dt,
            shape: shape.into(),
            data: std::ptr::null::<u8>() as *mut u8,
            layout: alloc::Layout::from_size_align(0, dt.size_of())?,
        })
    }

    /// Check weather self is a null tensor.
    pub fn is_null(&self) -> bool {
        self.null
    }

    /// Get the shape of the tensor.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the number of valeus in the tensor.
    pub fn len(&self) -> usize {
        self.shape.iter().cloned().product::<usize>()
    }

    /// Reshape the tensor to `shape`.
    pub unsafe fn into_shape(self, shape: &[usize]) -> TractResult<Tensor> {
        let t = Tensor { shape: shape.into(), ..self };
        std::mem::forget(self);
        Ok(t)
    }

    /// Get the datum type of the tensor.
    pub fn datum_type(&self) -> DatumType {
        self.dt
    }

    /// Dump the tensor in a human readable form.
    ///
    /// `force_full` will force the tensor to be dump in full even if it is big.
    pub fn dump_t<D: Datum>(&self, force_full: bool) -> TractResult<String> {
        use itertools::Itertools;
        let spec = TensorFact::dt_shape(D::datum_type(), &*self.shape);
        let data = self.to_array_view::<D>()?;
        let s = if force_full || data.len() <= 12 {
            format!("{} {}", spec.format_dt_shape(), data.iter().join(", "))
        } else {
            format!("{} {}...", spec.format_dt_shape(), data.iter().take(8).join(", "))
        };
        Ok(s)
    }

    /// Dump the tensor in a human readable form.
    ///
    /// `force_full` will force the tensor to be dump in full even if it is big.
    pub fn dump(&self, force_full: bool) -> TractResult<String> {
        dispatch_datum!(Self::dump_t(self.dt)(self, force_full))
    }

    /// Compute a normalized L1 distance between two tensors.
    pub fn l1(&self, other: &Self) -> TractResult<f64> {
        let ma = self.cast_to::<f32>()?;
        let ma = ma.to_array_view::<f32>()?;
        let mb = other.cast_to::<f32>()?;
        let mb = mb.to_array_view::<f32>()?;
        let sum = ma.iter().zip(mb.iter()).map(|(a, b)| (a - b).abs() as f64).sum::<f64>();
        Ok(sum / self.len() as f64)
    }

    pub fn all_close_default(&self, other: &Self) -> TractResult<bool> {
        self.all_close(other, 1e-4, 1e-4)
    }

    pub fn all_close(&self, other: &Self, rtol: f32, atol: f32) -> TractResult<bool> {
        let ma = self.cast_to::<f32>()?;
        let ma = ma.to_array_view::<f32>()?;
        let mb = other.cast_to::<f32>()?;
        let mb = mb.to_array_view::<f32>()?;
        Ok(ma.iter().zip(mb.iter()).all(|(a, b)| {
            if (a - b).abs() > atol + rtol * b.abs() {
                println!("|{}-{}| = {} > {}", a, b, (a - b).abs(), atol + rtol * b.abs());
            }
            (a.is_nan() && b.is_nan())
                || (a.is_infinite() && b.is_infinite() && a.signum() == b.signum())
                || (a - b).abs() <= atol + rtol * b.abs()
        }))
    }

    /// Compare two tensors, allowing for rounding errors.
    pub fn close_enough(&self, other: &Self, approx: bool) -> bool {
        if self.is_null() != other.is_null() {
            return false;
        }
        if approx {
            self.all_close_default(other).unwrap()
        } else {
            self.eq(other)
        }
    }

    /// Transform the tensor into a `ndarray::Array`.
    pub fn into_array<D: Datum>(self) -> TractResult<ArrayD<D>> {
        Ok(self.to_array_view::<D>()?.to_owned())
    }

    fn check_for_access<D: Datum>(&self) -> TractResult<()> {
        if self.datum_type() != D::datum_type() {
            bail!(
                "Incompatible datum type. Required {:?}, got {:?}",
                D::datum_type(),
                self.datum_type()
            );
        }
        if self.is_null() {
            bail!("Null tensor")
        }
        Ok(())
    }

    /// Transform the data as a `ndarray::Array`.
    pub fn to_array_view<'a, D: Datum>(&'a self) -> TractResult<ArrayViewD<'a, D>> {
        if self.len() != 0 {
            unsafe {
                return Ok(ArrayViewD::from_shape_ptr(&*self.shape, self.as_ptr()?));
            }
        } else {
            return Ok(ArrayViewD::from_shape(&*self.shape, &[])?);
        }
    }

    /// Transform the data as a mutable `ndarray::Array`.
    pub fn to_array_view_mut<'a, D: Datum>(&'a mut self) -> TractResult<ArrayViewMutD<'a, D>> {
        if self.len() != 0 {
            unsafe {
                return Ok(ArrayViewMutD::from_shape_ptr(&*self.shape, self.data as *mut D));
            }
        } else {
            return Ok(ArrayViewMutD::from_shape(&*self.shape, &mut [])?);
        }
    }

    /// Access the data as a pointer.
    pub fn as_ptr<D: Datum>(&self) -> TractResult<*const D> {
        self.check_for_access::<D>()?;
        Ok(self.data as *const D)
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

    /// Access the data as a scalar.
    pub fn to_scalar<'a, D: Datum>(&'a self) -> TractResult<&D> {
        unsafe { Ok(&*(self.as_ptr::<D>()?)) }
    }

    /// Convert data to a tensor for a new DatumType.
    fn cast<Source: Datum + crate::datum::TryInto<Target>, Target: Datum>(
        &self,
    ) -> TractResult<Tensor> {
        let casted_vec: Vec<Target> =
            self.as_slice::<Source>()?.iter().map(|s| s.try_into()).collect::<TractResult<_>>()?;
        let casted_array = ArrayD::from_shape_vec(&*self.shape, casted_vec)?;
        Ok(casted_array.into())
    }

    /// Optionnaly convert data to a tensor for a new DatumType.
    pub fn cast_to<D: Datum>(&self) -> TractResult<Cow<Tensor>> {
        self.cast_to_dt(D::datum_type())
    }

    /// Optionnaly convert data to a tensor for a new DatumType.
    pub fn cast_to_dt(&self, dt: DatumType) -> TractResult<Cow<Tensor>> {
        use DatumType::*;
        if self.dt == dt {
            return Ok(Cow::Borrowed(self));
        }
        let target = match (self.dt, dt) {
            (TDim, I32) => self.cast::<crate::dim::TDim, i32>()?,
            (TDim, I64) => self.cast::<crate::dim::TDim, i64>()?,
            (I32, TDim) => self.cast::<i32, crate::dim::TDim>()?,
            (I64, TDim) => self.cast::<i64, crate::dim::TDim>()?,

            (F16, F32) => self.cast::<f16, f32>()?,
            (F32, F16) => self.cast::<f32, f16>()?,
            (F16, F64) => self.cast::<f16, f64>()?,
            (F64, F16) => self.cast::<f64, f16>()?,
            (F32, F64) => self.cast::<f32, f64>()?,
            (F64, F32) => self.cast::<f64, f32>()?,

            (I8, I16) => self.cast::<i8, i16>()?,
            (I16, I8) => self.cast::<i16, i8>()?,
            (I8, I32) => self.cast::<i8, i32>()?,
            (I32, I8) => self.cast::<i32, i8>()?,
            (I8, I64) => self.cast::<i8, i64>()?,
            (I64, I8) => self.cast::<i64, i8>()?,
            (I16, I32) => self.cast::<i16, i32>()?,
            (I32, I16) => self.cast::<i32, i16>()?,
            (I16, I64) => self.cast::<i16, i64>()?,
            (I64, I16) => self.cast::<i64, i16>()?,
            (I32, I64) => self.cast::<i32, i64>()?,
            (I64, I32) => self.cast::<i64, i32>()?,

            (Bool, F32) => self.cast::<bool, f32>()?,
            (I8, F32) => self.cast::<i8, f32>()?,
            (I16, F32) => self.cast::<i16, f32>()?,
            (I32, F32) => self.cast::<i32, f32>()?,
            (I64, F32) => self.cast::<i64, f32>()?,

            (F32, String) => self.cast::<f32, std::string::String>()?,
            (String, F32) => self.cast::<std::string::String, f32>()?,

            _ => bail!("Unsupported cast from {:?} to {:?}", self.dt, dt),
        };
        Ok(Cow::Owned(target))
    }

    /// Strict equality test on tensors.
    fn eq_t<D: Datum>(&self, other: &Tensor) -> TractResult<bool> {
        Ok(self.to_array_view::<D>()? == other.to_array_view::<D>()?)
    }

    /// Strict equality test on tensors.
    fn eq_dt(&self, other: &Tensor) -> TractResult<bool> {
        dispatch_datum!(Self::eq_t(self.dt)(self, other))
    }

    fn from_copy_datum<D: ::ndarray::Dimension, T: Datum>(it: Array<T, D>) -> Tensor {
        let shape = it.shape().into();
        let vec = if it.is_standard_layout() {
            it.into_raw_vec()
        } else {
            it.view().into_iter().cloned().collect()
        };
        let layout =
            alloc::Layout::from_size_align(vec.capacity() * size_of::<T>(), align_of::<T>())
                .unwrap();
        let data = vec.as_ptr() as *mut u8;
        std::mem::forget(vec);
        Tensor { null: false, dt: T::datum_type(), shape, layout, data }
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
