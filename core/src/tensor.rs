//! `Tensor` is the equivalent of SharedTensor Tensor.
use crate::dim::TDim;
use crate::model::TVec;
use crate::TractResult;
use ndarray::prelude::*;
use std::alloc;
use std::borrow::Cow;
use std::fmt;
use std::mem::size_of;

use tract_linalg::f16::f16;

#[cfg(feature = "serialize")]
use serde::ser::{Serialize, Serializer};
use std::sync::Arc;

use crate::datum::TryInto;
use crate::ops::prelude::*;

#[derive(Debug, Clone)]
pub struct SharedTensor(Arc<Tensor>);

impl SharedTensor {
    /// Returns a reference to the Tensor wrapped inside a SharedTensor.
    pub fn as_tensor(&self) -> &Tensor {
        self.0.as_ref()
    }

    pub fn to_tensor(self) -> Tensor {
        Arc::try_unwrap(self.0).unwrap_or_else(|arc| arc.as_ref().clone())
    }

    pub fn to_array<'a, D: crate::datum::Datum>(self) -> TractResult<::ndarray::ArrayD<D>> {
        self.to_tensor().into_array()
    }
}

impl<M> From<M> for SharedTensor
where
    Tensor: From<M>,
{
    fn from(m: M) -> SharedTensor {
        SharedTensor::from(Arc::new(m.into()))
    }
}

impl From<Arc<Tensor>> for SharedTensor {
    fn from(m: Arc<Tensor>) -> SharedTensor {
        SharedTensor(m)
    }
}

impl ::std::ops::Deref for SharedTensor {
    type Target = Tensor;
    fn deref(&self) -> &Tensor {
        self.0.as_ref()
    }
}

impl PartialEq for SharedTensor {
    fn eq(&self, other: &SharedTensor) -> bool {
        self.as_tensor() == other.as_tensor()
    }
}

pub fn realign_slice(v: &[u8], alignment: usize) -> TractResult<Vec<u8>> {
    assert!(
        (alignment as u32).count_ones() == 1,
        "Invalid alignment required ({})",
        alignment
    );
    if v.len() == 0 {
        return Ok(vec![]);
    }
    unsafe {
        let aligned_buffer = alloc::alloc(
            alloc::Layout::from_size_align(v.len(), alignment).map_err(|e| {
                format!("Memory layout error: {:?} ({}, {})", e, v.len(), alignment)
            })?,
        );
        let mut output = Vec::from_raw_parts(aligned_buffer as _, v.len(), v.len());
        output.copy_from_slice(v);
        Ok(output)
    }
}

pub fn realign_vec(v: Vec<u8>, alignment: usize) -> TractResult<Vec<u8>> {
    if v.len() == 0 || v.as_ptr() as usize % alignment == 0 {
        return Ok(v);
    }
    realign_slice(&v, alignment)
}

pub struct Tensor {
    null: bool,
    dt: DatumType,
    shape: TVec<usize>,
    alignment: usize,
    data: Vec<u8>,
}

impl Clone for Tensor {
    fn clone(&self) -> Tensor {
        assert!(
            (self.alignment as u32).count_ones() == 1,
            "Invalid alignment in tensor ({})",
            self.alignment
        );
        Tensor {
            shape: self.shape.clone(),
            data: realign_slice(&self.data, self.alignment).unwrap(),
            ..*self
        }
    }
}

impl Tensor {
    pub unsafe fn uninitialized_aligned<T: Datum>(
        shape: &[usize],
        alignment: usize,
    ) -> TractResult<Tensor> {
        assert!(
            (alignment as u32).count_ones() == 1,
            "Invalid alignment required ({})",
            alignment
        );
        let len = shape.iter().cloned().product::<usize>() * size_of::<T>();
        let data = if len == 0 {
            vec![]
        } else {
            let aligned_buffer =
                alloc::alloc(alloc::Layout::from_size_align(len, alignment).map_err(|e| {
                    format!("Memory layout error: {:?} ({}, {})", e, len, alignment)
                })?);
            Vec::from_raw_parts(aligned_buffer as _, len, len)
        };
        Ok(Tensor {
            null: false,
            dt: T::datum_type(),
            shape: shape.into(),
            alignment,
            data,
        })
    }
    pub unsafe fn from_raw<T: Datum>(shape: &[usize], content: &[u8]) -> TractResult<Tensor> {
        let data = realign_slice(content, size_of::<T>())?;
        Ok(Tensor {
            null: false,
            dt: T::datum_type(),
            shape: shape.into(),
            data,
            alignment: T::datum_type().alignment(),
        })
    }

    pub fn into_aligned(self, alignment: usize) -> TractResult<Tensor> {
        Ok(Tensor {
            null: self.null,
            dt: self.datum_type(),
            shape: self.shape,
            data: realign_vec(self.data, alignment)?,
            alignment,
        })
    }

    pub unsafe fn null<T: Datum>(shape: &[usize]) -> TractResult<Tensor> {
        Self::null_dt(T::datum_type(), shape)
    }

    pub unsafe fn null_dt(dt: DatumType, shape: &[usize]) -> TractResult<Tensor> {
        Ok(Tensor {
            null: true,
            dt,
            shape: shape.into(),
            data: vec![],
            alignment: dt.alignment(),
        })
    }

    pub fn is_null(&self) -> bool {
        self.null
    }

    pub fn into_tensor(self) -> crate::SharedTensor {
        crate::SharedTensor::from(self)
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn into_shape(self, shape: &[usize]) -> TractResult<Tensor> {
        Ok(Tensor {
            shape: shape.into(),
            ..self
        })
    }

    pub fn datum_type(&self) -> DatumType {
        self.dt
    }

    pub fn dump_t<D: Datum>(&self, force_full: bool) -> TractResult<String> {
        use itertools::Itertools;
        let s = if self.shape.len() == 0 {
            let data = self.to_scalar::<D>()?;
            format!("Scalar: {} ({})", data, D::name())
        } else if self.shape().iter().product::<usize>() > 8 {
            let data = self.to_array_view::<D>()?;
            if force_full {
                format!("shape:{:?} ({}) {:?}", self.shape, D::name(), data)
            } else {
                format!(
                    "shape:{:?} ({}) {}...",
                    self.shape(),
                    D::name(),
                    data.iter().take(4).map(|v| format!("{:?}", v)).join(", ")
                )
            }
        } else {
            let data = self.to_array_view::<D>()?;
            format!("shape:{:?} ({}) {:?}", self.shape, D::name(), data)
        };
        Ok(s)
    }

    pub fn dump(&self, force_full: bool) -> TractResult<String> {
        dispatch_datum!(Self::dump_t(self.dt)(self, force_full))
    }

    pub fn close_enough(&self, other: &Self, approx: bool) -> bool {
        if self.is_null() != other.is_null() {
            return false;
        }
        let ma = self.cast_to::<f32>().unwrap();
        let ma = ma.to_array_view::<f32>().unwrap();
        let mb = other.cast_to::<f32>().unwrap();
        let mb = mb.to_array_view::<f32>().unwrap();
        let avg = ma
            .iter()
            .filter(|a| a.is_finite())
            .map(|&a| a.abs())
            .sum::<f32>()
            / ma.len() as f32;
        let dev = (ma
            .iter()
            .filter(|a| a.is_finite())
            .map(|&a| (a - avg).powi(2))
            .sum::<f32>()
            / ma.len() as f32)
            .sqrt();
        let margin = if approx {
            (dev / 5.0).max(avg.abs() / 10_000.0).max(1e-5)
        } else {
            0.0
        };
        trace!("close_enough 4");
        ma.shape() == mb.shape()
            && mb
                .iter()
                .zip(ma.iter())
                .map(|(&a, &b)| {
                    (
                        a,
                        b,
                        a.is_nan() && b.is_nan() || a == b || (b - a).abs() <= margin,
                    )
                })
                //.inspect(|t| println!("{:?}", t))
                .all(|t| t.2)
    }

    pub fn into_array<D: Datum>(self) -> TractResult<ArrayD<D>> {
        if self.is_null() {
            bail!("Null tensor")
        }
        let casted = unsafe { vec_to_datum::<D>(self.data) };
        unsafe { Ok(ArrayD::from_shape_vec_unchecked(&*self.shape, casted)) }
    }

    pub fn as_ptr<D: Datum>(&self) -> TractResult<*const D> {
        if self.is_null() {
            bail!("Null tensor")
        }
        Ok(self.data.as_ptr() as *const D)
    }

    pub fn as_slice<D: Datum>(&self) -> TractResult<&[D]> {
        if self.is_null() {
            bail!("Null tensor")
        }
        let datum_size = ::std::mem::size_of::<D>();
        unsafe {
            Ok(std::slice::from_raw_parts::<D>(
                self.data.as_ptr() as *const D,
                self.data.len() / datum_size,
            ))
        }
    }

    pub fn to_array_view<'a, D: Datum>(&'a self) -> TractResult<ArrayViewD<'a, D>> {
        assert_eq!(D::datum_type(), self.datum_type());
        if self.is_null() {
            bail!("Null tensor")
        }
        if self.data.len() != 0 {
            unsafe {
                return Ok(ArrayViewD::from_shape_ptr(
                    &*self.shape,
                    self.data.as_ptr() as _,
                ));
            }
        } else {
            return Ok(ArrayViewD::from_shape(&*self.shape, &[])?);
        }
    }

    pub fn as_slice_mut<D: Datum>(&mut self) -> TractResult<&mut [D]> {
        if self.is_null() {
            bail!("Null tensor")
        }
        let datum_size = ::std::mem::size_of::<D>();
        unsafe {
            Ok(std::slice::from_raw_parts_mut::<D>(
                self.data.as_mut_ptr() as *mut D,
                self.data.len() / datum_size,
            ))
        }
    }

    pub fn as_ptr_mut<D: Datum>(&mut self) -> TractResult<*mut D> {
        if self.is_null() {
            bail!("Null tensor")
        }
        Ok(self.data.as_mut_ptr() as *mut D)
    }

    pub fn to_array_view_mut<'a, D: Datum>(&'a mut self) -> TractResult<ArrayViewMutD<'a, D>> {
        assert_eq!(D::datum_type(), self.datum_type());
        if self.is_null() {
            bail!("Null tensor")
        }
        let shape = self.shape.clone();
        if self.data.len() != 0 {
            unsafe {
                Ok(ArrayViewMutD::from_shape_ptr(
                    &*shape,
                    self.data.as_mut_ptr() as _,
                ))
            }
        } else {
            return Ok(ArrayViewMutD::from_shape(&*self.shape, &mut [])?);
        }
    }

    pub fn to_scalar<'a, D: Datum>(&'a self) -> TractResult<D> {
        if self.is_null() {
            bail!("Null tensor")
        }
        unsafe { Ok(*(self.data.as_ptr() as *const D)) }
    }

    fn cast_data<Source: Datum + TryInto<Target>, Target: Datum>(
        &self,
    ) -> TractResult<Vec<Target>> {
        self.as_slice::<Source>()?
            .iter()
            .map(|s| s.try_into())
            .collect()
    }

    fn cast<Source: Datum + TryInto<Target>, Target: Datum>(&self) -> TractResult<Tensor> {
        let data = self.cast_data::<Source, Target>()?;
        Ok(Tensor {
            null: self.null,
            dt: Target::datum_type(),
            shape: self.shape.clone(),
            data: vec_to_u8(data),
            alignment: Target::datum_type().alignment(),
        })
    }

    pub fn cast_to<D: Datum>(&self) -> TractResult<Cow<Tensor>> {
        self.cast_to_dt(D::datum_type())
    }

    pub fn cast_to_dt(&self, dt: DatumType) -> TractResult<Cow<Tensor>> {
        use crate::DatumType::*;
        if self.dt == dt {
            return Ok(Cow::Borrowed(self));
        }
        let target = match (self.dt, dt) {
            (TDim, I32) => self.cast::<crate::dim::TDim, i32>()?,
            (I32, TDim) => self.cast::<i32, crate::dim::TDim>()?,

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
            _ => bail!("Unsupported cast from {:?} to {:?}", self.dt, dt),
        };
        Ok(Cow::Owned(target))
    }

    fn eq_t<D: Datum>(&self, other: &Tensor) -> TractResult<bool> {
        Ok(self.to_array_view::<D>()? == other.to_array_view::<D>()?)
    }

    fn eq_dt(&self, other: &Tensor) -> TractResult<bool> {
        dispatch_datum!(Self::eq_t(self.dt)(self, other))
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Tensor) -> bool {
        if self.dt != other.dt || self.shape != other.shape {
            return false;
        }
        if &*self.data == &*other.data {
            return true;
        }
        self.eq_dt(other).unwrap_or(false)
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let content = self
            .dump(false)
            .unwrap_or_else(|e| format!("Error : {:?}", e));
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
                let data = (
                    stringify!($type),
                    self.shape(),
                    $m.iter().cloned().collect::<Vec<_>>(),
                );
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

fn vec_to_u8<T: Datum>(mut data: Vec<T>) -> Vec<u8> {
    let v = unsafe {
        Vec::from_raw_parts(
            data.as_mut_ptr() as *mut u8,
            data.len() * ::std::mem::size_of::<T>(),
            data.capacity() * ::std::mem::size_of::<T>(),
        )
    };
    ::std::mem::forget(data);
    v
}

unsafe fn vec_to_datum<T: Datum>(mut data: Vec<u8>) -> Vec<T> {
    assert!(data.as_ptr() as usize % T::datum_type().alignment() == 0);
    let v = Vec::from_raw_parts(
        data.as_mut_ptr() as *mut T,
        data.len() / ::std::mem::size_of::<T>(),
        data.capacity() / ::std::mem::size_of::<T>(),
    );
    ::std::mem::forget(data);
    v
}

impl<D: ::ndarray::Dimension, T: Datum> From<Array<T, D>> for Tensor {
    fn from(it: Array<T, D>) -> Tensor {
        let shape = it.shape().into();
        let data = if it.is_standard_layout() {
            it.into_raw_vec()
        } else {
            it.view().into_iter().cloned().collect()
        };
        let raw_data = vec_to_u8(data);
        Tensor {
            null: false,
            dt: T::datum_type(),
            shape,
            data: raw_data,
            alignment: T::datum_type().alignment(),
        }
    }
}

pub fn arr4<A, V, U, T>(xs: &[V]) -> ::ndarray::Array4<A>
where
    V: ::ndarray::FixedInitializer<Elem = U> + Clone,
    U: ::ndarray::FixedInitializer<Elem = T> + Clone,
    T: ::ndarray::FixedInitializer<Elem = A> + Clone,
    A: Clone,
{
    use ndarray::*;
    let mut xs = xs.to_vec();
    let dim = Ix4(xs.len(), V::len(), U::len(), T::len());
    let ptr = xs.as_mut_ptr();
    let len = xs.len();
    let cap = xs.capacity();
    let expand_len = len * V::len() * U::len() * T::len();
    ::std::mem::forget(xs);
    unsafe {
        let v = if ::std::mem::size_of::<A>() == 0 {
            Vec::from_raw_parts(ptr as *mut A, expand_len, expand_len)
        } else if V::len() == 0 || U::len() == 0 || T::len() == 0 {
            Vec::new()
        } else {
            let expand_cap = cap * V::len() * U::len() * T::len();
            Vec::from_raw_parts(ptr as *mut A, expand_len, expand_cap)
        };
        ArrayBase::from_shape_vec_unchecked(dim, v)
    }
}
