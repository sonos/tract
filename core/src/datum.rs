//! `DtArray` is the equivalent of Tensorflow DtArray.
use dim::TDim;
use model::TVec;
use ndarray::prelude::*;
use std::borrow::Cow;
use std::fmt;
use TractResult;

use f16::f16;

#[cfg(feature = "serialize")]
use serde::ser::{Serialize, Serializer};

#[derive(Debug, Copy, Clone, PartialEq)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
pub enum DatumType {
    Bool,
    U8,
    U16,
    I8,
    I16,
    I32,
    I64,
    F16,
    F32,
    F64,
    TDim,
    String,
}

impl DatumType {
    pub fn super_types(&self) -> &'static [DatumType] {
        match self {
            DatumType::Bool => &[DatumType::Bool],
            DatumType::U8 => &[
                DatumType::U8,
                DatumType::I16,
                DatumType::I32,
                DatumType::I64,
                DatumType::TDim,
            ],
            DatumType::U16 => &[
                DatumType::U16,
                DatumType::I32,
                DatumType::I64,
                DatumType::TDim,
            ],
            DatumType::I8 => &[
                DatumType::I8,
                DatumType::I16,
                DatumType::I32,
                DatumType::I64,
                DatumType::TDim,
            ],
            DatumType::I16 => &[
                DatumType::I16,
                DatumType::I32,
                DatumType::I64,
                DatumType::TDim,
            ],
            DatumType::I32 => &[DatumType::I32, DatumType::I64, DatumType::TDim],
            DatumType::I64 => &[DatumType::I64, DatumType::TDim],
            DatumType::F16 => &[DatumType::F16, DatumType::F32, DatumType::F64],
            DatumType::F32 => &[DatumType::F32, DatumType::F64],
            DatumType::F64 => &[DatumType::F64],
            DatumType::String => &[DatumType::String],
            DatumType::TDim => &[DatumType::TDim],
        }
    }

    pub fn super_type_for<I: IntoIterator<Item = DatumType>>(i: I) -> Option<DatumType> {
        let mut iter = i.into_iter();
        let mut current = match iter.next() {
            None => return None,
            Some(it) => it,
        };
        while let Some(n) = iter.next() {
            match current.common_super_type(n) {
                None => return None,
                Some(it) => current = it,
            }
        }
        Some(current)
    }

    pub fn common_super_type(&self, rhs: DatumType) -> Option<DatumType> {
        for mine in self.super_types() {
            for theirs in rhs.super_types() {
                if mine == theirs {
                    return Some(*mine);
                }
            }
        }
        return None;
    }
}

pub trait Datum:
    Copy + Clone + Send + Sync + fmt::Debug + fmt::Display + Default + 'static + PartialEq
{
    fn name() -> &'static str;
    fn datum_type() -> DatumType;
}

#[derive(Clone)]
pub struct DtArray {
    dt: DatumType,
    shape: TVec<usize>,
    data: Vec<u8>,
}

impl DtArray {
    pub unsafe fn from_raw<T: Datum>(shape: &[usize], content: &[u8]) -> TractResult<DtArray> {
        Ok(DtArray {
            dt: T::datum_type(),
            shape: shape.into(),
            data: content.to_vec(),
        })
    }

    pub fn into_tensor(self) -> ::Tensor {
        ::Tensor::from(self)
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn into_shape(self, shape: &[usize]) -> TractResult<DtArray> {
        Ok(DtArray {
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
        ma.shape() == mb.shape() && mb
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
        let casted = unsafe { vec_to_datum::<D>(self.data) };
        unsafe { Ok(ArrayD::from_shape_vec_unchecked(&*self.shape, casted)) }
    }

    pub fn as_slice<D: Datum>(&self) -> TractResult<&[D]> {
        let datum_size = ::std::mem::size_of::<D>();
        unsafe {
            Ok(std::slice::from_raw_parts::<D>(
                self.data.as_ptr() as *const D,
                self.data.len() / datum_size,
            ))
        }
    }

    pub fn to_array_view<'a, D: Datum>(&'a self) -> TractResult<ArrayViewD<'a, D>> {
        unsafe {
            Ok(ArrayViewD::from_shape_ptr(
                &*self.shape,
                self.data.as_ptr() as _,
            ))
        }
    }

    pub fn as_slice_mut<D: Datum>(&mut self) -> TractResult<&mut [D]> {
        let datum_size = ::std::mem::size_of::<D>();
        unsafe {
            Ok(std::slice::from_raw_parts_mut::<D>(
                self.data.as_mut_ptr() as *mut D,
                self.data.len() / datum_size,
            ))
        }
    }

    pub fn to_array_view_mut<'a, D: Datum>(&'a mut self) -> TractResult<ArrayViewMutD<'a, D>> {
        let shape = self.shape.clone();
        unsafe {
            Ok(ArrayViewMutD::from_shape_ptr(
                &*shape,
                self.data.as_mut_ptr() as _,
            ))
        }
    }

    pub fn to_scalar<'a, D: Datum>(&'a self) -> TractResult<D> {
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

    fn cast<Source: Datum + TryInto<Target>, Target: Datum>(&self) -> TractResult<DtArray> {
        let data = self.cast_data::<Source, Target>()?;
        Ok(DtArray {
            dt: Target::datum_type(),
            shape: self.shape.clone(),
            data: vec_to_u8(data),
        })
    }

    pub fn cast_to<D: Datum>(&self) -> TractResult<Cow<DtArray>> {
        self.cast_to_dt(D::datum_type())
    }

    pub fn cast_to_dt(&self, dt: DatumType) -> TractResult<Cow<DtArray>> {
        use DatumType::*;
        if self.dt == dt {
            return Ok(Cow::Borrowed(self));
        }
        let target = match (self.dt, dt) {
            (TDim, I32) => self.cast::<::dim::TDim, i32>()?,
            (I32, TDim) => self.cast::<i32, ::dim::TDim>()?,

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

    fn eq_t<D: Datum>(&self, other: &DtArray) -> TractResult<bool> {
        Ok(self.to_array_view::<D>()? == other.to_array_view::<D>()?)
    }

    fn eq_dt(&self, other: &DtArray) -> TractResult<bool> {
        dispatch_datum!(Self::eq_t(self.dt)(self, other))
    }
}

impl PartialEq for DtArray {
    fn eq(&self, other: &DtArray) -> bool {
        if self.dt != other.dt || self.shape != other.shape {
            return false;
        }
        if &*self.data == &*other.data {
            return true;
        }
        self.eq_dt(other).unwrap_or(false)
    }
}

impl fmt::Debug for DtArray {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let content = self.dump(false).unwrap_or("Error".to_string());
        write!(formatter, "DtArray {}", content)
    }
}

#[cfg(feature = "serialize")]
impl Serialize for DtArray {
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

        use DtArray::*;
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
    let v = Vec::from_raw_parts(
        data.as_mut_ptr() as *mut T,
        data.len() / ::std::mem::size_of::<T>(),
        data.capacity() / ::std::mem::size_of::<T>(),
    );
    ::std::mem::forget(data);
    v
}

impl<D: ::ndarray::Dimension, T: Datum> From<Array<T, D>> for DtArray {
    fn from(it: Array<T, D>) -> DtArray {
        let shape = it.shape().into();
        let data = if it.is_standard_layout() {
            it.into_raw_vec()
        } else {
            it.view().into_iter().cloned().collect()
        };
        let raw_data = vec_to_u8(data);
        DtArray {
            dt: T::datum_type(),
            shape,
            data: raw_data,
        }
    }
}

macro_rules! datum {
    ($t:ident, $v:ident) => {
        impl From<$t> for DtArray {
            fn from(it: $t) -> DtArray {
                arr0(it).into()
            }
        }

        impl Datum for $t {
            fn name() -> &'static str {
                stringify!($t)
            }

            fn datum_type() -> DatumType {
                DatumType::$v
            }
        }
    };
}

trait TryInto<D: Datum> {
    fn try_into(self) -> TractResult<D>;
}

macro_rules! try_into {
    ($f:ty, $t:ty) => {
        impl TryInto<$t> for $f {
            fn try_into(self) -> TractResult<$t> {
                Ok(self as $t)
            }
        }
    };
}

try_into!(i8, i16);
try_into!(i8, i32);
try_into!(i8, i64);
try_into!(i16, i32);
try_into!(i16, i64);
try_into!(i32, i64);

try_into!(i16, i8);
try_into!(i32, i8);
try_into!(i64, i8);
try_into!(i32, i16);
try_into!(i64, i16);
try_into!(i64, i32);

try_into!(f64, f32);
try_into!(f32, f64);

try_into!(i8, f32);
try_into!(i16, f32);
try_into!(i32, f32);
try_into!(i64, f32);

impl TryInto<TDim> for i32 {
    fn try_into(self) -> TractResult<TDim> {
        Ok(self.into())
    }
}

impl TryInto<i32> for TDim {
    fn try_into(self) -> TractResult<i32> {
        self.to_integer().map(|i| i as i32)
    }
}

impl TryInto<i64> for TDim {
    fn try_into(self) -> TractResult<i64> {
        self.to_integer().map(|i| i as i64)
    }
}

impl TryInto<f32> for bool {
    fn try_into(self) -> TractResult<f32> {
        if self {
            Ok(1.0)
        } else {
            Ok(0.0)
        }
    }
}

impl TryInto<f32> for f16 {
    fn try_into(self) -> TractResult<f32> {
        Ok(self.0.to_f32())
    }
}

impl TryInto<f64> for f16 {
    fn try_into(self) -> TractResult<f64> {
        Ok(self.0.to_f64())
    }
}

impl TryInto<f16> for f32 {
    fn try_into(self) -> TractResult<f16> {
        Ok(f16(half::f16::from_f32(self)))
    }
}

impl TryInto<f16> for f64 {
    fn try_into(self) -> TractResult<f16> {
        Ok(f16(half::f16::from_f64(self)))
    }
}

datum!(bool, Bool);
datum!(f16, F16);
datum!(f32, F32);
datum!(f64, F64);
datum!(i8, I8);
datum!(i16, I16);
datum!(i32, I32);
datum!(i64, I64);
datum!(u8, U8);
datum!(u16, U16);
datum!(TDim, TDim);

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

#[cfg(test)]
mod tests {
    use datum::*;
    use dim::ToDim;

    #[test]
    fn test_array_to_tensor_to_array() {
        let array = arr1(&[12i32, 42]);
        let dt_array = DtArray::from(array.clone());
        let view = dt_array.to_array_view::<i32>().unwrap();
        assert_eq!(array, view.into_dimensionality().unwrap());
    }

    #[test]
    fn test_cast_dim_to_dim() {
        let t_dim: DtArray = arr1(&[12isize.to_dim(), 42isize.to_dim()]).into();
        let t_i32 = t_dim.cast_to::<i32>().unwrap();
        let t_dim_2 = t_i32.cast_to::<TDim>().unwrap().into_owned();
        assert_eq!(t_dim, t_dim_2);
    }

    #[test]
    fn test_cast_i32_to_dim() {
        let t_i32: DtArray = arr1(&[0i32, 0]).into();
        t_i32.cast_to::<TDim>().unwrap();
    }
}
