//! `DtArray` is the equivalent of Tensorflow DtArray.
use dim::TDim;
use ndarray::prelude::*;
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

pub enum MaybeOwnedArray<'a, T: 'a> {
    Owned(ArrayD<T>),
    View(ArrayViewD<'a, T>),
}

impl<'a, T: 'a> MaybeOwnedArray<'a, T> {
    pub fn view(&self) -> ArrayViewD<T> {
        match self {
            MaybeOwnedArray::Owned(it) => it.view(),
            MaybeOwnedArray::View(it) => it.view(),
        }
    }
}

impl<'a, T: Clone + 'a> MaybeOwnedArray<'a, T> {
    pub fn into_owned(self) -> ArrayD<T> {
        match self {
            MaybeOwnedArray::Owned(it) => it,
            MaybeOwnedArray::View(it) => it.view().to_owned(),
        }
    }
}

pub trait Datum: Copy + Clone + Send + Sync + fmt::Debug + Default + 'static {
    fn name() -> &'static str;
    fn datum_type() -> DatumType;

    fn tensor_into_array(m: DtArray) -> TractResult<ArrayD<Self>>;
    fn tensor_cast_to_array(m: &DtArray) -> TractResult<MaybeOwnedArray<Self>>;
    fn tensor_to_view(m: &DtArray) -> TractResult<ArrayViewD<Self>>;
    fn tensor_to_view_mut(m: &mut DtArray) -> TractResult<ArrayViewMutD<Self>>;
    fn tensor_to_scalar(m: &DtArray) -> TractResult<Self>;
    fn array_into_tensor(m: ArrayD<Self>) -> DtArray;
}

#[derive(Clone, PartialEq)]
pub enum DtArray {
    Bool(ArrayD<bool>),
    F16(ArrayD<f16>),
    F32(ArrayD<f32>),
    F64(ArrayD<f64>),
    I8(ArrayD<i8>),
    I16(ArrayD<i16>),
    I32(ArrayD<i32>),
    I64(ArrayD<i64>),
    U8(ArrayD<u8>),
    U16(ArrayD<u16>),
    TDim(ArrayD<TDim>),
    String(ArrayD<i8>),
}

impl DtArray {
    pub unsafe fn from_raw<T: Datum>(shape: &[usize], content: &[u8]) -> TractResult<DtArray> {
        let value: Vec<T> = ::std::slice::from_raw_parts(
            content.as_ptr() as _,
            content.len() / ::std::mem::size_of::<T>(),
        ).to_vec();
        Ok(ArrayD::from_shape_vec(shape, value)?.into())
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            &DtArray::Bool(ref it) => it.shape(),
            &DtArray::U8(ref it) => it.shape(),
            &DtArray::U16(ref it) => it.shape(),
            &DtArray::I8(ref it) => it.shape(),
            &DtArray::I16(ref it) => it.shape(),
            &DtArray::I32(ref it) => it.shape(),
            &DtArray::I64(ref it) => it.shape(),
            &DtArray::F16(ref it) => it.shape(),
            &DtArray::F32(ref it) => it.shape(),
            &DtArray::F64(ref it) => it.shape(),
            &DtArray::TDim(ref it) => it.shape(),
            &DtArray::String(ref it) => it.shape(),
        }
    }

    pub fn into_shape(self, shape: &[usize]) -> TractResult<DtArray> {
        let shaped = match self {
            DtArray::Bool(it) => it.into_shape(shape)?.into(),
            DtArray::U8(it) => it.into_shape(shape)?.into(),
            DtArray::U16(it) => it.into_shape(shape)?.into(),
            DtArray::I8(it) => it.into_shape(shape)?.into(),
            DtArray::I16(it) => it.into_shape(shape)?.into(),
            DtArray::I32(it) => it.into_shape(shape)?.into(),
            DtArray::I64(it) => it.into_shape(shape)?.into(),
            DtArray::F16(it) => it.into_shape(shape)?.into(),
            DtArray::F32(it) => it.into_shape(shape)?.into(),
            DtArray::F64(it) => it.into_shape(shape)?.into(),
            DtArray::TDim(it) => it.into_shape(shape)?.into(),
            DtArray::String(it) => it.into_shape(shape)?.into(),
        };
        Ok(shaped)
    }

    pub fn datum_type(&self) -> DatumType {
        match self {
            &DtArray::Bool(_) => DatumType::Bool,
            &DtArray::U8(_) => DatumType::U8,
            &DtArray::U16(_) => DatumType::U16,
            &DtArray::I8(_) => DatumType::I8,
            &DtArray::I16(_) => DatumType::I16,
            &DtArray::I32(_) => DatumType::I32,
            &DtArray::I64(_) => DatumType::I64,
            &DtArray::F16(_) => DatumType::F16,
            &DtArray::F32(_) => DatumType::F32,
            &DtArray::F64(_) => DatumType::F64,
            &DtArray::TDim(_) => DatumType::TDim,
            &DtArray::String(_) => DatumType::String,
        }
    }

    pub fn dump(&self, force_full: bool) -> TractResult<String> {
        macro_rules! fmt_scalar {
            ($a:ident) => {
                format!(
                    "Scalar {:?} {:?}",
                    self.datum_type(),
                    $a.as_slice().unwrap()[0]
                )
            };
        }
        macro_rules! fmt_big {
            ($a:ident) => {
                if force_full {
                    format!("shape:{:?} {:?} {:?}", self.shape(), self.datum_type(), $a)
                } else {
                    format!(
                        "shape:{:?} {:?} {}...",
                        self.shape(),
                        self.datum_type(),
                        $a.iter().take(4).map(|v| format!("{:?}", v)).join(", ")
                    )
                }
            };
        }
        macro_rules! fmt_small {
            ($a:ident) => {
                format!("{:?} {:?}", self.datum_type(), $a).replace("\n", " ")
            };
        }
        if self.shape().len() == 0 {
            Ok(match self {
                &DtArray::Bool(ref a) => fmt_scalar!(a),
                &DtArray::U8(ref a) => fmt_scalar!(a),
                &DtArray::U16(ref a) => fmt_scalar!(a),
                &DtArray::I8(ref a) => fmt_scalar!(a),
                &DtArray::I16(ref a) => fmt_scalar!(a),
                &DtArray::I32(ref a) => fmt_scalar!(a),
                &DtArray::I64(ref a) => fmt_scalar!(a),
                &DtArray::F16(ref a) => fmt_scalar!(a),
                &DtArray::F32(ref a) => fmt_scalar!(a),
                &DtArray::F64(ref a) => fmt_scalar!(a),
                &DtArray::String(ref a) => fmt_scalar!(a),
                &DtArray::TDim(ref a) => fmt_scalar!(a),
            })
        } else if self.shape().iter().product::<usize>() > 8 {
            use itertools::Itertools;
            Ok(match self {
                &DtArray::Bool(ref a) => fmt_big!(a),
                &DtArray::U8(ref a) => fmt_big!(a),
                &DtArray::U16(ref a) => fmt_big!(a),
                &DtArray::I8(ref a) => fmt_big!(a),
                &DtArray::I16(ref a) => fmt_big!(a),
                &DtArray::I32(ref a) => fmt_big!(a),
                &DtArray::I64(ref a) => fmt_big!(a),
                &DtArray::F16(ref a) => fmt_big!(a),
                &DtArray::F32(ref a) => fmt_big!(a),
                &DtArray::F64(ref a) => fmt_big!(a),
                &DtArray::String(ref a) => fmt_big!(a),
                &DtArray::TDim(ref a) => fmt_big!(a),
            })
        } else {
            Ok(match self {
                &DtArray::Bool(ref a) => fmt_small!(a),
                &DtArray::U8(ref a) => fmt_small!(a),
                &DtArray::U16(ref a) => fmt_small!(a),
                &DtArray::I8(ref a) => fmt_small!(a),
                &DtArray::I16(ref a) => fmt_small!(a),
                &DtArray::I32(ref a) => fmt_small!(a),
                &DtArray::I64(ref a) => fmt_small!(a),
                &DtArray::F16(ref a) => fmt_small!(a),
                &DtArray::F32(ref a) => fmt_small!(a),
                &DtArray::F64(ref a) => fmt_small!(a),
                &DtArray::String(ref a) => fmt_small!(a),
                &DtArray::TDim(ref a) => fmt_small!(a),
            })
        }
    }

    fn approx(&self) -> ArrayD<f32> {
        match self {
            &DtArray::Bool(ref data) => data.map(|&a| a as u32 as f32),
            &DtArray::U8(ref data) => data.map(|&a| a as f32),
            &DtArray::U16(ref data) => data.map(|&a| a as f32),
            &DtArray::I8(ref data) => data.map(|&a| a as f32),
            &DtArray::I16(ref data) => data.map(|&a| a as f32),
            &DtArray::I32(ref data) => data.map(|&a| a as f32),
            &DtArray::I64(ref data) => data.map(|&a| a as f32),
            &DtArray::F16(ref data) => data.map(|&a| f32::from(a.0)),
            &DtArray::F32(ref data) => data.to_owned(),
            &DtArray::F64(ref data) => data.map(|&a| a as f32),
            &DtArray::TDim(ref data) => data.map(|&a| a.to_integer().unwrap() as f32),
            &DtArray::String(_) => unimplemented!("not supported for string"),
        }
    }

    pub fn close_enough(&self, other: &Self, approx: bool) -> bool {
        let ma = self.approx();
        let mb = other.approx();
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
        <D as Datum>::tensor_into_array(self)
    }

    pub fn to_array_view<'a, D: Datum>(&'a self) -> TractResult<ArrayViewD<'a, D>> {
        <D as Datum>::tensor_to_view(self)
    }

    pub fn to_array_view_mut<'a, D: Datum>(&'a mut self) -> TractResult<ArrayViewMutD<'a, D>> {
        <D as Datum>::tensor_to_view_mut(self)
    }

    pub fn to_scalar<'a, D: Datum>(&'a self) -> TractResult<D> {
        <D as Datum>::tensor_to_scalar(self)
    }

    pub fn cast_to_array<D: Datum>(&self) -> TractResult<MaybeOwnedArray<D>> {
        <D as Datum>::tensor_cast_to_array(self)
    }

    pub fn cast_to<D: Datum>(&self) -> TractResult<DtArray> {
        Ok(<D as Datum>::tensor_cast_to_array(&self)?
            .into_owned()
            .into())
    }

    pub fn cast_to_dt(&self, dt: DatumType) -> TractResult<DtArray> {
        dispatch_datum!(Self::cast_to(dt)(self))
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

impl<D: ::ndarray::Dimension, T: Datum> From<Array<T, D>> for DtArray {
    fn from(it: Array<T, D>) -> DtArray {
        T::array_into_tensor(it.into_dyn())
    }
}

macro_rules! tensor {
    ($t:ident, $v:ident, $as_one:ident, $as:ident, $as_mut:ident, $take:ident, $make:ident,
        [$(($cast:ident, $as_cast:ident)),*]) => {
        impl From<$t> for DtArray {
            fn from(it: $t) -> DtArray {
                DtArray::$v(arr0(it).into_dyn())
            }
        }

        impl DtArray {
            #[allow(dead_code)]
            fn $as_one(&self) -> Option<$t> {
                if let &DtArray::$v(ref it) = self {
                    if it.shape().len() == 0 {
                        Some(*it.iter().next().unwrap())
                    } else {
                        None
                    }
                } else {
                    None
                }
            }

            #[allow(dead_code)]
            fn $as(&self) -> Option<&ArrayD<$t>> {
                if let &DtArray::$v(ref it) = self {
                    Some(it)
                } else {
                    None
                }
            }

            #[allow(dead_code)]
            fn $as_mut(&mut self) -> Option<&mut ArrayD<$t>> {
                if let &mut DtArray::$v(ref mut it) = self {
                    Some(it)
                } else {
                    None
                }
            }

            #[allow(dead_code)]
            fn $take(self) -> Option<ArrayD<$t>> {
                if let DtArray::$v(it) = self {
                    Some(it)
                } else {
                    None
                }
            }

            #[allow(dead_code)]
            fn $make(shape: &[usize], values: &[$t]) -> TractResult<DtArray> {
                Ok(Array::from_shape_vec(shape, values.to_vec())?.into())
            }
        }

        impl Datum for $t {
            fn name() -> &'static str {
                stringify!($t)
            }

            fn datum_type() -> DatumType {
                DatumType::$v
            }

            fn tensor_into_array(m: DtArray) -> TractResult<ArrayD<Self>> {
                let _ = Self::tensor_to_view(&m)?;
                Ok(m.$take().unwrap())
            }

            fn tensor_to_scalar(m: &DtArray) -> TractResult<Self> {
                let view = Self::tensor_to_view(m)?;
                if view.ndim() == 0 {
                    Ok(*view.iter().next().unwrap())
                } else {
                    bail!("Tensor is not a scalar")
                }
            }

            fn tensor_to_view(m: &DtArray) -> TractResult<ArrayViewD<Self>> {
                m.$as()
                    .map(|m| m.view())
                    .ok_or_else(|| format!("Type mismatch unwrapping to {}: {:?}", Self::name(), &m).into())
            }

            fn tensor_to_view_mut(m: &mut DtArray) -> TractResult<ArrayViewMutD<Self>> {
                m.$as_mut()
                    .map(|m| m.view_mut())
                    .ok_or_else(|| format!("Type mismatch unwrapping to {}", Self::name()).into())
            }

            fn array_into_tensor(m: ArrayD<Self>) -> DtArray {
                DtArray::$v(m)
            }

            fn tensor_cast_to_array(m: &DtArray) -> TractResult<MaybeOwnedArray<$t>> {
                match m.datum_type() {
                    DatumType::$v => Ok(MaybeOwnedArray::View($t::tensor_to_view(m)?)),
                    $(DatumType::$cast => {
                        let src = m.$as_cast().ok_or("Wrong type")?;
                        let vec:Vec<$t> = src.iter()
                            .map(|x| TryInto::<$t>::try_into(*x))
                            .collect::<TractResult<Vec<_>>>()?;
                        let dst:ArrayD<$t> = ArrayD::from_shape_vec(src.shape(), vec)?;
                        Ok(MaybeOwnedArray::Owned(dst))
                    })*
                    _ => bail!("Can not cast tensor from {:?} to {:?}", m.datum_type(), $t::datum_type())
                }
            }
        }
    };
}

trait TryInto<D: Datum> {
    fn try_into(self) -> TractResult<D>;
}

impl<F: ::num::cast::AsPrimitive<D>, D: Datum> TryInto<D> for F {
    fn try_into(self) -> TractResult<D> {
        Ok(self.as_())
    }
}

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

tensor!(
    bool,
    Bool,
    as_bool,
    as_bools,
    as_bools_mut,
    take_bools,
    bools,
    []
);
tensor!(
    f16,
    F16,
    as_f16,
    as_f16s,
    as_f16s_mut,
    take_f16s,
    f16s,
    [(F32, as_f32s), (F64, as_f64s)]
);
tensor!(
    f32,
    F32,
    as_f32,
    as_f32s,
    as_f32s_mut,
    take_f32s,
    f32s,
    [(F16, as_f16s), (F64, as_f64s)]
);
tensor!(
    f64,
    F64,
    as_f64,
    as_f64s,
    as_f64s_mut,
    take_f64s,
    f64s,
    [(F16, as_f16s), (F32, as_f32s)]
);
tensor!(i8, I8, as_i8, as_i8s, as_i8s_mut, take_i8s, i8s, []);
tensor!(
    i16,
    I16,
    as_i16,
    as_i16s,
    as_i16_mut,
    take_i16s,
    i16s,
    [(I8, as_i8s)]
);
tensor!(
    i32,
    I32,
    as_i32,
    as_i32s,
    as_i32_mut,
    take_i32s,
    i32s,
    [(TDim, as_dims), (I8, as_i8s), (I16, as_i16s)]
);
tensor!(
    i64,
    I64,
    as_i64,
    as_i64s,
    as_i64s_mut,
    take_i64s,
    i64s,
    [
        (TDim, as_dims),
        (I8, as_i8s),
        (I16, as_i16s),
        (I32, as_i32s)
    ]
);
tensor!(u8, U8, as_u8, as_u8s, as_u8s_mut, take_u8s, u8s, []);
tensor!(u16, U16, as_u16, as_u16s, as_u16s_mut, take_u16s, u16s, []);
tensor!(
    TDim,
    TDim,
    as_dim,
    as_dims,
    as_dims_mut,
    take_dims,
    dims,
    [(I32, as_i32s)]
);

#[cfg(test)]
mod tests {
    use datum::*;
    use dim::ToDim;

    #[test]
    fn test_cast_dim_to_dim() {
        let t_dim: DtArray = arr1(&[0isize.to_dim(), 0isize.to_dim()]).into();
        let _dims: MaybeOwnedArray<TDim> = TDim::tensor_cast_to_array(&t_dim).unwrap();
    }

    #[test]
    fn test_cast_i32_to_dim() {
        let t_i32: DtArray = arr1(&[0i32, 0]).into();
        let _dims: MaybeOwnedArray<TDim> = TDim::tensor_cast_to_array(&t_i32).unwrap();
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
