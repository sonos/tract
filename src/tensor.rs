//! `Tensor` is the equivalent of Tensorflow Tensor.
use dim::TDim;
use ndarray::prelude::*;
use std::fmt;
use TfdResult;

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
    F32,
    F64,
    TDim,
    String,
}

impl DatumType {
    pub fn super_types(&self) -> &'static [DatumType] {
        match self {
            DatumType::Bool => &[DatumType::Bool],
            DatumType::U8 => &[DatumType::U8, DatumType::I16, DatumType::I32, DatumType::I64,
DatumType::TDim],
            DatumType::U16 => &[DatumType::U16, DatumType::I32, DatumType::I64, DatumType::TDim],
            DatumType::I8 => &[DatumType::I8, DatumType::I16, DatumType::I32, DatumType::I64, DatumType::TDim],
            DatumType::I16 => &[DatumType::I16, DatumType::I32, DatumType::I64, DatumType::TDim],
            DatumType::I32 => &[DatumType::I32, DatumType::I64, DatumType::TDim],
            DatumType::I64 => &[DatumType::I64, DatumType::TDim],
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

pub trait Datum:
    Copy
    + Clone
    + Send
    + Sync
    + fmt::Debug
    + Default
    + 'static
{
    fn name() -> &'static str;
    fn datum_type() -> DatumType;

    fn tensor_into_array(m: Tensor) -> TfdResult<ArrayD<Self>>;
    fn tensor_cast_to_array(m: &Tensor) -> TfdResult<MaybeOwnedArray<Self>>;
    fn tensor_to_view(m: &Tensor) -> TfdResult<ArrayViewD<Self>>;
    fn array_into_tensor(m: ArrayD<Self>) -> Tensor;
}

#[derive(Clone, PartialEq)]
pub enum Tensor {
    Bool(ArrayD<bool>),
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

impl Tensor {
    pub unsafe fn from_raw<T: Datum>(shape: &[usize], content: &[u8]) -> TfdResult<Tensor> {
        let value: Vec<T> = 
            ::std::slice::from_raw_parts(
                content.as_ptr() as _,
                content.len() / ::std::mem::size_of::<T>(),
            )
        .to_vec();
        Ok(ArrayD::from_shape_vec(shape, value)?.into())
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            &Tensor::Bool(ref it) => it.shape(),
            &Tensor::U8(ref it) => it.shape(),
            &Tensor::U16(ref it) => it.shape(),
            &Tensor::I8(ref it) => it.shape(),
            &Tensor::I16(ref it) => it.shape(),
            &Tensor::I32(ref it) => it.shape(),
            &Tensor::I64(ref it) => it.shape(),
            &Tensor::F32(ref it) => it.shape(),
            &Tensor::F64(ref it) => it.shape(),
            &Tensor::TDim(ref it) => it.shape(),
            &Tensor::String(ref it) => it.shape(),
        }
    }

    pub fn reduce(&mut self) {
        match self {
            Tensor::TDim(ref mut it) => {
                it.mapv_inplace(|mut e| {
                    e.reduce();
                    e
                });
            }
            _ => (),
        }
    }

    pub fn datum_type(&self) -> DatumType {
        match self {
            &Tensor::Bool(_) => DatumType::Bool,
            &Tensor::U8(_) => DatumType::U8,
            &Tensor::U16(_) => DatumType::U16,
            &Tensor::I8(_) => DatumType::I8,
            &Tensor::I16(_) => DatumType::I16,
            &Tensor::I32(_) => DatumType::I32,
            &Tensor::I64(_) => DatumType::I64,
            &Tensor::F32(_) => DatumType::F32,
            &Tensor::F64(_) => DatumType::F64,
            &Tensor::TDim(_) => DatumType::TDim,
            &Tensor::String(_) => DatumType::String,
        }
    }

    pub fn axis_chunks(&self, axis: usize, size: usize) -> TfdResult<Vec<Tensor>> {
        match self {
            &Tensor::Bool(_) => self.axis_chunks_t::<bool>(axis, size),
            &Tensor::U8(_) => self.axis_chunks_t::<u8>(axis, size),
            &Tensor::U16(_) => self.axis_chunks_t::<u16>(axis, size),
            &Tensor::I8(_) => self.axis_chunks_t::<i8>(axis, size),
            &Tensor::I16(_) => self.axis_chunks_t::<i16>(axis, size),
            &Tensor::I32(_) => self.axis_chunks_t::<i32>(axis, size),
            &Tensor::I64(_) => self.axis_chunks_t::<i64>(axis, size),
            &Tensor::F32(_) => self.axis_chunks_t::<f32>(axis, size),
            &Tensor::F64(_) => self.axis_chunks_t::<f64>(axis, size),
            &Tensor::TDim(_) => self.axis_chunks_t::<TDim>(axis, size),
            &Tensor::String(_) => bail!("String is not a datum")
        }
    }

    pub fn axis_chunks_t<T: Datum>(&self, axis: usize, size: usize) -> TfdResult<Vec<Tensor>> {
        let array = T::tensor_to_view(self)?;
        Ok(array
            .axis_chunks_iter(Axis(axis), size)
            .map(|v| T::array_into_tensor(v.to_owned()))
            .collect())
    }


    pub fn dump(&self, force_full:bool) -> TfdResult<String> {
        macro_rules! fmt_scalar {
            ($a:ident) => { format!(
                    "Scalar {:?} {:?}",
                    self.datum_type(),
                    $a.as_slice().unwrap()[0]
                )
            }
        }
        macro_rules! fmt_big {
            ($a:ident) => {
                if force_full {
                    format!(
                        "shape:{:?} {:?} {:?}",
                        self.shape(),
                        self.datum_type(),
                        $a)
                } else {
                    format!(
                        "shape:{:?} {:?} {}...",
                        self.shape(),
                        self.datum_type(),
                        $a.iter().take(4).map(|v| format!("{:?}", v)).join(", ")
                    )
                }
            }
        }
        macro_rules! fmt_small {
            ($a:ident) => { format!("{:?} {:?}", self.datum_type(), $a).replace("\n", " ") }
        }
        if self.shape().len() == 0 {
            Ok(match self {
                &Tensor::Bool(ref a) => fmt_scalar!(a),
                &Tensor::U8(ref a) => fmt_scalar!(a),
                &Tensor::U16(ref a) => fmt_scalar!(a),
                &Tensor::I8(ref a) => fmt_scalar!(a),
                &Tensor::I16(ref a) => fmt_scalar!(a),
                &Tensor::I32(ref a) => fmt_scalar!(a),
                &Tensor::I64(ref a) => fmt_scalar!(a),
                &Tensor::F32(ref a) => fmt_scalar!(a),
                &Tensor::F64(ref a) => fmt_scalar!(a),
                &Tensor::String(ref a) => fmt_scalar!(a),
                &Tensor::TDim(ref a) => fmt_scalar!(a),
            })
        } else if self.shape().iter().product::<usize>() > 8 {
            use itertools::Itertools;
            Ok(match self {
                &Tensor::Bool(ref a) => fmt_big!(a),
                &Tensor::U8(ref a) => fmt_big!(a),
                &Tensor::U16(ref a) => fmt_big!(a),
                &Tensor::I8(ref a) => fmt_big!(a),
                &Tensor::I16(ref a) => fmt_big!(a),
                &Tensor::I32(ref a) => fmt_big!(a),
                &Tensor::I64(ref a) => fmt_big!(a),
                &Tensor::F32(ref a) => fmt_big!(a),
                &Tensor::F64(ref a) => fmt_big!(a),
                &Tensor::String(ref a) => fmt_big!(a),
                &Tensor::TDim(ref a) => fmt_big!(a),
            })
        } else {
            Ok(match self {
                &Tensor::Bool(ref a) => fmt_small!(a),
                &Tensor::U8(ref a) => fmt_small!(a),
                &Tensor::U16(ref a) => fmt_small!(a),
                &Tensor::I8(ref a) => fmt_small!(a),
                &Tensor::I16(ref a) => fmt_small!(a),
                &Tensor::I32(ref a) => fmt_small!(a),
                &Tensor::I64(ref a) => fmt_small!(a),
                &Tensor::F32(ref a) => fmt_small!(a),
                &Tensor::F64(ref a) => fmt_small!(a),
                &Tensor::String(ref a) => fmt_small!(a),
                &Tensor::TDim(ref a) => fmt_small!(a),
            })
        }
    }

    fn approx(&self) -> ArrayD<f32> {
        match self {
            &Tensor::Bool(ref data) => data.map(|&a| a as u32 as f32),
            &Tensor::U8(ref data) => data.map(|&a| a as f32),
            &Tensor::U16(ref data) => data.map(|&a| a as f32),
            &Tensor::I8(ref data) => data.map(|&a| a as f32),
            &Tensor::I16(ref data) => data.map(|&a| a as f32),
            &Tensor::I32(ref data) => data.map(|&a| a as f32),
            &Tensor::I64(ref data) => data.map(|&a| a as f32),
            &Tensor::F32(ref data) => data.to_owned(),
            &Tensor::F64(ref data) => data.map(|&a| a as f32),
            &Tensor::TDim(ref data) => data.map(|&a| a.to_integer().unwrap() as f32),
            &Tensor::String(_) => unimplemented!("not supported for string"),
        }
    }

    pub fn close_enough(&self, other: &Self, approx: bool) -> bool {
        let ma = self.approx();
        let mb = other.approx();
        let avg = ma.iter().filter(|a| a.is_finite()).map(|&a| a.abs()).sum::<f32>() / ma.len() as f32;
        let dev = (ma.iter().filter(|a| a.is_finite()).map(|&a| (a - avg).powi(2)).sum::<f32>() / ma.len() as f32).sqrt();
        let margin = if approx {
            (dev / 5.0).max(avg.abs() / 10_000.0).max(1e-5)
        } else {
            0.0
        };
        ma.shape() == mb.shape()
            && mb
                .iter()
                .zip(ma.iter())
                .map(|(&a, &b)| (a, b, a.is_nan() && b.is_nan() || a == b || (b - a).abs() <= margin))
                .inspect(|t| println!("{:?}",t))
                .all(|t| t.2)
    }

    pub fn into_array<D: Datum>(self) -> TfdResult<ArrayD<D>> {
        <D as Datum>::tensor_into_array(self)
    }

    pub fn to_array_view<'a, D: Datum>(&'a self) -> TfdResult<ArrayViewD<'a, D>> {
        <D as Datum>::tensor_to_view(self)
    }

    pub fn cast_to_array<D: Datum>(&self) -> TfdResult<MaybeOwnedArray<D>> {
        <D as Datum>::tensor_cast_to_array(self)
    }

    pub fn cast_to<D: Datum>(&self) -> TfdResult<Tensor> {
        Ok(<D as Datum>::tensor_cast_to_array(&self)?.into_owned().into())
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let content = self.dump(false).unwrap_or("Error".to_string());
        write!(formatter, "Tensor {}", content)
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
            F32(m) => serialize_inner!(f32, m),
            F64(m) => serialize_inner!(f64, m),
            TDim(m) => serialize_inner!(TDim, m),
            String(m) => serialize_inner!(str, m),
        }
    }
}

impl<D: ::ndarray::Dimension, T: Datum> From<Array<T, D>> for Tensor {
    fn from(it: Array<T, D>) -> Tensor {
        T::array_into_tensor(it.into_dyn())
    }
}

macro_rules! tensor {
    ($t:ident, $v:ident, $as_one:ident, $as:ident, $take:ident, $make:ident,
        [$(($cast:ident, $as_cast:ident)),*]) => {
        impl From<$t> for Tensor {
            fn from(it: $t) -> Tensor {
                Tensor::$v(arr0(it).into_dyn())
            }
        }

        impl Tensor {
            pub fn $as_one(&self) -> Option<$t> {
                if let &Tensor::$v(ref it) = self {
                    if it.shape().len() == 0 {
                        Some(*it.iter().next().unwrap())
                    } else {
                        None
                    }
                } else {
                    None
                }
            }

            pub fn $as(&self) -> Option<&ArrayD<$t>> {
                if let &Tensor::$v(ref it) = self {
                    Some(it)
                } else {
                    None
                }
            }

            pub fn $take(self) -> Option<ArrayD<$t>> {
                if let Tensor::$v(it) = self {
                    Some(it)
                } else {
                    None
                }
            }

            pub fn $make(shape: &[usize], values: &[$t]) -> TfdResult<Tensor> {
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

            fn tensor_into_array(m: Tensor) -> TfdResult<ArrayD<Self>> {
                let _ = Self::tensor_to_view(&m)?;
                Ok(m.$take().unwrap())
            }

            fn tensor_to_view(m: &Tensor) -> TfdResult<ArrayViewD<Self>> {
                m.$as()
                    .map(|m| m.view())
                    .ok_or_else(|| format!("Type mismatch unwrapping to {}: {:?}", Self::name(), &m).into())
            }

            fn array_into_tensor(m: ArrayD<Self>) -> Tensor {
                Tensor::$v(m)
            }

            fn tensor_cast_to_array(m: &Tensor) -> TfdResult<MaybeOwnedArray<$t>> {
                match m.datum_type() {
                    DatumType::$v => Ok(MaybeOwnedArray::View($t::tensor_to_view(m)?)),
                    $(DatumType::$cast => {
                        let src = m.$as_cast().ok_or("Wrong type")?;
                        let vec:Vec<$t> = src.iter()
                            .map(|x| TryInto::<$t>::try_into(*x))
                            .collect::<TfdResult<Vec<_>>>()?;
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
    fn try_into(self) -> TfdResult<D>;
}

impl<F: ::num::cast::AsPrimitive<D>, D: Datum> TryInto<D> for F {
    fn try_into(self) -> TfdResult<D> {
        Ok(self.as_())
    }
}

impl TryInto<TDim> for i32 {
    fn try_into(self) -> TfdResult<TDim> {
        Ok(self.into())
    }
}

impl TryInto<i32> for TDim {
    fn try_into(self) -> TfdResult<i32> {
        self.to_integer().map(|i| i as i32)
    }
}

impl TryInto<i64> for TDim {
    fn try_into(self) -> TfdResult<i64> {
        self.to_integer().map(|i| i as i64)
    }
}

tensor!(bool, Bool, as_bool, as_bools, take_bools, bools, []);
tensor!(f64, F64, as_f64, as_f64s, take_f64s, f64s, []);
tensor!(f32, F32, as_f32, as_f32s, take_f32s, f32s, []);
tensor!(i8, I8, as_i8, as_i8s, take_i8s, i8s, []);
tensor!(i16, I16, as_i16, as_i16s, take_i16s, i16s,
        [(I8, as_i8s)]
);
tensor!(
    i32,
    I32,
    as_i32,
    as_i32s,
    take_i32s,
    i32s,
    [(TDim, as_dims), (I8, as_i8s), (I16, as_i16s)]
);
tensor!(
    i64,
    I64,
    as_i64,
    as_i64s,
    take_i64s,
    i64s,
    [(TDim, as_dims), (I8, as_i8s), (I16, as_i16s), (I32, as_i32s)]
);
tensor!(u8, U8, as_u8, as_u8s, take_u8s, u8s, []);
tensor!(u16, U16, as_u16, as_u16s, take_u16s, u16s, []);
tensor!(
    TDim,
    TDim,
    as_dim,
    as_dims,
    take_dims,
    dims,
    [(I32, as_i32s)]
);

#[cfg(test)]
mod tests {
    use dim::ToDim;
    use tensor::*;

    #[test]
    fn test_cast_dim_to_dim() {
        let t_dim: Tensor = arr1(&[0isize.to_dim(), 0isize.to_dim()]).into();
        let _dims: MaybeOwnedArray<TDim> = TDim::tensor_cast_to_array(&t_dim).unwrap();
    }

    #[test]
    fn test_cast_i32_to_dim() {
        let t_i32: Tensor = arr1(&[0i32, 0]).into();
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
