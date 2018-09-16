//! `Tensor` is the equivalent of Tensorflow Tensor.
use dim::TDim;
use ndarray::prelude::*;
use std::fmt;

use tf::tfpb;
use tf::tfpb::types::DataType as DataType;

#[cfg(feature = "serialize")]
use serde::ser::{Serialize, Serializer};

#[derive(Debug, Copy, Clone, PartialEq)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
pub enum DatumType {
    U8,
    I8,
    I32,
    F32,
    F64,
    TDim,
    String,
}

impl DatumType {
    pub fn from_pb(t: &tfpb::types::DataType) -> ::Result<DatumType> {
        match t {
            &DataType::DT_UINT8 => Ok(DatumType::U8),
            &DataType::DT_INT8 => Ok(DatumType::I8),
            &DataType::DT_INT32 => Ok(DatumType::I32),
            &DataType::DT_FLOAT => Ok(DatumType::F32),
            &DataType::DT_DOUBLE => Ok(DatumType::F64),
            &DataType::DT_STRING => Ok(DatumType::String),
            _ => Err(format!("Unknown DatumType {:?}", t))?,
        }
    }

    pub fn to_pb(&self) -> ::Result<tfpb::types::DataType> {
        match self {
            DatumType::U8 => Ok(DataType::DT_UINT8),
            DatumType::I8 => Ok(DataType::DT_INT8),
            DatumType::I32 => Ok(DataType::DT_INT32),
            DatumType::F32 => Ok(DataType::DT_FLOAT),
            DatumType::F64 => Ok(DataType::DT_DOUBLE),
            DatumType::String => Ok(DataType::DT_STRING),
            DatumType::TDim => bail!("Dimension is not translatable in protobuf"),
        }
    }

    pub fn super_types(&self) -> &'static [DatumType] {
        match self {
            DatumType::U8 => &[DatumType::U8, DatumType::I32],
            DatumType::I8 => &[DatumType::I8, DatumType::I32],
            DatumType::I32 => &[DatumType::I32, DatumType::TDim],
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
    + ::num::Zero
    + ::num::One
    + ::ndarray::LinalgScalar
    + ::std::ops::AddAssign
    + ::std::ops::MulAssign
    + ::std::ops::DivAssign
    + ::std::ops::SubAssign
    + ::std::ops::RemAssign
{
    fn name() -> &'static str;
    fn datum_type() -> DatumType;

    fn tensor_into_array(m: Tensor) -> ::Result<ArrayD<Self>>;
    fn tensor_cast_to_array(m: &Tensor) -> ::Result<MaybeOwnedArray<Self>>;
    fn tensor_to_view(m: &Tensor) -> ::Result<ArrayViewD<Self>>;
    fn array_into_tensor(m: ArrayD<Self>) -> Tensor;
}

#[derive(Clone, PartialEq)]
pub enum Tensor {
    F32(ArrayD<f32>),
    F64(ArrayD<f64>),
    I32(ArrayD<i32>),
    I8(ArrayD<i8>),
    U8(ArrayD<u8>),
    TDim(ArrayD<TDim>),
    String(ArrayD<i8>),
}

impl Tensor {
    pub fn from_pb(t: &tfpb::tensor::TensorProto) -> ::Result<Tensor> {
        use tf::tfpb::types::DataType::*;
        let dtype = t.get_dtype();
        let shape = t.get_tensor_shape();
        let dims = shape
            .get_dim()
            .iter()
            .map(|d| d.size as usize)
            .collect::<Vec<_>>();
        let rank = dims.len();
        let content = t.get_tensor_content();
        let mat: Tensor = if content.len() != 0 {
            match dtype {
                DT_FLOAT => Self::from_content::<f32, u8>(dims, content)?.into(),
                DT_INT32 => Self::from_content::<i32, u8>(dims, content)?.into(),
                _ => unimplemented!("missing type"),
            }
        } else {
            match dtype {
                DT_INT32 => Self::from_content::<i32, i32>(dims, t.get_int_val())?.into(),
                DT_FLOAT => Self::from_content::<f32, f32>(dims, t.get_float_val())?.into(),
                _ => unimplemented!("missing type"),
            }
        };
        assert_eq!(rank, mat.shape().len());
        Ok(mat)
    }

    pub fn from_content<T: Copy, V: Copy>(dims: Vec<usize>, content: &[V]) -> ::Result<ArrayD<T>> {
        let value: &[T] = unsafe {
            ::std::slice::from_raw_parts(
                content.as_ptr() as _,
                content.len() * ::std::mem::size_of::<V>() / ::std::mem::size_of::<T>(),
            )
        };
        Ok(Array1::from_iter(value.iter().cloned())
            .into_shape(dims)?
            .into_dyn())
    }

    pub fn to_pb(&self) -> ::Result<tfpb::tensor::TensorProto> {
        let mut shape = tfpb::tensor_shape::TensorShapeProto::new();
        let dims = self
            .shape()
            .iter()
            .map(|d| {
                let mut dim = tfpb::tensor_shape::TensorShapeProto_Dim::new();
                dim.size = *d as _;
                dim
            })
            .collect();
        shape.set_dim(::protobuf::RepeatedField::from_vec(dims));
        let mut tensor = tfpb::tensor::TensorProto::new();
        tensor.set_tensor_shape(shape);
        match self {
            &Tensor::F32(ref it) => {
                tensor.set_dtype(DatumType::F32.to_pb()?);
                tensor.set_float_val(it.iter().cloned().collect());
            }
            &Tensor::F64(ref it) => {
                tensor.set_dtype(DatumType::F64.to_pb()?);
                tensor.set_double_val(it.iter().cloned().collect());
            }
            &Tensor::I32(ref it) => {
                tensor.set_dtype(DatumType::I32.to_pb()?);
                tensor.set_int_val(it.iter().cloned().collect());
            }
            _ => unimplemented!("missing type"),
        }
        Ok(tensor)
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            &Tensor::F64(ref it) => it.shape(),
            &Tensor::F32(ref it) => it.shape(),
            &Tensor::I32(ref it) => it.shape(),
            &Tensor::I8(ref it) => it.shape(),
            &Tensor::U8(ref it) => it.shape(),
            &Tensor::TDim(ref it) => it.shape(),
            _ => unimplemented!("missing type"),
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
            &Tensor::F64(_) => DatumType::F64,
            &Tensor::F32(_) => DatumType::F32,
            &Tensor::I32(_) => DatumType::I32,
            &Tensor::I8(_) => DatumType::I8,
            &Tensor::U8(_) => DatumType::U8,
            &Tensor::TDim(_) => DatumType::TDim,
            _ => unimplemented!("missing type"),
        }
    }

    pub fn axis_chunks(&self, axis: usize, size: usize) -> ::Result<Vec<Tensor>> {
        match self {
            &Tensor::F64(_) => self.axis_chunks_t::<f64>(axis, size),
            &Tensor::F32(_) => self.axis_chunks_t::<f32>(axis, size),
            &Tensor::I8(_) => self.axis_chunks_t::<i8>(axis, size),
            &Tensor::U8(_) => self.axis_chunks_t::<u8>(axis, size),
            &Tensor::I32(_) => self.axis_chunks_t::<i32>(axis, size),
            &Tensor::TDim(_) => self.axis_chunks_t::<TDim>(axis, size),
            _ => unimplemented!("missing type"),
        }
    }

    pub fn axis_chunks_t<T: Datum>(&self, axis: usize, size: usize) -> ::Result<Vec<Tensor>> {
        let array = T::tensor_to_view(self)?;
        Ok(array
            .axis_chunks_iter(Axis(axis), size)
            .map(|v| T::array_into_tensor(v.to_owned()))
            .collect())
    }

    pub fn partial_dump(&self, _single_line: bool) -> ::Result<String> {
        if self.shape().len() == 0 {
            Ok(match self {
                &Tensor::I32(ref a) => format!(
                    "Scalar {:?} {:?}",
                    self.datum_type(),
                    a.as_slice().unwrap()[0]
                ),
                &Tensor::F32(ref a) => format!(
                    "Scalar {:?} {:?}",
                    self.datum_type(),
                    a.as_slice().unwrap()[0]
                ),
                &Tensor::U8(ref a) => format!(
                    "Scalar {:?} {:?}",
                    self.datum_type(),
                    a.as_slice().unwrap()[0]
                ),
                &Tensor::TDim(ref a) => format!(
                    "Scalar {:?} {:?}",
                    self.datum_type(),
                    a.as_slice().unwrap()[0]
                ),
                _ => unimplemented!("missing type"),
            })
        } else if self.shape().iter().product::<usize>() > 8 {
            use itertools::Itertools;
            Ok(match self {
                &Tensor::I32(ref a) => format!(
                    "shape:{:?} {:?} {}...",
                    self.shape(),
                    self.datum_type(),
                    a.iter().take(4).join(", ")
                ),
                &Tensor::F32(ref a) => format!(
                    "shape:{:?} {:?} {}...",
                    self.shape(),
                    self.datum_type(),
                    a.iter().take(4).join(", ")
                ),
                &Tensor::U8(ref a) => format!(
                    "shape:{:?} {:?} {}...",
                    self.shape(),
                    self.datum_type(),
                    a.iter().take(4).join(", ")
                ),
                &Tensor::TDim(ref a) => format!(
                    "shape:{:?} {:?} {}...",
                    self.shape(),
                    self.datum_type(),
                    a.iter().take(4).map(|s| format!("{:?}", s)).join(", ")
                ),
                _ => unimplemented!("missing type"),
            })
        } else {
            Ok(match self {
                &Tensor::I32(ref a) => {
                    format!("{:?} {:?}", self.datum_type(), a).replace("\n", " ")
                }
                &Tensor::F32(ref a) => {
                    format!("{:?} {:?}", self.datum_type(), a).replace("\n", " ")
                }
                &Tensor::U8(ref a) => format!("{:?} {:?}", self.datum_type(), a).replace("\n", " "),
                &Tensor::TDim(ref a) => {
                    format!("{:?} {:?}", self.datum_type(), a).replace("\n", " ")
                }
                _ => unimplemented!("missing type"),
            })
        }
    }

    fn to_f32(&self) -> Tensor {
        match self {
            &Tensor::I32(ref data) => Tensor::F32(data.map(|&a| a as f32)),
            &Tensor::F32(_) => self.clone(),
            _ => unimplemented!("missing type"),
        }
    }

    pub fn close_enough(&self, other: &Self, approx: bool) -> bool {
        let ma = self.to_f32().take_f32s().unwrap();
        let mb = other.to_f32().take_f32s().unwrap();
        let avg = ma.iter().map(|&a| a.abs()).sum::<f32>() / ma.len() as f32;
        let dev = (ma.iter().map(|&a| (a - avg).powi(2)).sum::<f32>() / ma.len() as f32).sqrt();
        let margin = if approx {
            (dev / 10.0).max(avg.abs() / 10_000.0)
        } else {
            0.0
        };
        ma.shape() == mb.shape()
            && mb
                .iter()
                .zip(ma.iter())
                .all(|(&a, &b)| (b - a).abs() <= margin)
    }

    pub fn into_array<D: Datum>(self) -> ::Result<ArrayD<D>> {
        <D as Datum>::tensor_into_array(self)
    }

    pub fn to_array_view<'a, D: Datum>(&'a self) -> ::Result<ArrayViewD<'a, D>> {
        <D as Datum>::tensor_to_view(self)
    }

    pub fn cast_to_array<D: Datum>(&self) -> ::Result<MaybeOwnedArray<D>> {
        <D as Datum>::tensor_cast_to_array(self)
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let content = self.partial_dump(true).unwrap_or("Error".to_string());
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
            F32(m) => serialize_inner!(f32, m),
            F64(m) => serialize_inner!(f64, m),
            I32(m) => serialize_inner!(i32, m),
            I8(m) => serialize_inner!(i8, m),
            U8(m) => serialize_inner!(u8, m),
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
    ($t:ident, $v:ident, $as_one:ident, $as:ident, $take:ident, $make:ident, [$(($cast:ident, $as_cast:ident)),*]) => {
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

            pub fn $make(shape: &[usize], values: &[$t]) -> ::Result<Tensor> {
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

            fn tensor_into_array(m: Tensor) -> ::Result<ArrayD<Self>> {
                let _ = Self::tensor_to_view(&m)?;
                Ok(m.$take().unwrap())
            }

            fn tensor_to_view(m: &Tensor) -> ::Result<ArrayViewD<Self>> {
                m.$as()
                    .map(|m| m.view())
                    .ok_or_else(|| format!("Type mismatch unwrapping to {}: {:?}", Self::name(), &m).into())
            }

            fn array_into_tensor(m: ArrayD<Self>) -> Tensor {
                Tensor::$v(m)
            }

            fn tensor_cast_to_array(m: &Tensor) -> ::Result<MaybeOwnedArray<$t>> {
                match m.datum_type() {
                    DatumType::$v => Ok(MaybeOwnedArray::View($t::tensor_to_view(m)?)),
                    $(DatumType::$cast => {
                        let src = m.$as_cast().ok_or("Wrong type")?;
                        let vec:Vec<$t> = src.iter()
                            .map(|x| TryInto::<$t>::try_into(*x))
                            .collect::<::Result<Vec<_>>>()?;
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
    fn try_into(self) -> ::Result<D>;
}

impl<A: Into<D>, D: Datum> TryInto<D> for A {
    fn try_into(self) -> ::Result<D> {
        Ok(self.into())
    }
}

impl TryInto<i32> for TDim {
    fn try_into(self) -> ::Result<i32> {
        self.to_integer().map(|i| i as i32)
    }
}

tensor!(f64, F64, as_f64, as_f64s, take_f64s, f64s, []);
tensor!(f32, F32, as_f32, as_f32s, take_f32s, f32s, []);
tensor!(
    i32,
    I32,
    as_i32,
    as_i32s,
    take_i32s,
    i32s,
    [(TDim, as_dims)]
);
tensor!(u8, U8, as_u8, as_u8s, take_u8s, u8s, []);
tensor!(i8, I8, as_i8, as_i8s, take_i8s, i8s, []);
tensor!(
    TDim,
    TDim,
    as_dim,
    as_dims,
    take_dims,
    dims,
    [(I32, as_i32s)]
);

#[macro_export]
macro_rules! map_tensor {
    ($tensor:expr, | $array:ident | $return:expr) => {{
        use Tensor::*;
        match $tensor {
            F64($array) => F64($return),
            F32($array) => F32($return),
            I32($array) => I32($return),
            I8($array) => I8($return),
            U8($array) => U8($return),
            String($array) => String($return),
        }
    }};
}

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
