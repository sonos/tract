//! `Tensor` is the equivalent of Tensorflow Tensor.
use ndarray::prelude::*;
use std::fmt::Debug;
use tfpb::types::DataType;

#[cfg(feature = "serialize")]
use serde::ser::{Serialize, Serializer};

pub trait Datum:
    Copy
    + Clone
    + Send
    + Sync
    + Debug
    + 'static
    + ::num_traits::Zero
    + ::num_traits::One
    + ::ndarray::LinalgScalar
    + ::std::ops::AddAssign
    + ::std::ops::MulAssign
    + ::std::ops::DivAssign
    + ::std::ops::SubAssign
    + ::std::ops::RemAssign
{
    fn name() -> &'static str;
    fn datatype() -> DataType;
    fn tensor_into_array(m: Tensor) -> ::Result<ArrayD<Self>>;
    fn tensor_to_view(m: &Tensor) -> ::Result<ArrayViewD<Self>>;
    fn array_into_tensor(m: ArrayD<Self>) -> Tensor;
}

#[derive(Clone, Debug, PartialEq)]
pub enum Tensor {
    F32(ArrayD<f32>),
    F64(ArrayD<f64>),
    I32(ArrayD<i32>),
    I8(ArrayD<i8>),
    U8(ArrayD<u8>),
    String(ArrayD<i8>),
}

impl Tensor {
    pub fn from_pb(t: &::tfpb::tensor::TensorProto) -> ::Result<Tensor> {
        use tfpb::types::DataType::*;
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
                _ => unimplemented!(),
            }
        } else {
            match dtype {
                DT_INT32 => Self::from_content::<i32, i32>(dims, t.get_int_val())?.into(),
                DT_FLOAT => Self::from_content::<f32, f32>(dims, t.get_float_val())?.into(),
                _ => unimplemented!(),
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

    pub fn to_pb(&self) -> ::Result<::tfpb::tensor::TensorProto> {
        let mut shape = ::tfpb::tensor_shape::TensorShapeProto::new();
        let dims = self.shape()
            .iter()
            .map(|d| {
                let mut dim = ::tfpb::tensor_shape::TensorShapeProto_Dim::new();
                dim.size = *d as _;
                dim
            })
            .collect();
        shape.set_dim(::protobuf::RepeatedField::from_vec(dims));
        let mut tensor = ::tfpb::tensor::TensorProto::new();
        tensor.set_tensor_shape(shape);
        match self {
            &Tensor::F32(ref it) => {
                tensor.set_dtype(DataType::DT_FLOAT);
                tensor.set_float_val(it.iter().cloned().collect());
            },
            &Tensor::F64(ref it) => {
                tensor.set_dtype(DataType::DT_DOUBLE);
                tensor.set_double_val(it.iter().cloned().collect());
            },
            &Tensor::I32(ref it) => {
                tensor.set_dtype(DataType::DT_INT32);
                tensor.set_int_val(it.iter().cloned().collect());
            },
            _ => unimplemented!(),
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
            _ => unimplemented!(),
        }
    }

    pub fn datatype(&self) -> ::tfpb::types::DataType {
        use tfpb::types::DataType;
        match self {
            &Tensor::F64(_) => DataType::DT_DOUBLE,
            &Tensor::F32(_) => DataType::DT_FLOAT,
            &Tensor::I32(_) => DataType::DT_INT32,
            &Tensor::I8(_) => DataType::DT_INT8,
            &Tensor::U8(_) => DataType::DT_UINT8,
            _ => unimplemented!(),
        }
    }

    pub fn partial_dump(&self, _single_line: bool) -> ::Result<String> {
        if self.shape().iter().product::<usize>() > 25 {
            Ok(format!("{:?} {:?}", self.datatype(), self.shape()))
        } else {
            Ok(match self {
                &Tensor::I32(ref a) => format!("{:?} {:?}", self.datatype(), a).replace("\n", " "),
                &Tensor::F32(ref a) => format!("{:?} {:?}", self.datatype(), a).replace("\n", " "),
                &Tensor::U8(ref a) => format!("{:?} {:?}", self.datatype(), a).replace("\n", " "),
                _ => unimplemented!(),
            })
        }
    }

    fn to_f32(&self) -> Tensor {
        match self {
            &Tensor::I32(ref data) => Tensor::F32(data.map(|&a| a as f32)),
            &Tensor::F32(_) => self.clone(),
            _ => unimplemented!(),
        }
    }

    pub fn close_enough(&self, other: &Self) -> bool {
        let ma = self.to_f32().take_f32s().unwrap();
        let mb = other.to_f32().take_f32s().unwrap();
        let avg = ma.iter().map(|&a| a.abs()).sum::<f32>() / ma.len() as f32;
        let dev = (ma.iter().map(|&a| (a - avg).powi(2)).sum::<f32>() / ma.len() as f32).sqrt();
        ma.shape() == mb.shape()
            && mb.iter()
                .zip(ma.iter())
                .all(|(&a, &b)| (b - a).abs() <= dev / 10.0)
    }
}

pub trait CastFrom<T>
where
    Self: Sized,
{
    fn cast_from(value: T) -> Option<Self>;
}

pub trait CastInto<U> {
    fn cast_into(self) -> Option<U>;
}

impl<T, U> CastInto<U> for T
where
    U: CastFrom<T>,
{
    fn cast_into(self) -> Option<U> {
        U::cast_from(self)
    }
}

#[cfg(feature = "serialize")]
impl Serialize for Tensor
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        macro_rules! serialize_inner {
            ($type:ident, $m:ident) => ({
                let data = (
                    stringify!($type),
                    self.shape(),
                    $m.iter().cloned().collect::<Vec<_>>()
                );
                data.serialize(serializer)
            });
        };

        use Tensor::*;
        match self {
            F32(m) => serialize_inner!(f32, m),
            F64(m) => serialize_inner!(f64, m),
            I32(m) => serialize_inner!(i32, m),
            I8(m) => serialize_inner!(i8, m),
            U8(m) => serialize_inner!(u8, m),
            String(m) => serialize_inner!(str, m),
        }
    }
}

macro_rules! tensor {
    ($t:ident, $pbt:ident, $v:ident, $as:ident, $take:ident, $make:ident) => {
        impl<D: ::ndarray::Dimension> From<Array<$t, D>> for Tensor {
            fn from(it: Array<$t, D>) -> Tensor {
                Tensor::$v(it.into_dyn())
            }
        }

        impl Tensor {
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

        impl CastFrom<Tensor> for ArrayD<$t> {
            fn cast_from(mat: Tensor) -> Option<ArrayD<$t>> {
                if let Tensor::$v(it) = mat {
                    Some(it)
                } else {
                    None
                }
            }
        }

        impl Datum for $t {
            fn name() -> &'static str {
                stringify!($t)
            }

            fn datatype() -> DataType {
                DataType::$pbt
            }

            fn tensor_into_array(m: Tensor) -> ::Result<ArrayD<Self>> {
                m.$take().ok_or("unmatched data type".into())
            }

            fn tensor_to_view(m: &Tensor) -> ::Result<ArrayViewD<Self>> {
                m.$as()
                    .map(|m| m.view())
                    .ok_or("unmatched data type".into())
            }

            fn array_into_tensor(m: ArrayD<Self>) -> Tensor {
                Tensor::from(m)
            }
        }
    };
}

tensor!(f64, DT_DOUBLE, F64, as_f64s, take_f64s, f64s);
tensor!(f32, DT_FLOAT, F32, as_f32s, take_f32s, f32s);
tensor!(i32, DT_INT32, I32, as_i32s, take_i32s, i32s);
tensor!(u8, DT_UINT8, U8, as_u8s, take_u8s, u8s);
tensor!(i8, DT_INT8, I8, as_i8s, take_i8s, i8s);

#[macro_export]
macro_rules! map_tensor {
    ($tensor:expr, |$array:ident| $return:expr) => ({
        use Tensor::*;
        match $tensor {
            F64($array) => F64($return),
            F32($array) => F32($return),
            I32($array) => I32($return),
            I8($array) => I8($return),
            U8($array) => U8($return),
            String($array) => String($return),
        }
    })
}
