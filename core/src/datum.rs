//! `Tensor` is the main data container for tract
use crate::dim::TDim;
use crate::tensor::litteral::*;
use crate::tensor::Tensor;
use crate::TractResult;
use std::fmt;

use tract_linalg::f16::f16;

mod arrays;
pub use arrays::ArrayDatum;

#[cfg(feature = "serialize")]
use serde::ser::{Serialize, Serializer};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
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
                DatumType::F32,
            ],
            DatumType::U16 => {
                &[DatumType::U16, DatumType::I32, DatumType::I64, DatumType::TDim, DatumType::F32]
            }
            DatumType::I8 => &[
                DatumType::I8,
                DatumType::I16,
                DatumType::I32,
                DatumType::I64,
                DatumType::TDim,
                DatumType::F32,
            ],
            DatumType::I16 => {
                &[DatumType::I16, DatumType::I32, DatumType::I64, DatumType::TDim, DatumType::F32]
            }
            DatumType::I32 => &[DatumType::I32, DatumType::I64, DatumType::TDim, DatumType::F32],
            DatumType::I64 => &[DatumType::I64, DatumType::TDim, DatumType::F32],
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

    pub fn size_of(&self) -> usize {
        match self {
            DatumType::Bool => std::mem::size_of::<bool>(),
            DatumType::U8 => std::mem::size_of::<u8>(),
            DatumType::U16 => std::mem::size_of::<u16>(),
            DatumType::I8 => std::mem::size_of::<i8>(),
            DatumType::I16 => std::mem::size_of::<i16>(),
            DatumType::I32 => std::mem::size_of::<i32>(),
            DatumType::I64 => std::mem::size_of::<i64>(),
            DatumType::F16 => std::mem::size_of::<f16>(),
            DatumType::F32 => std::mem::size_of::<f32>(),
            DatumType::F64 => std::mem::size_of::<f64>(),
            DatumType::TDim => std::mem::size_of::<TDim>(),
            DatumType::String => std::mem::size_of::<String>(),
        }
    }

    pub fn alignment(&self) -> usize {
        match self {
            DatumType::TDim => std::mem::size_of::<usize>(),
            DatumType::String => std::mem::size_of::<usize>(),
            _ => self.size_of(),
        }
    }
}

pub trait Datum:
    Clone + Send + Sync + fmt::Debug + fmt::Display + Default + 'static + PartialEq + ArrayDatum
{
    fn name() -> &'static str;
    fn datum_type() -> DatumType;
}

pub(crate) trait TryInto<D> {
    fn try_into(&self) -> TractResult<D>;
}

macro_rules! datum {
    ($t:ident, $v:ident) => {
        impl From<$t> for Tensor {
            fn from(it: $t) -> Tensor {
                tensor0(it)
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

macro_rules! try_into {
    ($f:ty, $t:ty) => {
        impl TryInto<$t> for $f {
            fn try_into(&self) -> TractResult<$t> {
                Ok(*self as $t)
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

try_into!(u8, f32);
try_into!(u16, f32);

try_into!(u8, i32);
try_into!(u16, i32);

try_into!(i8, f64);
try_into!(i16, f64);
try_into!(i32, f64);
try_into!(i64, f64);

try_into!(f32, i8);
try_into!(f32, i16);
try_into!(f32, i32);
try_into!(f32, i64);

try_into!(f64, i8);
try_into!(f64, i16);
try_into!(f64, i32);
try_into!(f64, i64);

impl TryInto<TDim> for i32 {
    fn try_into(&self) -> TractResult<TDim> {
        Ok((*self).into())
    }
}

impl TryInto<TDim> for i64 {
    fn try_into(&self) -> TractResult<TDim> {
        Ok((*self).into())
    }
}

impl TryInto<i32> for TDim {
    fn try_into(&self) -> TractResult<i32> {
        self.to_integer().map(|i| i as i32)
    }
}

impl TryInto<i64> for TDim {
    fn try_into(&self) -> TractResult<i64> {
        self.to_integer().map(|i| i as i64)
    }
}

impl TryInto<bool> for f32 {
    fn try_into(&self) -> TractResult<bool> {
        Ok(*self != 0.0)
    }
}

impl TryInto<f32> for bool {
    fn try_into(&self) -> TractResult<f32> {
        if *self {
            Ok(1.0)
        } else {
            Ok(0.0)
        }
    }
}

impl TryInto<bool> for f64 {
    fn try_into(&self) -> TractResult<bool> {
        Ok(*self != 0.0)
    }
}

impl TryInto<f64> for bool {
    fn try_into(&self) -> TractResult<f64> {
        if *self {
            Ok(1.0)
        } else {
            Ok(0.0)
        }
    }
}

impl TryInto<f32> for f16 {
    fn try_into(&self) -> TractResult<f32> {
        Ok(self.0.to_f32())
    }
}

impl TryInto<f64> for f16 {
    fn try_into(&self) -> TractResult<f64> {
        Ok(self.0.to_f64())
    }
}

impl TryInto<f16> for f32 {
    fn try_into(&self) -> TractResult<f16> {
        Ok(f16(half::f16::from_f32(*self)))
    }
}

impl TryInto<f16> for f64 {
    fn try_into(&self) -> TractResult<f16> {
        Ok(f16(half::f16::from_f64(*self)))
    }
}

impl TryInto<String> for f32 {
    fn try_into(&self) -> TractResult<String> {
        Ok(self.to_string())
    }
}

impl TryInto<f32> for String {
    fn try_into(&self) -> TractResult<f32> {
        // this is onnx casts
        if self == "INF" || self == "+INF" {
            Ok(std::f32::INFINITY)
        } else if self == "-INF" {
            Ok(-std::f32::INFINITY)
        } else {
            Ok(self.parse::<f32>().map_err(|_| format!("Can not parse {} as f32", self))?)
        }
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
datum!(String, String);

pub trait FloatLike: Datum {
    fn mmm(m: usize, k: usize, n: usize) -> Box<dyn tract_linalg::mmm::MatMatMul<Self, Self, Self, Self>>;
    fn packed_vec_mat_mul(k: usize, n: usize) -> Box<dyn tract_linalg::vecmatmul::VecMatMul<Self>>;
    fn sigmoid() -> Box<dyn tract_linalg::sigmoid::Sigmoid<Self>>;
    fn tanh() -> Box<dyn tract_linalg::tanh::Tanh<Self>>;
}

impl FloatLike for f16 {
    fn mmm(_m: usize, _k: usize, _n: usize) -> Box<dyn tract_linalg::mmm::MatMatMul<Self, Self, Self, Self>> {
        unimplemented!("f16 ops");
    }
    fn packed_vec_mat_mul(
        _k: usize,
        _n: usize,
    ) -> Box<dyn tract_linalg::vecmatmul::VecMatMul<Self>> {
        unimplemented!("f16 ops");
    }
    fn sigmoid() -> Box<dyn tract_linalg::sigmoid::Sigmoid<Self>> {
        unimplemented!("f16 ops");
    }
    fn tanh() -> Box<dyn tract_linalg::tanh::Tanh<Self>> {
        unimplemented!("f16 ops");
    }
}

impl FloatLike for f32 {
    fn mmm(m: usize, k: usize, n: usize) -> Box<dyn tract_linalg::mmm::MatMatMul<Self, Self, Self, Self>> {
        (tract_linalg::ops().smmm)(m, k, n)
    }
    fn packed_vec_mat_mul(k: usize, n: usize) -> Box<dyn tract_linalg::vecmatmul::VecMatMul<Self>> {
        (tract_linalg::ops().svmm)(k, n)
    }
    fn sigmoid() -> Box<dyn tract_linalg::sigmoid::Sigmoid<Self>> {
        (tract_linalg::ops().ssigmoid)()
    }
    fn tanh() -> Box<dyn tract_linalg::tanh::Tanh<Self>> {
        (tract_linalg::ops().stanh)()
    }
}

impl FloatLike for f64 {
    fn mmm(_m: usize, _k: usize, _n: usize) -> Box<dyn tract_linalg::mmm::MatMatMul<Self, Self, Self, Self>> {
        unimplemented!("f64 ops");
    }
    fn packed_vec_mat_mul(
        _k: usize,
        _n: usize,
    ) -> Box<dyn tract_linalg::vecmatmul::VecMatMul<Self>> {
        unimplemented!("f64 ops");
    }
    fn sigmoid() -> Box<dyn tract_linalg::sigmoid::Sigmoid<Self>> {
        unimplemented!("f64 ops");
    }
    fn tanh() -> Box<dyn tract_linalg::tanh::Tanh<Self>> {
        unimplemented!("f64 ops");
    }
}

#[cfg(test)]
mod tests {
    use crate::internal::*;
    use ndarray::arr1;

    #[test]
    fn test_array_to_tensor_to_array() {
        let array = arr1(&[12i32, 42]);
        let tensor = Tensor::from(array.clone());
        let view = tensor.to_array_view::<i32>().unwrap();
        assert_eq!(array, view.into_dimensionality().unwrap());
    }

    #[test]
    fn test_cast_dim_to_dim() {
        let t_dim: Tensor = tensor1(&[12isize.to_dim(), 42isize.to_dim()]);
        let t_i32 = t_dim.cast_to::<i32>().unwrap();
        let t_dim_2 = t_i32.cast_to::<TDim>().unwrap().into_owned();
        assert_eq!(t_dim, t_dim_2);
    }

    #[test]
    fn test_cast_i32_to_dim() {
        let t_i32: Tensor = tensor1(&[0i32, 0]);
        t_i32.cast_to::<TDim>().unwrap();
    }
}
