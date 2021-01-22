//! `Tensor` is the main data container for tract
use crate::dim::TDim;
use crate::f16::f16;
use crate::tensor::litteral::*;
use crate::tensor::Tensor;
use crate::TVec;
use std::hash::Hash;
use std::{fmt, ops};

mod arrays;
pub use arrays::ArrayDatum;

#[derive(Debug, Default, Clone, PartialEq, Eq, Educe)]
#[educe(Hash)]
pub struct Blob(pub Vec<u8>);

impl ops::Deref for Blob {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        &self.0
    }
}

impl fmt::Display for Blob {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "Blob of {} bytes: {}", self.len(), String::from_utf8_lossy(self))
    }
}

impl std::str::FromStr for Blob {
    type Err = ();
    fn from_str(s: &str) -> Result<Blob, ()> {
        Ok(Blob(s.as_bytes().to_vec()))
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub enum DatumType {
    Bool,
    U8,
    U16,
    U32,
    U64,
    I8,
    I16,
    I32,
    I64,
    F16,
    F32,
    F64,
    TDim,
    Blob,
    String,
}

impl DatumType {
    pub fn super_types(&self) -> TVec<DatumType> {
        use DatumType::*;
        if *self == String || *self == TDim || *self == Blob || *self == Bool {
            tvec!(*self)
        } else if self.is_float() {
            [F16, F32, F64].iter().filter(|s| s.size_of() >= self.size_of()).copied().collect()
        } else if self.is_signed() {
            [I8, I16, I32, I64, TDim]
                .iter()
                .filter(|s| s.size_of() >= self.size_of())
                .copied()
                .collect()
        } else {
            [U8, U16, U32, U64].iter().filter(|s| s.size_of() >= self.size_of()).copied().collect()
        }
    }

    pub fn super_type_for(
        i: impl IntoIterator<Item = impl std::borrow::Borrow<DatumType>>,
    ) -> Option<DatumType> {
        let mut iter = i.into_iter();
        let mut current = match iter.next() {
            None => return None,
            Some(it) => *it.borrow(),
        };
        while let Some(n) = iter.next() {
            match current.common_super_type(*n.borrow()) {
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
                    return Some(mine);
                }
            }
        }
        return None;
    }

    pub fn is_unsigned(&self) -> bool {
        match self {
            DatumType::U8 | DatumType::U16 | DatumType::U32 | DatumType::U64 => true,
            _ => false,
        }
    }

    pub fn is_signed(&self) -> bool {
        match self {
            DatumType::I8 | DatumType::I16 | DatumType::I32 | DatumType::I64 => true,
            _ => false,
        }
    }

    pub fn is_float(&self) -> bool {
        match self {
            DatumType::F16 | DatumType::F32 | DatumType::F64 => true,
            _ => false,
        }
    }

    pub fn is_copy(&self) -> bool {
        *self == DatumType::Bool || self.is_unsigned() || self.is_signed() || self.is_float()
    }

    pub fn integer(signed: bool, size: usize) -> Self {
        use DatumType::*;
        match (signed, size) {
            (false, 8) => U8,
            (false, 16) => U16,
            (false, 32) => U32,
            (false, 64) => U64,
            (true, 8) => U8,
            (true, 16) => U16,
            (true, 32) => U32,
            (true, 64) => U64,
            _ => panic!("No integer for signed:{} size:{}"),
        }
    }

    pub fn is_integer(&self) -> bool {
        self.is_signed() || self.is_unsigned()
    }

    #[inline]
    pub fn size_of(&self) -> usize {
        dispatch_datum!(std::mem::size_of(self)())
    }

    #[inline]
    pub fn alignment(&self) -> usize {
        match self {
            DatumType::TDim => std::mem::size_of::<usize>(),
            DatumType::String => std::mem::size_of::<usize>(),
            _ => self.size_of(),
        }
    }
}

impl std::str::FromStr for DatumType {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "I8" | "i8" => Ok(DatumType::I8),
            "I16" | "i16" => Ok(DatumType::I16),
            "I32" | "i32" => Ok(DatumType::I32),
            "I64" | "i64" => Ok(DatumType::I64),
            "U8" | "u8" => Ok(DatumType::U8),
            "U16" | "u16" => Ok(DatumType::U16),
            "U32" | "u32" => Ok(DatumType::U32),
            "U64" | "u64" => Ok(DatumType::U64),
            "F16" | "f16" => Ok(DatumType::F16),
            "F32" | "f32" => Ok(DatumType::F32),
            "F64" | "f64" => Ok(DatumType::F64),
            "Bool" | "bool" => Ok(DatumType::Bool),
            "Blob" | "blob" => Ok(DatumType::Blob),
            "String" | "string" => Ok(DatumType::String),
            "TDim" | "tdim" => Ok(DatumType::TDim),
            _ => anyhow::bail!("Unknown type {}", s),
        }
    }
}

pub trait Datum:
    Clone + Send + Sync + fmt::Debug + fmt::Display + Default + 'static + PartialEq
{
    fn name() -> &'static str;
    fn datum_type() -> DatumType;
}

macro_rules! datum {
    ($t:ty, $v:ident) => {
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
datum!(u32, U32);
datum!(u64, U64);
datum!(TDim, TDim);
datum!(String, String);
datum!(Blob, Blob);

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

    #[test]
    fn test_cast_i64_to_bool() {
        let t_i64: Tensor = tensor1(&[0i64]);
        t_i64.cast_to::<bool>().unwrap();
    }
}
