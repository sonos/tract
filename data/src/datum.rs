//! `Tensor` is the main data container for tract
use crate::dim::TDim;
use crate::tensor::litteral::*;
use crate::tensor::Tensor;
use crate::TVec;
use half::f16;
use num_complex::Complex;
use scan_fmt::scan_fmt;
use std::hash::Hash;
use std::{fmt, ops};

use num_traits::AsPrimitive;

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

#[derive(Copy, Clone, PartialEq, PartialOrd)]
pub enum QParams {
    MinMax { min: f32, max: f32 },
    ZpScale { zero_point: i32, scale: f32 },
}

impl Eq for QParams {}

impl Ord for QParams {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl Hash for QParams {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            QParams::MinMax { min, max } => {
                0.hash(state);
                min.to_bits().hash(state);
                max.to_bits().hash(state);
            }
            QParams::ZpScale { zero_point, scale } => {
                1.hash(state);
                zero_point.hash(state);
                scale.to_bits().hash(state);
            }
        }
    }
}

impl QParams {
    pub fn zp_scale(&self) -> (i32, f32) {
        match self {
            QParams::MinMax { min, max } => {
                let scale = (max - min) / 255.;
                ((-(min + max) / 2. / scale) as i32, scale)
            }
            QParams::ZpScale { zero_point, scale } => (*zero_point, *scale),
        }
    }

    pub fn q(&self, f: f32) -> i32 {
        let (zp, scale) = self.zp_scale();
        (f / scale) as i32 + zp
    }

    pub fn dq(&self, i: i32) -> f32 {
        let (zp, scale) = self.zp_scale();
        (i - zp) as f32 * scale
    }
}

impl std::fmt::Debug for QParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (zp, scale) = self.zp_scale();
        write!(f, "Z:{} S:{}", zp, scale)
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
    QI8(QParams),
    QU8(QParams),
    QI32(QParams),
    ComplexI16,
    ComplexI32,
    ComplexI64,
    ComplexF16,
    ComplexF32,
    ComplexF64,
}

impl DatumType {
    pub fn super_types(&self) -> TVec<DatumType> {
        use DatumType::*;
        if *self == String || *self == TDim || *self == Blob || *self == Bool || self.is_quantized()
        {
            tvec!(*self)
        } else if self.is_complex_float() {
            [ComplexF16, ComplexF32, ComplexF64]
                .iter()
                .filter(|s| s.size_of() >= self.size_of())
                .copied()
                .collect()
        } else if self.is_complex_signed() {
            [ComplexI16, ComplexI32, ComplexI64]
                .iter()
                .filter(|s| s.size_of() >= self.size_of())
                .copied()
                .collect()
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
        matches!(
            self.unquantized(),
            DatumType::U8 | DatumType::U16 | DatumType::U32 | DatumType::U64
        )
    }

    pub fn is_signed(&self) -> bool {
        matches!(
            self.unquantized(),
            DatumType::I8 | DatumType::I16 | DatumType::I32 | DatumType::I64
        )
    }

    pub fn is_float(&self) -> bool {
        matches!(self, DatumType::F16 | DatumType::F32 | DatumType::F64)
    }

    pub fn is_complex(&self) -> bool {
        self.is_complex_float() || self.is_complex_signed()
    }

    pub fn is_complex_float(&self) -> bool {
        matches!(self, DatumType::ComplexF16 | DatumType::ComplexF32 | DatumType::ComplexF64)
    }

    pub fn is_complex_signed(&self) -> bool {
        matches!(self, DatumType::ComplexI16 | DatumType::ComplexI32 | DatumType::ComplexI64)
    }

    pub fn is_copy(&self) -> bool {
        *self == DatumType::Bool
            || self.is_unsigned()
            || self.is_signed()
            || self.is_float()
            || self.is_complex()
    }

    pub fn is_quantized(&self) -> bool {
        self.qparams().is_some()
    }

    pub fn qparams(&self) -> Option<QParams> {
        match self {
            DatumType::QI8(qparams) | DatumType::QU8(qparams) | DatumType::QI32(qparams) => {
                Some(*qparams)
            }
            _ => None,
        }
    }

    pub fn with_qparams(&self, qparams: QParams) -> DatumType {
        match self {
            DatumType::QI8(_) => DatumType::QI8(qparams),
            DatumType::QU8(_) => DatumType::QI8(qparams),
            DatumType::QI32(_) => DatumType::QI32(qparams),
            _ => *self,
        }
    }
    #[inline(always)]
    pub fn zp_scale(&self) -> (i32, f32) {
        self.qparams().map(|q| q.zp_scale()).unwrap_or((0, 1.))
    }

    pub fn unquantized(&self) -> DatumType {
        match self {
            DatumType::QI8(_) => DatumType::I8,
            DatumType::QU8(_) => DatumType::U8,
            DatumType::QI32(_) => DatumType::I32,
            _ => *self,
        }
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
            _ => panic!("No integer for signed:{} size:{}", signed, size),
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

    pub fn min_value(&self) -> Tensor {
        match self {
            DatumType::QU8(_)
            | DatumType::U8
            | DatumType::U16
            | DatumType::U32
            | DatumType::U64 => Tensor::zero_dt(*self, &[1]).unwrap(),
            DatumType::I8 | DatumType::QI8(_) => tensor0(i8::MIN),
            DatumType::QI32(_) => tensor0(i32::MIN),
            DatumType::I16 => tensor0(i16::MIN),
            DatumType::I32 => tensor0(i32::MIN),
            DatumType::I64 => tensor0(i64::MIN),
            DatumType::F16 => tensor0(f16::MIN),
            DatumType::F32 => tensor0(f32::MIN),
            DatumType::F64 => tensor0(f64::MIN),
            _ => panic!("No min value for datum type {:?}", self),
        }
    }
    pub fn max_value(&self) -> Tensor {
        match self {
            DatumType::U8 | DatumType::QU8(_) => tensor0(u8::MAX),
            DatumType::U16 => tensor0(u16::MAX),
            DatumType::U32 => tensor0(u32::MAX),
            DatumType::U64 => tensor0(u64::MAX),
            DatumType::I8 | DatumType::QI8(_) => tensor0(i8::MAX),
            DatumType::I16 => tensor0(i16::MAX),
            DatumType::I32 => tensor0(i32::MAX),
            DatumType::I64 => tensor0(i64::MAX),
            DatumType::QI32(_) => tensor0(i32::MAX),
            DatumType::F16 => tensor0(f16::MAX),
            DatumType::F32 => tensor0(f32::MAX),
            DatumType::F64 => tensor0(f64::MAX),
            _ => panic!("No max value for datum type {:?}", self),
        }
    }
}

impl std::str::FromStr for DatumType {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if let Ok((z, s)) = scan_fmt!(s, "QU8(Z:{d} S:{f})", i32, f32) {
            Ok(DatumType::QU8(QParams::ZpScale { zero_point: z, scale: s }))
        } else if let Ok((z, s)) = scan_fmt!(s, "QI8(Z:{d} S:{f})", i32, f32) {
            Ok(DatumType::QI8(QParams::ZpScale { zero_point: z, scale: s }))
        } else if let Ok((z, s)) = scan_fmt!(s, "QI32(Z:{d} S:{f})", i32, f32) {
            Ok(DatumType::QI32(QParams::ZpScale { zero_point: z, scale: s }))
        } else {
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
                "ComplexI16" | "complexi16" => Ok(DatumType::ComplexI16),
                "ComplexI32" | "complexi32" => Ok(DatumType::ComplexI32),
                "ComplexI64" | "complexi64" => Ok(DatumType::ComplexI64),
                "ComplexF16" | "complexf16" => Ok(DatumType::ComplexF16),
                "ComplexF32" | "complexf32" => Ok(DatumType::ComplexF32),
                "ComplexF64" | "complexf64" => Ok(DatumType::ComplexF64),
                _ => anyhow::bail!("Unknown type {}", s),
            }
        }
    }
}

const TOINT: f32 = 1.0f32 / std::f32::EPSILON;

pub fn round_ties_to_even(x: f32) -> f32 {
    let u = x.to_bits();
    let e = u >> 23 & 0xff;
    if e >= 0x7f + 23 {
        return x;
    }
    let s = u >> 31;
    let y = if s == 1 { x - TOINT + TOINT } else { x + TOINT - TOINT };
    if y == 0.0 {
        if s == 1 {
            -0f32
        } else {
            0f32
        }
    } else {
        y
    }
}

#[inline]
pub fn scale_by<T: Datum + AsPrimitive<f32>>(b: T, a: f32) -> T
where
    f32: AsPrimitive<T>,
{
    let b = b.as_();
    (round_ties_to_even(b.abs() * a) * b.signum()).as_()
}

pub trait ClampCast: PartialOrd + Copy + 'static {
    #[inline(always)]
    fn clamp_cast<O>(self) -> O
    where
        Self: AsPrimitive<O> + Datum,
        O: AsPrimitive<Self> + num_traits::Bounded + Datum,
    {
        // this fails if we're upcasting, in which case clamping is useless
        if O::min_value().as_() < O::max_value().as_() {
            num_traits::clamp(self, O::min_value().as_(), O::max_value().as_()).as_()
        } else {
            self.as_()
        }
    }
}
impl<T: PartialOrd + Copy + 'static> ClampCast for T {}

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
datum!(Complex<i16>, ComplexI16);
datum!(Complex<i32>, ComplexI32);
datum!(Complex<i64>, ComplexI64);
datum!(Complex<f16>, ComplexF16);
datum!(Complex<f32>, ComplexF32);
datum!(Complex<f64>, ComplexF64);

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

    #[test]
    fn test_parse_qu8() {
        assert_eq!(
            "QU8(Z:128 S:0.01)".parse::<DatumType>().unwrap(),
            DatumType::QU8(QParams::ZpScale { zero_point: 128, scale: 0.01 })
        );
    }
}
