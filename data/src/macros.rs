#[macro_export]
macro_rules! tvec {
    // count helper: transform any expression into 1
    (@one $x:expr) => (1usize);
    ($elem:expr; $n:expr) => ({
        $crate::TVec::from_elem($elem, $n)
    });
    ($($x:expr),*$(,)*) => ({
        let count = 0usize $(+ tvec!(@one $x))*;
        #[allow(unused_mut)]
        let mut vec = $crate::TVec::new();
        if count <= vec.inline_size() {
            $(vec.push($x);)*
            vec
        } else {
            $crate::TVec::from_vec(vec![$($x,)*])
        }
    });
}

#[macro_export]
macro_rules! dispatch_datum {
    ($($path:ident)::* ($dt:expr) ($($args:expr),*)) => { {
        use $crate::prelude::DatumType;
        #[allow(unexpected_cfgs)]
        match $dt {
            DatumType::Bool => $($path)::*::<bool>($($args),*),
            DatumType::U8   => $($path)::*::<u8>($($args),*),
            DatumType::U16  => $($path)::*::<u16>($($args),*),
            DatumType::U32  => $($path)::*::<u32>($($args),*),
            DatumType::U64  => $($path)::*::<u64>($($args),*),
            DatumType::I8   => $($path)::*::<i8>($($args),*),
            DatumType::I16  => $($path)::*::<i16>($($args),*),
            DatumType::I32  => $($path)::*::<i32>($($args),*),
            DatumType::I64  => $($path)::*::<i64>($($args),*),
            DatumType::F16  => $($path)::*::<f16>($($args),*),
            DatumType::F32  => $($path)::*::<f32>($($args),*),
            DatumType::F64  => $($path)::*::<f64>($($args),*),
            DatumType::Blob => $($path)::*::<$crate::prelude::Blob>($($args),*),
            DatumType::TDim => $($path)::*::<TDim>($($args),*),
            DatumType::String => $($path)::*::<String>($($args),*),
            DatumType::Opaque => $($path)::*::<$crate::prelude::Opaque>($($args),*),
            DatumType::QI8(_) => $($path)::*::<i8>($($args),*),
            DatumType::QU8(_) => $($path)::*::<u8>($($args),*),
            DatumType::QI32(_) => $($path)::*::<i32>($($args),*),
            #[cfg(feature = "complex")]
            DatumType::ComplexI16 => $($path)::*::<Complex<i16>>($($args),*),
            #[cfg(feature = "complex")]
            DatumType::ComplexI32 => $($path)::*::<Complex<i32>>($($args),*),
            #[cfg(feature = "complex")]
            DatumType::ComplexI64 => $($path)::*::<Complex<i64>>($($args),*),
            #[cfg(feature = "complex")]
            DatumType::ComplexF16 => $($path)::*::<Complex<f16>>($($args),*),
            #[cfg(feature = "complex")]
            DatumType::ComplexF32 => $($path)::*::<Complex<f32>>($($args),*),
            #[cfg(feature = "complex")]
            DatumType::ComplexF64 => $($path)::*::<Complex<f64>>($($args),*),
        }
    } }
}

#[macro_export]
macro_rules! dispatch_datum_by_size {
    ($($path:ident)::* ($dt:expr) ($($args:expr),*)) => { {
        use $crate::prelude::DatumType;
        #[allow(unexpected_cfgs)]
        match $dt {
            DatumType::Bool => $($path)::*::<i8>($($args),*),
            DatumType::U8   => $($path)::*::<i8>($($args),*),
            DatumType::U16  => $($path)::*::<i16>($($args),*),
            DatumType::U32  => $($path)::*::<i32>($($args),*),
            DatumType::U64  => $($path)::*::<i64>($($args),*),
            DatumType::I8   => $($path)::*::<i8>($($args),*),
            DatumType::I16  => $($path)::*::<i16>($($args),*),
            DatumType::I32  => $($path)::*::<i32>($($args),*),
            DatumType::I64  => $($path)::*::<i64>($($args),*),
            DatumType::F16  => $($path)::*::<i16>($($args),*),
            DatumType::F32  => $($path)::*::<i32>($($args),*),
            DatumType::F64  => $($path)::*::<i64>($($args),*),
            DatumType::Blob => $($path)::*::<Blob>($($args),*),
            DatumType::TDim => $($path)::*::<TDim>($($args),*),
            DatumType::String => $($path)::*::<String>($($args),*),
            DatumType::Opaque => $($path)::*::<$crate::prelude::Opaque>($($args),*),
            DatumType::QI8(_)   => $($path)::*::<i8>($($args),*),
            DatumType::QU8(_)   => $($path)::*::<u8>($($args),*),
            DatumType::QI32(_)   => $($path)::*::<i32>($($args),*),
            #[cfg(feature = "complex")]
            DatumType::ComplexI16 => $($path)::*::<Complex<i16>>($($args),*),
            #[cfg(feature = "complex")]
            DatumType::ComplexI32 => $($path)::*::<Complex<i32>>($($args),*),
            #[cfg(feature = "complex")]
            DatumType::ComplexI64 => $($path)::*::<Complex<i64>>($($args),*),
            #[cfg(feature = "complex")]
            DatumType::ComplexF16 => $($path)::*::<Complex<f16>>($($args),*),
            #[cfg(feature = "complex")]
            DatumType::ComplexF32 => $($path)::*::<Complex<f32>>($($args),*),
            #[cfg(feature = "complex")]
            DatumType::ComplexF64 => $($path)::*::<Complex<f64>>($($args),*),
        }
    } }
}

#[macro_export]
macro_rules! dispatch_copy {
    ($($path:ident)::* ($dt:expr) ($($args:expr),*)) => { {
        use $crate::prelude::DatumType;
        #[allow(unexpected_cfgs)]
        match $dt {
            DatumType::Bool => $($path)::*::<bool>($($args),*),
            DatumType::U8   => $($path)::*::<u8>($($args),*),
            DatumType::U16  => $($path)::*::<u16>($($args),*),
            DatumType::U32  => $($path)::*::<u32>($($args),*),
            DatumType::U64  => $($path)::*::<u64>($($args),*),
            DatumType::I8   => $($path)::*::<i8>($($args),*),
            DatumType::I16  => $($path)::*::<i16>($($args),*),
            DatumType::I32  => $($path)::*::<i32>($($args),*),
            DatumType::I64  => $($path)::*::<i64>($($args),*),
            DatumType::F16  => $($path)::*::<f16>($($args),*),
            DatumType::F32  => $($path)::*::<f32>($($args),*),
            DatumType::F64  => $($path)::*::<f64>($($args),*),
            DatumType::QI8(_)  => $($path)::*::<i8>($($args),*),
            DatumType::QU8(_)  => $($path)::*::<u8>($($args),*),
            DatumType::QI32(_)  => $($path)::*::<u8>($($args),*),
            #[cfg(feature = "complex")]
            DatumType::ComplexI16 => $($path)::*::<Complex<i16>>($($args),*),
            #[cfg(feature = "complex")]
            DatumType::ComplexI32 => $($path)::*::<Complex<i32>>($($args),*),
            #[cfg(feature = "complex")]
            DatumType::ComplexI64 => $($path)::*::<Complex<i64>>($($args),*),
            #[cfg(feature = "complex")]
            DatumType::ComplexF16 => $($path)::*::<Complex<f16>>($($args),*),
            #[cfg(feature = "complex")]
            DatumType::ComplexF32 => $($path)::*::<Complex<f32>>($($args),*),
            #[cfg(feature = "complex")]
            DatumType::ComplexF64 => $($path)::*::<Complex<f64>>($($args),*),
            _ => panic!("{:?} is not Copy", $dt)
        }
    } }
}

#[macro_export]
macro_rules! dispatch_copy_by_size {
    ($($path:ident)::* ($dt:expr) ($($args:expr),*)) => { {
        use $crate::prelude::DatumType;
        #[allow(unexpected_cfgs)]
        match $dt {
            DatumType::Bool => $($path)::*::<i8>($($args),*),
            DatumType::U8   => $($path)::*::<i8>($($args),*),
            DatumType::U16  => $($path)::*::<i16>($($args),*),
            DatumType::U32  => $($path)::*::<i32>($($args),*),
            DatumType::U64  => $($path)::*::<i64>($($args),*),
            DatumType::I8   => $($path)::*::<i8>($($args),*),
            DatumType::I16  => $($path)::*::<i16>($($args),*),
            DatumType::I32  => $($path)::*::<i32>($($args),*),
            DatumType::I64  => $($path)::*::<i64>($($args),*),
            DatumType::F16  => $($path)::*::<i16>($($args),*),
            DatumType::F32  => $($path)::*::<i32>($($args),*),
            DatumType::F64  => $($path)::*::<i64>($($args),*),
            DatumType::QI8(_)  => $($path)::*::<i8>($($args),*),
            DatumType::QU8(_)  => $($path)::*::<u8>($($args),*),
            DatumType::QI32(_)  => $($path)::*::<i32>($($args),*),
            #[cfg(feature = "complex")]
            DatumType::ComplexI32 => $($path)::*::<Complex<i32>>($($args),*),
            #[cfg(feature = "complex")]
            DatumType::ComplexI64 => $($path)::*::<Complex<i64>>($($args),*),
            #[cfg(feature = "complex")]
            DatumType::ComplexF16 => $($path)::*::<Complex<f16>>($($args),*),
            #[cfg(feature = "complex")]
            DatumType::ComplexF32 => $($path)::*::<Complex<f32>>($($args),*),
            #[cfg(feature = "complex")]
            DatumType::ComplexF64 => $($path)::*::<Complex<f64>>($($args),*),
            _ => panic!("{:?} is not Copy", $dt)
        }
    } }
}

#[macro_export]
macro_rules! dispatch_numbers {
    ($($path:ident)::* ($dt:expr) ($($args:expr),*)) => { {
        use $crate::prelude::DatumType;
        match $dt {
            DatumType::U8   => $($path)::*::<u8>($($args),*),
            DatumType::U16  => $($path)::*::<u16>($($args),*),
            DatumType::U32  => $($path)::*::<u32>($($args),*),
            DatumType::U64  => $($path)::*::<u64>($($args),*),
            DatumType::I8   => $($path)::*::<i8>($($args),*),
            DatumType::I16  => $($path)::*::<i16>($($args),*),
            DatumType::I32  => $($path)::*::<i32>($($args),*),
            DatumType::I64  => $($path)::*::<i64>($($args),*),
            DatumType::F16  => $($path)::*::<f16>($($args),*),
            DatumType::F32  => $($path)::*::<f32>($($args),*),
            DatumType::F64  => $($path)::*::<f64>($($args),*),
            DatumType::QI8(_)  => $($path)::*::<i8>($($args),*),
            DatumType::QU8(_)  => $($path)::*::<u8>($($args),*),
            DatumType::QI32(_)  => $($path)::*::<i32>($($args),*),
            _ => $crate::internal::bail!("{:?} is not a number", $dt)
        }
    } }
}

#[macro_export]
macro_rules! dispatch_zerolike {
    ($($path:ident)::* ($dt:expr) ($($args:expr),*)) => { {
        use $crate::prelude::DatumType;
        match $dt {
            DatumType::TDim => $($path)::*::<TDim>($($args),*),
            DatumType::U8   => $($path)::*::<u8>($($args),*),
            DatumType::U16  => $($path)::*::<u16>($($args),*),
            DatumType::U32  => $($path)::*::<u32>($($args),*),
            DatumType::U64  => $($path)::*::<u64>($($args),*),
            DatumType::I8   => $($path)::*::<i8>($($args),*),
            DatumType::I16  => $($path)::*::<i16>($($args),*),
            DatumType::I32  => $($path)::*::<i32>($($args),*),
            DatumType::I64  => $($path)::*::<i64>($($args),*),
            DatumType::F16  => $($path)::*::<f16>($($args),*),
            DatumType::F32  => $($path)::*::<f32>($($args),*),
            DatumType::F64  => $($path)::*::<f64>($($args),*),
            DatumType::QI8(_)  => $($path)::*::<i8>($($args),*),
            DatumType::QU8(_)  => $($path)::*::<u8>($($args),*),
            DatumType::QI32(_)  => $($path)::*::<i32>($($args),*),
            #[cfg(feature = "complex")]
            DatumType::ComplexI32 => $($path)::*::<Complex<i32>>($($args),*),
            #[cfg(feature = "complex")]
            DatumType::ComplexI64 => $($path)::*::<Complex<i64>>($($args),*),
            #[cfg(feature = "complex")]
            DatumType::ComplexF16 => $($path)::*::<Complex<f16>>($($args),*),
            #[cfg(feature = "complex")]
            DatumType::ComplexF32 => $($path)::*::<Complex<f32>>($($args),*),
            #[cfg(feature = "complex")]
            DatumType::ComplexF64 => $($path)::*::<Complex<f64>>($($args),*),
            _ => $crate::internal::bail!("{:?} doesn't implement num_traits::Zero", $dt)
        }
    } }
}

#[macro_export]
macro_rules! dispatch_floatlike {
    ($($path:ident)::* ($dt:expr) ($($args:expr),*)) => { {
        use $crate::prelude::DatumType;
        match $dt {
            DatumType::F16  => $($path)::*::<f16>($($args),*),
            DatumType::F32  => $($path)::*::<f32>($($args),*),
            DatumType::F64  => $($path)::*::<f64>($($args),*),
            _ => $crate::internal::bail!("{:?} is not float-like", $dt)
        }
    } }
}

#[macro_export]
macro_rules! dispatch_signed {
    ($($path:ident)::* ($dt:expr) ($($args:expr),*)) => { {
        use $crate::prelude::DatumType;
        match $dt {
            DatumType::F16  => $($path)::*::<f16>($($args),*),
            DatumType::F32  => $($path)::*::<f32>($($args),*),
            DatumType::F64  => $($path)::*::<f64>($($args),*),
            DatumType::I8   => $($path)::*::<i8>($($args),*),
            DatumType::I16  => $($path)::*::<i16>($($args),*),
            DatumType::I32  => $($path)::*::<i32>($($args),*),
            DatumType::I64  => $($path)::*::<i64>($($args),*),
            DatumType::TDim => $($path)::*::<TDim>($($args),*),
            _ => $crate::internal::bail!("{:?} is not signed", $dt)
        }
    } }
}

#[macro_export]
macro_rules! dispatch_hash {
    ($($path:ident)::* ($dt:expr) ($($args:expr),*)) => { {
        use $crate::prelude::DatumType;
        #[allow(unexpected_cfgs)]
        match $dt {
            DatumType::Bool => $($path)::*::<bool>($($args),*),
            DatumType::U8   => $($path)::*::<u8>($($args),*),
            DatumType::U16  => $($path)::*::<u16>($($args),*),
            DatumType::U32  => $($path)::*::<u32>($($args),*),
            DatumType::U64  => $($path)::*::<u64>($($args),*),
            DatumType::I8   => $($path)::*::<i8>($($args),*),
            DatumType::I16  => $($path)::*::<i16>($($args),*),
            DatumType::I32  => $($path)::*::<i32>($($args),*),
            DatumType::I64  => $($path)::*::<i64>($($args),*),
            DatumType::Blob => $($path)::*::<Blob>($($args),*),
            DatumType::TDim => $($path)::*::<TDim>($($args),*),
            DatumType::String => $($path)::*::<String>($($args),*),
            DatumType::Opaque => $($path)::*::<$crate::prelude::Opaque>($($args),*),
            #[cfg(feature="complex")]
            DatumType::ComplexI16 => $($path)::*::<Complex<i16>>($($args),*),
            #[cfg(feature="complex")]
            DatumType::ComplexI32 => $($path)::*::<Complex<i32>>($($args),*),
            #[cfg(feature="complex")]
            DatumType::ComplexI64 => $($path)::*::<Complex<i64>>($($args),*),
            _ => $crate::internal::bail!("{:?} is not Hash", $dt)
        }
    } }
}
