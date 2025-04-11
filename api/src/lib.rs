use anyhow::{ensure, Result};
use boow::Bow;
use std::fmt::{Debug, Display};
use std::path::Path;

#[macro_use]
pub mod macros;

/// an implementation of tract's NNEF framework object
///
/// Entry point for NNEF model manipulation: loading from file, dumping to file.
pub trait NnefInterface: Sized {
    type Model: ModelInterface;
    /// Load a NNEF model from the path into a tract-core model.
    ///
    /// * `path` can point to a directory, a `tar` file or a `tar.gz` file.
    fn model_for_path(&self, path: impl AsRef<Path>) -> Result<Self::Model>;

    /// Transform model according to transform spec
    fn transform_model(&self, model: &mut Self::Model, transform_spec: &str) -> Result<()>;

    /// Allow the framework to use tract_core extensions instead of a stricter NNEF definition.
    fn enable_tract_core(&mut self) -> Result<()>;

    /// Allow the framework to use tract_extra extensions.
    fn enable_tract_extra(&mut self) -> Result<()>;

    /// Allow the framework to use tract_transformers extensions to support common transformer operators.
    fn enable_tract_transformers(&mut self) -> Result<()>;

    /// Allow the framework to use tract_onnx extensions to support operators in ONNX that are
    /// absent from NNEF.
    fn enable_onnx(&mut self) -> Result<()>;

    /// Allow the framework to use tract_pulse extensions to support stateful streaming operation.
    fn enable_pulse(&mut self) -> Result<()>;

    /// Allow the framework to use a tract-proprietary extension that can support special characters
    /// in node names. If disable, tract will replace everything by underscore '_' to keep
    /// compatibility with NNEF. If enabled, the extended syntax will be used, allowing to maintain
    /// the node names in serialized form.
    fn enable_extended_identifier_syntax(&mut self) -> Result<()>;

    /// Convenience function, similar with enable_tract_core but allowing method chaining.
    fn with_tract_core(mut self) -> Result<Self> {
        self.enable_tract_core()?;
        Ok(self)
    }

    /// Convenience function, similar with enable_tract_core but allowing method chaining.
    fn with_tract_extra(mut self) -> Result<Self> {
        self.enable_tract_extra()?;
        Ok(self)
    }

    /// Convenience function, similar with enable_tract_transformers but allowing method chaining.
    fn with_tract_transformers(mut self) -> Result<Self> {
        self.enable_tract_transformers()?;
        Ok(self)
    }

    /// Convenience function, similar with enable_onnx but allowing method chaining.
    fn with_onnx(mut self) -> Result<Self> {
        self.enable_onnx()?;
        Ok(self)
    }

    /// Convenience function, similar with enable_pulse but allowing method chaining.
    fn with_pulse(mut self) -> Result<Self> {
        self.enable_pulse()?;
        Ok(self)
    }

    /// Convenience function, similar with enable_extended_identifier_syntax but allowing method chaining.
    fn with_extended_identifier_syntax(mut self) -> Result<Self> {
        self.enable_extended_identifier_syntax()?;
        Ok(self)
    }

    /// Dump a TypedModel as a NNEF directory.
    ///
    /// `path` is the directory name to dump to
    fn write_model_to_dir(&self, path: impl AsRef<Path>, model: &Self::Model) -> Result<()>;

    /// Dump a TypedModel as a NNEF tar file.
    ///
    /// This function creates a plain, non-compressed, archive.
    ///
    /// `path` is the archive name
    fn write_model_to_tar(&self, path: impl AsRef<Path>, model: &Self::Model) -> Result<()>;
    fn write_model_to_tar_gz(&self, path: impl AsRef<Path>, model: &Self::Model) -> Result<()>;
}

pub trait OnnxInterface {
    type InferenceModel: InferenceModelInterface;
    fn model_for_path(&self, path: impl AsRef<Path>) -> Result<Self::InferenceModel>;
}

pub trait InferenceModelInterface: Sized {
    type Model: ModelInterface;
    type InferenceFact: InferenceFactInterface;
    fn set_output_names(
        &mut self,
        outputs: impl IntoIterator<Item = impl AsRef<str>>,
    ) -> Result<()>;
    fn input_count(&self) -> Result<usize>;
    fn output_count(&self) -> Result<usize>;
    fn input_name(&self, id: usize) -> Result<String>;
    fn output_name(&self, id: usize) -> Result<String>;

    fn input_fact(&self, id: usize) -> Result<Self::InferenceFact>;

    fn set_input_fact(
        &mut self,
        id: usize,
        fact: impl AsFact<Self, Self::InferenceFact>,
    ) -> Result<()>;

    fn output_fact(&self, id: usize) -> Result<Self::InferenceFact>;

    fn set_output_fact(
        &mut self,
        id: usize,
        fact: impl AsFact<Self, Self::InferenceFact>,
    ) -> Result<()>;

    fn analyse(&mut self) -> Result<()>;

    fn into_typed(self) -> Result<Self::Model>;

    fn into_optimized(self) -> Result<Self::Model>;
}

pub trait ModelInterface: Sized {
    type Fact: FactInterface;
    type Runnable: RunnableInterface;
    type Value: ValueInterface;
    fn input_count(&self) -> Result<usize>;

    fn output_count(&self) -> Result<usize>;

    fn input_name(&self, id: usize) -> Result<String>;

    fn output_name(&self, id: usize) -> Result<String>;

    fn set_output_names(
        &mut self,
        outputs: impl IntoIterator<Item = impl AsRef<str>>,
    ) -> Result<()>;

    fn input_fact(&self, id: usize) -> Result<Self::Fact>;

    fn output_fact(&self, id: usize) -> Result<Self::Fact>;

    fn declutter(&mut self) -> Result<()>;

    fn optimize(&mut self) -> Result<()>;

    fn into_decluttered(self) -> Result<Self>;

    fn into_optimized(self) -> Result<Self>;

    fn into_runnable(self) -> Result<Self::Runnable>;

    fn concretize_symbols(
        &mut self,
        values: impl IntoIterator<Item = (impl AsRef<str>, i64)>,
    ) -> Result<()>;

    fn transform(&mut self, transform: &str) -> Result<()>;

    fn pulse(&mut self, name: impl AsRef<str>, value: impl AsRef<str>) -> Result<()>;

    fn cost_json(&self) -> Result<String>;

    fn profile_json<I, V, E>(&self, inputs: Option<I>) -> Result<String>
    where
        I: IntoIterator<Item = V>,
        V: TryInto<Self::Value, Error = E>,
        E: Into<anyhow::Error> + Debug;

    fn property_keys(&self) -> Result<Vec<String>>;

    fn property(&self, name: impl AsRef<str>) -> Result<Self::Value>;
}

pub trait RunnableInterface {
    type Value: ValueInterface;
    type State: StateInterface<Value = Self::Value>;
    fn run<I, V, E>(&self, inputs: I) -> Result<Vec<Self::Value>>
    where
        I: IntoIterator<Item = V>,
        V: TryInto<Self::Value, Error = E>,
        E: Into<anyhow::Error>,
    {
        self.spawn_state()?.run(inputs)
    }

    fn input_count(&self) -> Result<usize>;
    fn output_count(&self) -> Result<usize>;

    fn spawn_state(&self) -> Result<Self::State>;
}

pub trait StateInterface {
    type Value: ValueInterface;

    fn input_count(&self) -> Result<usize>;
    fn output_count(&self) -> Result<usize>;

    fn run<I, V, E>(&mut self, inputs: I) -> Result<Vec<Self::Value>>
    where
        I: IntoIterator<Item = V>,
        V: TryInto<Self::Value, Error = E>,
        E: Into<anyhow::Error>;
}

pub trait ValueInterface: Sized + Clone {
    fn from_bytes(dt: DatumType, shape: &[usize], data: &[u8]) -> Result<Self>;
    fn as_bytes(&self) -> Result<(DatumType, &[usize], &[u8])>;

    fn from_slice<T: Datum>(shape: &[usize], data: &[T]) -> Result<Self> {
        let data = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
        };
        Self::from_bytes(T::datum_type(), shape, data)
    }

    fn as_slice<T: Datum>(&self) -> Result<(&[usize], &[T])> {
        let (dt, shape, data) = self.as_bytes()?;
        ensure!(T::datum_type() == dt);
        let data = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const T,
                data.len() / std::mem::size_of::<T>(),
            )
        };
        Ok((shape, data))
    }

    fn view<T: Datum>(&self) -> Result<ndarray::ArrayViewD<T>> {
        let (shape, data) = self.as_slice()?;
        Ok(unsafe { ndarray::ArrayViewD::from_shape_ptr(shape, data.as_ptr()) })
    }
}

pub trait FactInterface: Debug + Display + Clone {}
pub trait InferenceFactInterface: Debug + Display + Default + Clone {
    fn empty() -> Result<Self>;
}

pub trait AsFact<M, F> {
    fn as_fact(&self, model: &mut M) -> Result<Bow<F>>;
}

#[repr(C)]
#[allow(non_camel_case_types)]
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum DatumType {
    TRACT_DATUM_TYPE_BOOL = 0x01,
    TRACT_DATUM_TYPE_U8 = 0x11,
    TRACT_DATUM_TYPE_U16 = 0x12,
    TRACT_DATUM_TYPE_U32 = 0x14,
    TRACT_DATUM_TYPE_U64 = 0x18,
    TRACT_DATUM_TYPE_I8 = 0x21,
    TRACT_DATUM_TYPE_I16 = 0x22,
    TRACT_DATUM_TYPE_I32 = 0x24,
    TRACT_DATUM_TYPE_I64 = 0x28,
    TRACT_DATUM_TYPE_F16 = 0x32,
    TRACT_DATUM_TYPE_F32 = 0x34,
    TRACT_DATUM_TYPE_F64 = 0x38,
    #[cfg(feature = "complex")]
    TRACT_DATUM_TYPE_COMPLEX_I16 = 0x42,
    #[cfg(feature = "complex")]
    TRACT_DATUM_TYPE_COMPLEX_I32 = 0x44,
    #[cfg(feature = "complex")]
    TRACT_DATUM_TYPE_COMPLEX_I64 = 0x48,
    #[cfg(feature = "complex")]
    TRACT_DATUM_TYPE_COMPLEX_F16 = 0x52,
    #[cfg(feature = "complex")]
    TRACT_DATUM_TYPE_COMPLEX_F32 = 0x54,
    #[cfg(feature = "complex")]
    TRACT_DATUM_TYPE_COMPLEX_F64 = 0x58,
}

impl DatumType {
    pub fn size_of(&self) -> usize {
        use DatumType::*;
        match &self {
            TRACT_DATUM_TYPE_BOOL | TRACT_DATUM_TYPE_U8 | TRACT_DATUM_TYPE_I8 => 1,
            TRACT_DATUM_TYPE_U16 | TRACT_DATUM_TYPE_I16 | TRACT_DATUM_TYPE_F16 => 2,
            TRACT_DATUM_TYPE_U32 | TRACT_DATUM_TYPE_I32 | TRACT_DATUM_TYPE_F32 => 4,
            TRACT_DATUM_TYPE_U64 | TRACT_DATUM_TYPE_I64 | TRACT_DATUM_TYPE_F64 => 8,
            #[cfg(feature = "complex")]
            TRACT_DATUM_TYPE_COMPLEX_I16 | TRACT_DATUM_TYPE_F16 => 4,
            #[cfg(feature = "complex")]
            TRACT_DATUM_TYPE_COMPLEX_I32 | TRACT_DATUM_TYPE_F32 => 8,
            #[cfg(feature = "complex")]
            TRACT_DATUM_TYPE_COMPLEX_I64 | TRACT_DATUM_TYPE_F64 => 16,
        }
    }
}

pub trait Datum {
    fn datum_type() -> DatumType;
}

macro_rules! impl_datum_type {
    ($ty:ty, $c_repr:expr) => {
        impl Datum for $ty {
            fn datum_type() -> DatumType {
                $c_repr
            }
        }
    };
}

impl_datum_type!(bool, DatumType::TRACT_DATUM_TYPE_BOOL);
impl_datum_type!(u8, DatumType::TRACT_DATUM_TYPE_U8);
impl_datum_type!(u16, DatumType::TRACT_DATUM_TYPE_U16);
impl_datum_type!(u32, DatumType::TRACT_DATUM_TYPE_U32);
impl_datum_type!(u64, DatumType::TRACT_DATUM_TYPE_U64);
impl_datum_type!(i8, DatumType::TRACT_DATUM_TYPE_I8);
impl_datum_type!(i16, DatumType::TRACT_DATUM_TYPE_I16);
impl_datum_type!(i32, DatumType::TRACT_DATUM_TYPE_I32);
impl_datum_type!(i64, DatumType::TRACT_DATUM_TYPE_I64);
impl_datum_type!(half::f16, DatumType::TRACT_DATUM_TYPE_F16);
impl_datum_type!(f32, DatumType::TRACT_DATUM_TYPE_F32);
impl_datum_type!(f64, DatumType::TRACT_DATUM_TYPE_F64);
