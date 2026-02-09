use anyhow::{Result, ensure};
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
    fn load(&self, path: impl AsRef<Path>) -> Result<Self::Model>;

    /// Load a NNEF model from a buffer into a tract-core model.
    ///
    /// data is the content of a NNEF model, as a `tar` file or a `tar.gz` file.
    fn load_buffer(&self, data: &[u8]) -> Result<Self::Model>;

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
    fn load(&self, path: impl AsRef<Path>) -> Result<Self::InferenceModel>;
    /// Load a ONNX model from a buffer into an InferenceModel.
    fn load_buffer(&self, data: &[u8]) -> Result<Self::InferenceModel>;
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

    fn into_tract(self) -> Result<Self::Model>;
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

    fn into_runnable(self) -> Result<Self::Runnable>;

    fn concretize_symbols(
        &mut self,
        values: impl IntoIterator<Item = (impl AsRef<str>, i64)>,
    ) -> Result<()>;

    fn transform(&mut self, transform: &str) -> Result<()>;

    fn pulse(&mut self, name: impl AsRef<str>, value: impl AsRef<str>) -> Result<()>;

    fn property_keys(&self) -> Result<Vec<String>>;

    fn property(&self, name: impl AsRef<str>) -> Result<Self::Value>;

    fn parse_fact(&self, spec: &str) -> Result<Self::Fact>;

    fn input_facts(&self) -> Result<impl Iterator<Item = Self::Fact>> {
        Ok((0..self.input_count()?)
            .map(|ix| self.input_fact(ix))
            .collect::<Result<Vec<_>>>()?
            .into_iter())
    }

    fn output_facts(&self) -> Result<impl Iterator<Item = Self::Fact>> {
        Ok((0..self.output_count()?)
            .map(|ix| self.output_fact(ix))
            .collect::<Result<Vec<_>>>()?
            .into_iter())
    }
}

pub trait RuntimeInterface {
    type Runnable: RunnableInterface;
    type Model: ModelInterface;
    fn prepare(&self, model: Self::Model) -> Result<Self::Runnable>;
}

pub trait RunnableInterface {
    type Value: ValueInterface;
    type Fact: FactInterface;
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
    fn input_fact(&self, id: usize) -> Result<Self::Fact>;

    fn output_fact(&self, id: usize) -> Result<Self::Fact>;

    fn input_facts(&self) -> Result<impl Iterator<Item = Self::Fact>> {
        Ok((0..self.input_count()?)
            .map(|ix| self.input_fact(ix))
            .collect::<Result<Vec<_>>>()?
            .into_iter())
    }

    fn output_facts(&self) -> Result<impl Iterator<Item = Self::Fact>> {
        Ok((0..self.output_count()?)
            .map(|ix| self.output_fact(ix))
            .collect::<Result<Vec<_>>>()?
            .into_iter())
    }

    fn property_keys(&self) -> Result<Vec<String>>;
    fn property(&self, name: impl AsRef<str>) -> Result<Self::Value>;

    fn spawn_state(&self) -> Result<Self::State>;

    fn cost_json(&self) -> Result<String>;

    fn profile_json<I, IV, IE, S, SV, SE>(
        &self,
        inputs: Option<I>,
        state_initializers: Option<S>,
    ) -> Result<String>
    where
        I: IntoIterator<Item = IV>,
        IV: TryInto<Self::Value, Error = IE>,
        IE: Into<anyhow::Error> + Debug,
        S: IntoIterator<Item = SV>,
        SV: TryInto<Self::Value, Error = SE>,
        SE: Into<anyhow::Error> + Debug;
}

pub trait StateInterface {
    type Fact: FactInterface;
    type Value: ValueInterface;

    fn input_count(&self) -> Result<usize>;
    fn output_count(&self) -> Result<usize>;

    fn run<I, V, E>(&mut self, inputs: I) -> Result<Vec<Self::Value>>
    where
        I: IntoIterator<Item = V>,
        V: TryInto<Self::Value, Error = E>,
        E: Into<anyhow::Error>;

    fn initializable_states_count(&self) -> Result<usize>;

    fn get_states_facts(&self) -> Result<Vec<Self::Fact>>;

    fn set_states<I, V, E>(&mut self, state_initializers: I) -> Result<()>
    where
        I: IntoIterator<Item = V>,
        V: TryInto<Self::Value, Error = E>,
        E: Into<anyhow::Error> + Debug;

    fn get_states(&self) -> Result<Vec<Self::Value>>;
}

pub trait ValueInterface: Debug + Sized + Clone + PartialEq + Send + Sync {
    fn datum_type(&self) -> Result<DatumType>;
    fn from_bytes(dt: DatumType, shape: &[usize], data: &[u8]) -> Result<Self>;
    fn as_bytes(&self) -> Result<(DatumType, &[usize], &[u8])>;

    fn from_slice<T: Datum>(shape: &[usize], data: &[T]) -> Result<Self> {
        let data = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
        };
        Self::from_bytes(T::datum_type(), shape, data)
    }

    fn as_slice<T: Datum>(&self) -> Result<&[T]> {
        let (dt, _shape, data) = self.as_bytes()?;
        ensure!(T::datum_type() == dt);
        let data = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const T,
                data.len() / std::mem::size_of::<T>(),
            )
        };
        Ok(data)
    }

    fn as_shape_and_slice<T: Datum>(&self) -> Result<(&[usize], &[T])> {
        let (_, shape, _) = self.as_bytes()?;
        let data = self.as_slice()?;
        Ok((shape, data))
    }

    fn shape(&self) -> Result<&[usize]> {
        let (_, shape, _) = self.as_bytes()?;
        Ok(shape)
    }

    fn view<T: Datum>(&self) -> Result<ndarray::ArrayViewD<'_, T>> {
        let (shape, data) = self.as_shape_and_slice()?;
        Ok(unsafe { ndarray::ArrayViewD::from_shape_ptr(shape, data.as_ptr()) })
    }

    fn convert_to(&self, to: DatumType) -> Result<Self>;
}

pub trait FactInterface: Debug + Display + Clone {
    type Dim: DimInterface;
    fn datum_type(&self) -> Result<DatumType>;
    fn rank(&self) -> Result<usize>;
    fn dim(&self, axis: usize) -> Result<Self::Dim>;

    fn dims(&self) -> Result<impl Iterator<Item = Self::Dim>> {
        Ok((0..self.rank()?).map(|axis| self.dim(axis)).collect::<Result<Vec<_>>>()?.into_iter())
    }
}

pub trait DimInterface: Debug + Display + Clone {
    fn eval(&self, values: impl IntoIterator<Item = (impl AsRef<str>, i64)>) -> Result<Self>;
    fn to_int64(&self) -> Result<i64>;
}

pub trait InferenceFactInterface: Debug + Display + Default + Clone {
    fn empty() -> Result<Self>;
}

pub trait AsFact<M, F> {
    fn as_fact(&self, model: &M) -> Result<Bow<'_, F>>;
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

    pub fn is_bool(&self) -> bool {
        use DatumType::*;
        *self == TRACT_DATUM_TYPE_BOOL
    }

    pub fn is_number(&self) -> bool {
        use DatumType::*;
        *self != TRACT_DATUM_TYPE_BOOL
    }

    pub fn is_unsigned(&self) -> bool {
        use DatumType::*;
        *self == TRACT_DATUM_TYPE_U8
            || *self == TRACT_DATUM_TYPE_U16
            || *self == TRACT_DATUM_TYPE_U32
            || *self == TRACT_DATUM_TYPE_U64
    }

    pub fn is_signed(&self) -> bool {
        use DatumType::*;
        *self == TRACT_DATUM_TYPE_I8
            || *self == TRACT_DATUM_TYPE_I16
            || *self == TRACT_DATUM_TYPE_I32
            || *self == TRACT_DATUM_TYPE_I64
    }

    pub fn is_float(&self) -> bool {
        use DatumType::*;
        *self == TRACT_DATUM_TYPE_F16
            || *self == TRACT_DATUM_TYPE_F32
            || *self == TRACT_DATUM_TYPE_F64
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
