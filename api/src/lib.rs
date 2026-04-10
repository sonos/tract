use anyhow::{Result, ensure};
use boow::Bow;
use std::fmt::{Debug, Display};
use std::path::Path;

#[macro_use]
pub mod macros;
pub mod transform;

pub use transform::{ConcretizeSymbols, FloatPrecision, Pulse, TransformConfig, TransformSpec};

/// an implementation of tract's NNEF framework object
///
/// Entry point for NNEF model manipulation: loading from file, dumping to file.
pub trait NnefInterface: Debug + Sized {
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

pub trait OnnxInterface: Debug {
    type InferenceModel: InferenceModelInterface;
    fn load(&self, path: impl AsRef<Path>) -> Result<Self::InferenceModel>;
    /// Load a ONNX model from a buffer into an InferenceModel.
    fn load_buffer(&self, data: &[u8]) -> Result<Self::InferenceModel>;
}

pub trait InferenceModelInterface: Debug + Sized {
    type Model: ModelInterface;
    type InferenceFact: InferenceFactInterface;
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

    fn into_model(self) -> Result<Self::Model>;
}

pub trait ModelInterface: Debug + Sized {
    type Fact: FactInterface;
    type Runnable: RunnableInterface;
    type Tensor: TensorInterface;
    fn input_count(&self) -> Result<usize>;

    fn output_count(&self) -> Result<usize>;

    fn input_name(&self, id: usize) -> Result<String>;

    fn output_name(&self, id: usize) -> Result<String>;

    fn input_fact(&self, id: usize) -> Result<Self::Fact>;

    fn output_fact(&self, id: usize) -> Result<Self::Fact>;

    fn into_runnable(self) -> Result<Self::Runnable>;

    fn transform(&mut self, spec: impl Into<TransformSpec>) -> Result<()>;

    fn property_keys(&self) -> Result<Vec<String>>;

    fn property(&self, name: impl AsRef<str>) -> Result<Self::Tensor>;

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

pub trait RuntimeInterface: Debug {
    type Runnable: RunnableInterface;
    type Model: ModelInterface;
    fn name(&self) -> Result<String>;
    fn prepare(&self, model: Self::Model) -> Result<Self::Runnable>;
}

pub trait RunnableInterface: Debug + Send + Sync {
    type Tensor: TensorInterface;
    type Fact: FactInterface;
    type State: StateInterface<Tensor = Self::Tensor>;
    fn run(&self, inputs: impl IntoInputs<Self::Tensor>) -> Result<Vec<Self::Tensor>> {
        self.spawn_state()?.run(inputs.into_inputs()?)
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
    fn property(&self, name: impl AsRef<str>) -> Result<Self::Tensor>;

    fn spawn_state(&self) -> Result<Self::State>;

    fn cost_json(&self) -> Result<String>;

    fn profile_json<I, IV, IE>(&self, inputs: Option<I>) -> Result<String>
    where
        I: IntoIterator<Item = IV>,
        IV: TryInto<Self::Tensor, Error = IE>,
        IE: Into<anyhow::Error> + Debug;
}

pub trait StateInterface: Debug + Clone + Send {
    type Fact: FactInterface;
    type Tensor: TensorInterface;

    fn input_count(&self) -> Result<usize>;
    fn output_count(&self) -> Result<usize>;

    fn run(&mut self, inputs: impl IntoInputs<Self::Tensor>) -> Result<Vec<Self::Tensor>>;
}

pub trait TensorInterface: Debug + Sized + Clone + PartialEq + Send + Sync {
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

    fn view1<T: Datum>(&self) -> Result<ndarray::ArrayView1<'_, T>> {
        Ok(self.view::<T>()?.into_dimensionality()?)
    }

    fn view2<T: Datum>(&self) -> Result<ndarray::ArrayView2<'_, T>> {
        Ok(self.view::<T>()?.into_dimensionality()?)
    }

    fn view3<T: Datum>(&self) -> Result<ndarray::ArrayView3<'_, T>> {
        Ok(self.view::<T>()?.into_dimensionality()?)
    }

    fn view4<T: Datum>(&self) -> Result<ndarray::ArrayView4<'_, T>> {
        Ok(self.view::<T>()?.into_dimensionality()?)
    }

    fn view5<T: Datum>(&self) -> Result<ndarray::ArrayView5<'_, T>> {
        Ok(self.view::<T>()?.into_dimensionality()?)
    }

    fn view6<T: Datum>(&self) -> Result<ndarray::ArrayView6<'_, T>> {
        Ok(self.view::<T>()?.into_dimensionality()?)
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

pub trait AsFact<M, F>: Debug {
    fn as_fact(&self, model: &M) -> Result<Bow<'_, F>>;
}

#[repr(C)]
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum DatumType {
    Bool = 0x01,
    U8 = 0x11,
    U16 = 0x12,
    U32 = 0x14,
    U64 = 0x18,
    I8 = 0x21,
    I16 = 0x22,
    I32 = 0x24,
    I64 = 0x28,
    F16 = 0x32,
    F32 = 0x34,
    F64 = 0x38,
    #[cfg(feature = "complex")]
    ComplexI16 = 0x42,
    #[cfg(feature = "complex")]
    ComplexI32 = 0x44,
    #[cfg(feature = "complex")]
    ComplexI64 = 0x48,
    #[cfg(feature = "complex")]
    ComplexF16 = 0x52,
    #[cfg(feature = "complex")]
    ComplexF32 = 0x54,
    #[cfg(feature = "complex")]
    ComplexF64 = 0x58,
}

impl DatumType {
    pub fn size_of(&self) -> usize {
        use DatumType::*;
        match &self {
            Bool | U8 | I8 => 1,
            U16 | I16 | F16 => 2,
            U32 | I32 | F32 => 4,
            U64 | I64 | F64 => 8,
            #[cfg(feature = "complex")]
            ComplexI16 | ComplexF16 => 4,
            #[cfg(feature = "complex")]
            ComplexI32 | ComplexF32 => 8,
            #[cfg(feature = "complex")]
            ComplexI64 | ComplexF64 => 16,
        }
    }

    pub fn is_bool(&self) -> bool {
        *self == DatumType::Bool
    }

    pub fn is_number(&self) -> bool {
        *self != DatumType::Bool
    }

    pub fn is_unsigned(&self) -> bool {
        use DatumType::*;
        *self == U8 || *self == U16 || *self == U32 || *self == U64
    }

    pub fn is_signed(&self) -> bool {
        use DatumType::*;
        *self == I8 || *self == I16 || *self == I32 || *self == I64
    }

    pub fn is_float(&self) -> bool {
        use DatumType::*;
        *self == F16 || *self == F32 || *self == F64
    }
}

pub trait Datum {
    fn datum_type() -> DatumType;
}

// IntoInputs trait — ergonomic input conversion for run()
pub trait IntoInputs<V: TensorInterface> {
    fn into_inputs(self) -> Result<Vec<V>>;
}

// Arrays of anything convertible to Tensor
impl<V, T, E, const N: usize> IntoInputs<V> for [T; N]
where
    V: TensorInterface,
    T: TryInto<V, Error = E>,
    E: Into<anyhow::Error>,
{
    fn into_inputs(self) -> Result<Vec<V>> {
        self.into_iter().map(|v| v.try_into().map_err(|e| e.into())).collect()
    }
}

// Vec<V> passthrough
impl<V: TensorInterface> IntoInputs<V> for Vec<V> {
    fn into_inputs(self) -> Result<Vec<V>> {
        Ok(self)
    }
}

// Tuples — each element converts independently
macro_rules! impl_into_inputs_tuple {
    ($($idx:tt : $T:ident),+) => {
        impl<V, $($T),+> IntoInputs<V> for ($($T,)+)
        where
            V: TensorInterface,
            $($T: TryInto<V>,
              <$T as TryInto<V>>::Error: Into<anyhow::Error>,)+
        {
            fn into_inputs(self) -> Result<Vec<V>> {
                Ok(vec![$(self.$idx.try_into().map_err(|e| e.into())?),+])
            }
        }
    };
}

impl_into_inputs_tuple!(0: A);
impl_into_inputs_tuple!(0: A, 1: B);
impl_into_inputs_tuple!(0: A, 1: B, 2: C);
impl_into_inputs_tuple!(0: A, 1: B, 2: C, 3: D);
impl_into_inputs_tuple!(0: A, 1: B, 2: C, 3: D, 4: E_);
impl_into_inputs_tuple!(0: A, 1: B, 2: C, 3: D, 4: E_, 5: F);
impl_into_inputs_tuple!(0: A, 1: B, 2: C, 3: D, 4: E_, 5: F, 6: G);
impl_into_inputs_tuple!(0: A, 1: B, 2: C, 3: D, 4: E_, 5: F, 6: G, 7: H);

/// Convert any compatible input into a `V: TensorInterface`.
pub fn tensor<V, T, E>(v: T) -> Result<V>
where
    V: TensorInterface,
    T: TryInto<V, Error = E>,
    E: Into<anyhow::Error>,
{
    v.try_into().map_err(|e| e.into())
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

impl_datum_type!(bool, DatumType::Bool);
impl_datum_type!(u8, DatumType::U8);
impl_datum_type!(u16, DatumType::U16);
impl_datum_type!(u32, DatumType::U32);
impl_datum_type!(u64, DatumType::U64);
impl_datum_type!(i8, DatumType::I8);
impl_datum_type!(i16, DatumType::I16);
impl_datum_type!(i32, DatumType::I32);
impl_datum_type!(i64, DatumType::I64);
impl_datum_type!(half::f16, DatumType::F16);
impl_datum_type!(f32, DatumType::F32);
impl_datum_type!(f64, DatumType::F64);
