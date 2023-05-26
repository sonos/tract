use std::fmt::{Debug, Display};
use std::path::Path;

use boow::Bow;
use sys::TractDatumType;
use tract_nnef::prelude::{Datum, DatumType};
use tract_rs_sys as sys;

use anyhow::Result;

pub trait TractInterface {
    type Nnef: NnefInterface;
    type Onnx: OnnxInterface;
    type Value: ValueInterface;

    fn version() -> &'static str;
    fn nnef() -> Result<Self::Nnef>;
    fn onnx() -> Result<Self::Onnx>;

    fn value_from_shape_and_slice<T: TractProxyDatumType>(shape: &[usize], data: &[T]) -> Result<Self::Value>;
}

pub trait NnefInterface: Sized {
    type Model: ModelInterface;
    fn model_for_path(&self, path: impl AsRef<Path>) -> Result<Self::Model>;

    fn with_tract_core(self) -> Result<Self>;
    fn with_onnx(self) -> Result<Self>;
    fn with_extended_identifier_syntax(self) -> Result<Self>;

    fn write_model_to_dir(&self, path: impl AsRef<Path>, model: &Self::Model) -> Result<()>;
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

    fn half(&mut self) -> Result<()>;

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

    fn spawn_state(&self) -> Result<Self::State>;
}

pub trait StateInterface {
    type Value: ValueInterface;
    fn run<I, V, E>(&mut self, inputs: I) -> Result<Vec<Self::Value>>
    where
        I: IntoIterator<Item = V>,
        V: TryInto<Self::Value, Error = E>,
        E: Into<anyhow::Error>;
}

pub trait ValueInterface: Sized {
    fn from_shape_and_slice<T: TractProxyDatumType>(shape: &[usize], data: &[T]) -> Result<Self>;

    fn as_parts<T: TractProxyDatumType>(&self) -> Result<(&[usize], &[T])>;

    fn view<T: TractProxyDatumType>(&self) -> Result<ndarray::ArrayViewD<T>>;
}

pub trait FactInterface: Display {}
pub trait InferenceFactInterface: Display {}

pub trait AsFact<M, F> {
    fn as_fact(&self, model: &mut M) -> Result<Bow<F>>;
}

pub use tract_onnx::prelude::Datum as TractProxyDatumType;

/*
pub trait TractProxyDatumType: Datum {
    fn c_repr() -> TractDatumType;
}

macro_rules! impl_datum_type {
    ($ty:ty, $c_repr:expr) => {
        impl TractProxyDatumType for $ty {
            fn c_repr() -> TractDatumType {
                $c_repr
            }
        }
    };
}

impl_datum_type!(bool, sys::TractDatumType_TRACT_DATUM_TYPE_BOOL);
impl_datum_type!(u8, sys::TractDatumType_TRACT_DATUM_TYPE_U8);
impl_datum_type!(u16, sys::TractDatumType_TRACT_DATUM_TYPE_U16);
impl_datum_type!(u32, sys::TractDatumType_TRACT_DATUM_TYPE_U32);
impl_datum_type!(u64, sys::TractDatumType_TRACT_DATUM_TYPE_U64);
impl_datum_type!(i8, sys::TractDatumType_TRACT_DATUM_TYPE_I8);
impl_datum_type!(i16, sys::TractDatumType_TRACT_DATUM_TYPE_I16);
impl_datum_type!(i32, sys::TractDatumType_TRACT_DATUM_TYPE_I32);
impl_datum_type!(i64, sys::TractDatumType_TRACT_DATUM_TYPE_I64);
impl_datum_type!(half::f16, sys::TractDatumType_TRACT_DATUM_TYPE_F16);
impl_datum_type!(f32, sys::TractDatumType_TRACT_DATUM_TYPE_F32);
impl_datum_type!(f64, sys::TractDatumType_TRACT_DATUM_TYPE_F64);
*/
