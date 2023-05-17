use std::fmt::Display;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Result, Context};
use ndarray::{RawData, Data, Dimension};
use tract_nnef::prelude::{Framework, TypedModel, TValue, Datum, tensor1, IntoTValue, TypedRunnableModel, TypedSimpleState, TypedSimplePlan, TVec, TypedFact};
use tract_onnx_opl::WithOnnx;

// NNEF
pub struct Nnef(tract_nnef::internal::Nnef);

pub fn nnef() -> Result<Nnef> {
    Nnef::new()
}

impl Nnef {
    pub fn new() -> Result<Nnef> {
        Ok(Nnef(tract_nnef::nnef()))
    }

    pub fn model_for_path(&self, path: impl AsRef<Path>) -> Result<Model> {
        self.0.model_for_path(path).map(Model)
    }

    pub fn with_tract_core(self) -> Result<Nnef> {
        Ok(Nnef(self.0.with_tract_core()))
    }

    pub fn with_onnx(self) -> Result<Nnef> {
        Ok(Nnef(self.0.with_onnx()))
    }

    pub fn with_extended_identifier_syntax(mut self) -> Result<Nnef> {
        self.0.allow_extended_identifier_syntax(true);
        Ok(self)
    }

    pub fn write_model_to_dir(&self, path: impl AsRef<Path>, model: &Model) -> Result<()> {
        self.0.write_to_dir(&model.0, path)
    }

    pub fn write_model_to_tar(&self, path: impl AsRef<Path>, model: &Model) -> Result<()> {
        let file = std::fs::File::create(path)?;
        self.0.write_to_tar(&model.0, file)?;
        Ok(())
    }

    pub fn write_model_to_tar_gz(&self, path: impl AsRef<Path>, model: &Model) -> Result<()> {
        let file = std::fs::File::create(path)?;
        let gz = flate2::write::GzEncoder::new(file, flate2::Compression::default());
        self.0.write_to_tar(&model.0, gz)?;
        Ok(())
    }
}

// MODEL
pub struct Model(TypedModel);

impl Model {
    pub fn input_count(&self) -> Result<usize> {
        Ok(self.0.inputs.len())
    }

    pub fn output_count(&self) -> Result<usize> {
        Ok(self.0.outputs.len())
    }

    pub fn input_name(&self, id: usize) -> Result<String> {
        let node = self.0.inputs[id].slot;
        Ok(self.0.node(node).name.to_string())
    }

    pub fn output_name(&self, id: usize) -> Result<String> {
        let node = self.0.outputs[id].node;
        Ok(self.0.node(node).name.to_string())
    }

    pub fn set_output_names(&mut self, outputs: impl IntoIterator<Item = impl AsRef<str>>) -> Result<()> {
        self.0.set_output_names(outputs)
    }

   pub fn input_fact(&self, id: usize) -> Result<Fact> {
       Ok(Fact(self.0.input_fact(id)?.clone()))
   }

   pub fn output_fact(&self, id: usize) -> Result<Fact> {
       Ok(Fact(self.0.output_fact(id)?.clone()))
   }

   pub fn declutter(&mut self) -> Result<()> {
       self.0.declutter()
   }

   pub fn optimize(&mut self) -> Result<()> {
       self.0.optimize()
   }

   pub fn into_decluttered(mut self) -> Result<Model> {
       self.0.declutter()?;
       Ok(self)
   }

   pub fn into_optimized(self) -> Result<Model> {
       Ok(Model(self.0.into_optimized()?))
   }

   pub fn into_runnable(self) -> Result<Runnable> {
       Ok(Runnable(Arc::new(self.0.into_runnable()?)))
   }
//
//    pub fn concretize_symbols(&mut self, values: impl IntoIterator<Item=(impl AsRef<str>, i64)>) -> Result<()> {
//        let (names, values):(Vec<_>, Vec<_>) = values.into_iter().unzip();
//        let c_strings:Vec<CString> = names.into_iter().map(|a| Ok(CString::new(a.as_ref())?)).collect::<Result<_>>()?;
//        let ptrs:Vec<_> = c_strings.iter().map(|cs| cs.as_ptr()).collect();
//        check!(sys::tract_model_concretize_symbols(self.0, ptrs.len(), ptrs.as_ptr(), values.as_ptr()))?;
//        Ok(())
//    }
//
//    pub fn pulse(&mut self, name: impl AsRef<str>, value: impl AsRef<str>) -> Result<()> {
//        let name = CString::new(name.as_ref())?;
//        let value = CString::new(value.as_ref())?;
//        check!(sys::tract_model_pulse_simple(&mut self.0, name.as_ptr(), value.as_ptr()))?;
//        Ok(())
//    }
//
//    pub fn cost_json(&self) -> Result<String> {
//        let input:Option<Vec<Value>> = None;
//        self.profile_json(input)
//    }
//
//    pub fn profile_json<I, V, E>(&self, inputs: Option<I>) -> Result<String>
//        where I: IntoIterator<Item = V>,
//              V: TryInto<Value, Error = E>,
//              E: Into<anyhow::Error>
//    {
//        let inputs = if let Some(inputs) =  inputs {
//            let inputs = inputs.into_iter().map(|i| i.try_into().map_err(|e| e.into())).collect::<Result<Vec<Value>>>()?;
//            anyhow::ensure!(self.input_count()? == inputs.len());
//            Some(inputs)
//        } else { None };
//        let mut iptrs:Option<Vec<*mut sys::TractValue>> = inputs.as_ref().map(|is| is.iter().map(|v| v.0).collect());
//        let mut json : *mut i8 = null_mut();
//        let values = iptrs.as_mut().map(|it| it.as_mut_ptr()).unwrap_or(null_mut());
//        check!(sys::tract_model_profile_json(self.0, values, &mut json))?;
//        anyhow::ensure!(!json.is_null());
//        unsafe {
//            let s = CStr::from_ptr(json).to_owned();
//            sys::tract_free_cstring(json);
//            Ok(s.to_str()?.to_owned())
//        }
//    }
//
//    pub fn property_keys(&self) -> Result<Vec<String>> {
//        let mut len = 0;
//        check!(sys::tract_model_property_count(self.0, &mut len))?;
//        let mut keys = vec!(null_mut(); len);
//        check!(sys::tract_model_property_names(self.0, keys.as_mut_ptr()))?;
//        unsafe {
//            keys.into_iter().map(|pc| Ok(CStr::from_ptr(pc).to_str()?.to_owned())).collect()
//        }
//    }
//
//    pub fn property(&self, name: impl AsRef<str>) -> Result<Value> {
//        let mut v = null_mut();
//        let name = CString::new(name.as_ref())?;
//        check!(sys::tract_model_property(self.0, name.as_ptr(), &mut v))?;
//        Ok(Value(v))
//    }
}

// RUNNABLE
pub struct Runnable(Arc<TypedRunnableModel<TypedModel>>);

impl Runnable {
    pub fn run<I, V, E>(&self, inputs: I) -> Result<Vec<Value>> 
        where I: IntoIterator<Item = V>,
              V: TryInto<Value, Error = E>,
              E: Into<anyhow::Error>
    {
        self.spawn_state()?.run(inputs)
    }

    pub fn spawn_state(&self) -> Result<State> {
        let state = TypedSimpleState::new(self.0.clone())?;
        Ok(State(state))
    }
}

// STATE
pub struct State(TypedSimpleState<TypedModel, Arc<TypedSimplePlan<TypedModel>>>);

impl State {
    pub fn run<I, V, E>(&mut self, inputs: I) -> Result<Vec<Value>> 
        where I: IntoIterator<Item = V>,
              V: TryInto<Value, Error = E>,
              E: Into<anyhow::Error>
    {
        let inputs:TVec<TValue> = inputs.into_iter().map(|i| i.try_into().map_err(|e| e.into()).map(|v| v.0)).collect::<Result<_>>()?;
        let outputs = self.0.run(inputs)?;
        Ok(outputs.into_iter().map(Value).collect())
    }
}

// VALUE
pub struct Value(TValue);

impl Value {
    pub fn from_shape_and_slice<T: Datum>(shape: &[usize], data: &[T]) -> Result<Value> {
        Ok(Value(tensor1(data).into_shape(&shape)?.into_tvalue()))
    }

    pub fn as_parts<T: Datum>(&self) -> Result<(&[usize], &[T])> {
        let slice = self.0.as_slice()?;
        Ok((self.0.shape(), slice))
    }

    pub fn view<T: Datum>(&self) -> Result<ndarray::ArrayViewD<T>> {
        self.0.to_array_view()
    }
}

impl<T,S,D> TryFrom<ndarray::ArrayBase<S, D>> for Value 
where T: Datum, S: RawData<Elem=T> + Data, D: Dimension
{
    type Error = anyhow::Error;
    fn try_from(view: ndarray::ArrayBase<S, D>) -> Result<Value> {
        if let Some(slice) = view.as_slice_memory_order() {
            Value::from_shape_and_slice(view.shape(), slice)
        } else {
            let slice: Vec<_> = view.iter().cloned().collect();
            Value::from_shape_and_slice(view.shape(), &slice)
        }
    }
}

impl<'a, T: Datum> TryFrom<&'a Value> for ndarray::ArrayViewD<'a, T> {
    type Error = anyhow::Error;
    fn try_from(value: &'a Value) -> Result<Self, Self::Error> {
        value.view()
    }
}

pub struct Fact(TypedFact);

impl Fact {
    pub fn new(model: &mut Model, spec: impl ToString) -> Result<Fact> {
        let fact = tract_nnef::prelude::Fact::to_typed_fact(&tract_libcli::tensor::parse_spec(&model.0.symbol_table, &spec.to_string())?)?.into_owned();
        Ok(Fact(fact))
    }

    fn dump(&self) -> Result<String> {
        Ok(format!("{:?}", self.0))
    }
}

impl Display for Fact {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.dump().unwrap())
    }
}

