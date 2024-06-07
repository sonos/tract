use metal::{FunctionConstantValues, MTLDataType};
use std::ffi::c_void;

/// From candle-metal-kernels
#[derive(Debug, PartialEq)]
pub enum Value {
    USize(usize),
    Bool(bool),
    F32(f32),
    U16(u16),
}

impl std::hash::Hash for Value {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Value::F32(v) => v.to_bits().hash(state),
            Value::USize(v) => v.hash(state),
            Value::U16(v) => v.hash(state),
            Value::Bool(v) => v.hash(state),
        }
    }
}

impl Value {
    fn data_type(&self) -> MTLDataType {
        match self {
            Value::USize(_) => MTLDataType::UInt,
            Value::F32(_) => MTLDataType::Float,
            Value::U16(_) => MTLDataType::UShort,
            Value::Bool(_) => MTLDataType::Bool,
        }
    }
}

// Not true, good enough for our purposes.
impl Eq for Value {}

/// From candle-metal-kernels
#[derive(Debug, Eq, PartialEq, Hash)]
pub(crate) struct ConstantValues(Vec<(usize, Value)>);

impl ConstantValues {
    pub fn new(values: Vec<(usize, Value)>) -> Self {
        Self(values)
    }

    pub fn function_constant_values(&self) -> FunctionConstantValues {
        let f = FunctionConstantValues::new();
        for (index, value) in &self.0 {
            let ty = value.data_type();
            match value {
                Value::USize(v) => {
                    f.set_constant_value_at_index(
                        v as *const usize as *const c_void,
                        ty,
                        *index as u64,
                    );
                }
                Value::F32(v) => {
                    f.set_constant_value_at_index(
                        v as *const f32 as *const c_void,
                        ty,
                        *index as u64,
                    );
                }
                Value::U16(v) => {
                    f.set_constant_value_at_index(
                        v as *const u16 as *const c_void,
                        ty,
                        *index as u64,
                    );
                }
                Value::Bool(v) => {
                    f.set_constant_value_at_index(
                        v as *const bool as *const c_void,
                        ty,
                        *index as u64,
                    );
                }
            }
        }
        f
    }
}
