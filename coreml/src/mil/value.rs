//! Constructors for MIL `ValueType`, `Dimension`, and `TensorValue` protos.

use std::collections::HashMap;

use crate::proto::core_ml::specification::mil_spec as mil;

pub use mil::DataType;

/// A static (constant) tensor dimension.
pub fn dim(size: u64) -> mil::Dimension {
    mil::Dimension {
        dimension: Some(mil::dimension::Dimension::Constant(mil::dimension::ConstantDimension {
            size,
        })),
    }
}

/// `ValueType` for a tensor of the given dtype + static shape.
///
/// `shape.len()` becomes `rank`. Pass `&[]` for a rank-0 (scalar) tensor.
pub fn tensor_type(dt: DataType, shape: &[i64]) -> mil::ValueType {
    mil::ValueType {
        r#type: Some(mil::value_type::Type::TensorType(mil::TensorType {
            data_type: dt as i32,
            rank: shape.len() as i64,
            dimensions: shape.iter().map(|&s| dim(s as u64)).collect(),
            attributes: HashMap::new(),
        })),
    }
}

/// `ValueType` for a rank-0 (scalar) tensor of the given dtype.
pub fn tensor_type_scalar(dt: DataType) -> mil::ValueType {
    tensor_type(dt, &[])
}

// -------- TensorValue constructors (one per supported repeated-primitive type) --------

pub fn tv_ints(values: Vec<i32>) -> mil::TensorValue {
    mil::TensorValue {
        value: Some(mil::tensor_value::Value::Ints(mil::tensor_value::RepeatedInts { values })),
    }
}

pub fn tv_long_ints(values: Vec<i64>) -> mil::TensorValue {
    mil::TensorValue {
        value: Some(mil::tensor_value::Value::LongInts(mil::tensor_value::RepeatedLongInts {
            values,
        })),
    }
}

pub fn tv_floats(values: Vec<f32>) -> mil::TensorValue {
    mil::TensorValue {
        value: Some(mil::tensor_value::Value::Floats(mil::tensor_value::RepeatedFloats { values })),
    }
}

pub fn tv_doubles(values: Vec<f64>) -> mil::TensorValue {
    mil::TensorValue {
        value: Some(mil::tensor_value::Value::Doubles(mil::tensor_value::RepeatedDoubles {
            values,
        })),
    }
}

pub fn tv_bools(values: Vec<bool>) -> mil::TensorValue {
    mil::TensorValue {
        value: Some(mil::tensor_value::Value::Bools(mil::tensor_value::RepeatedBools { values })),
    }
}

pub fn tv_strings(values: Vec<String>) -> mil::TensorValue {
    mil::TensorValue {
        value: Some(mil::tensor_value::Value::Strings(mil::tensor_value::RepeatedStrings {
            values,
        })),
    }
}

pub fn tv_bytes(values: Vec<u8>) -> mil::TensorValue {
    mil::TensorValue {
        value: Some(mil::tensor_value::Value::Bytes(mil::tensor_value::RepeatedBytes { values })),
    }
}
