//! Constructors for MIL `Operation` protos.

use std::collections::HashMap;

use crate::proto::core_ml::specification::mil_spec as mil;

/// Build a `const` op whose value lives inline in the proto (`ImmediateValue::Tensor`).
///
/// Use for small parameters: scalars, short lists like padding/strides/dilations.
/// For weight tensors of any size, prefer [`op_const_blob`] — large immediate
/// values bloat `model.mlmodel` and bypass the on-disk weight blob's mmap path.
pub fn op_const_immediate(
    name: &str,
    ty: mil::ValueType,
    value: mil::TensorValue,
) -> mil::Operation {
    let val = mil::Value {
        doc_string: String::new(),
        r#type: Some(ty.clone()),
        value: Some(mil::value::Value::ImmediateValue(mil::value::ImmediateValue {
            value: Some(mil::value::immediate_value::Value::Tensor(value)),
        })),
    };
    let mut attrs = HashMap::new();
    attrs.insert("val".to_string(), val);
    mil::Operation {
        r#type: "const".to_string(),
        inputs: HashMap::new(),
        outputs: vec![mil::NamedValueType { name: name.to_string(), r#type: Some(ty) }],
        blocks: vec![],
        attributes: attrs,
    }
}

/// Build a `const` op whose value is referenced by file/offset in an external
/// MILBlob v2 file (typically `weights/weight.bin`).
///
/// `offset` is the offset of the **`blob_metadata`** struct inside the file —
/// **not** the data offset. The metadata struct then carries the data offset
/// internally. This is the offset returned by [`crate::mil::blob::BlobBuilder::add`].
///
/// `file_name` should be the package-relative path Core ML expects, typically
/// [`crate::mlpackage::WEIGHT_BLOB_PATH`] (`"@model_path/weights/weight.bin"`).
pub fn op_const_blob(
    name: &str,
    ty: mil::ValueType,
    file_name: &str,
    offset: u64,
) -> mil::Operation {
    let val = mil::Value {
        doc_string: String::new(),
        r#type: Some(ty.clone()),
        value: Some(mil::value::Value::BlobFileValue(mil::value::BlobFileValue {
            file_name: file_name.to_string(),
            offset,
        })),
    };
    let mut attrs = HashMap::new();
    attrs.insert("val".to_string(), val);
    mil::Operation {
        r#type: "const".to_string(),
        inputs: HashMap::new(),
        outputs: vec![mil::NamedValueType { name: name.to_string(), r#type: Some(ty) }],
        blocks: vec![],
        attributes: attrs,
    }
}

/// A single-binding `Argument` referencing a previously-defined value by name.
///
/// MIL `Argument` is conceptually a list of bindings (to support variadic ops
/// like concat). For the common single-input case, this helper wraps a single
/// name reference.
pub fn arg_name(name: &str) -> mil::Argument {
    mil::Argument {
        arguments: vec![mil::argument::Binding {
            binding: Some(mil::argument::binding::Binding::Name(name.to_string())),
        }],
    }
}

/// Multi-binding `Argument` for variadic inputs (e.g. `concat`'s `values`).
pub fn arg_names(names: &[&str]) -> mil::Argument {
    mil::Argument {
        arguments: names
            .iter()
            .map(|n| mil::argument::Binding {
                binding: Some(mil::argument::binding::Binding::Name((*n).to_string())),
            })
            .collect(),
    }
}
