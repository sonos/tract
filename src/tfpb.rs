//! Generated protobuf codec for Tensorflow models, plus a handful of helper for
//! writting tests.

#![allow(unknown_lints)]
#![allow(clippy)]

#![cfg_attr(rustfmt, rustfmt_skip)]

#![allow(box_pointers)]
#![allow(dead_code)]
#![allow(missing_docs)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(trivial_casts)]
#![allow(unsafe_code)]
#![allow(unused_imports)]
#![allow(unused_results)]

pub mod attr_value {
    include!(concat!(env!("OUT_DIR"), "/attr_value.rs"));
}
pub mod function {
    include!(concat!(env!("OUT_DIR"), "/function.rs"));
}
pub mod graph {
    include!(concat!(env!("OUT_DIR"), "/graph.rs"));
}
pub mod node_def {
    include!(concat!(env!("OUT_DIR"), "/node_def.rs"));
}
pub mod op_def {
    include!(concat!(env!("OUT_DIR"), "/op_def.rs"));
}
pub mod resource_handle {
    include!(concat!(env!("OUT_DIR"), "/resource_handle.rs"));
}
pub mod tensor {
    include!(concat!(env!("OUT_DIR"), "/tensor.rs"));
}
pub mod tensor_shape {
    include!(concat!(env!("OUT_DIR"), "/tensor_shape.rs"));
}
pub mod types {
    include!(concat!(env!("OUT_DIR"), "/types.rs"));
}
pub mod versions {
    include!(concat!(env!("OUT_DIR"), "/versions.rs"));
}

use self::node_def::NodeDef;
use self::attr_value::AttrValue;

pub fn graph() -> graph::GraphDef {
    graph::GraphDef::new()
}

pub fn node() -> NodeDef {
    node_def::NodeDef::new()
}

pub fn tensor_f32(dim:Vec<usize>, values:Vec<f32>) -> tensor::TensorProto {
    use protobuf::singular::SingularPtrField;
    let mut tensor = tensor::TensorProto::new();
    tensor.set_dtype(types::DataType::DT_FLOAT);
    let mut shape = tensor_shape::TensorShapeProto::new();
    shape.set_dim(dim.into_iter().map(|i| {
        let mut d = tensor_shape::TensorShapeProto_Dim::new();
        d.set_size(i as _);
        d
    }).collect());
    tensor.set_tensor_shape(shape);
    tensor.set_float_val(values);
    tensor
}

impl graph::GraphDef {
    pub fn node(mut self, n: node_def::NodeDef) -> Self {
        self.mut_node().push(n);
        self
    }
    pub fn save_to<P: AsRef<::std::path::Path>>(self, p: P) -> ::Result<()> {
        use protobuf::Message;
        use std::io::Write;
        ::std::fs::File::create(p)?.write(&*self.write_to_bytes()?)?;
        Ok(())
    }
}

impl NodeDef {
    pub fn name<S: ToString>(mut self, n: S) -> NodeDef {
        self.set_name(n.to_string());
        self
    }
    pub fn op<S: ToString>(mut self, n: S) -> NodeDef {
        self.set_op(n.to_string());
        self
    }
    pub fn input<S: ToString>(mut self, n: S) -> NodeDef {
        self.mut_input().push(n.to_string());
        self
    }
    pub fn attr<S: ToString, V: Into<AttrValue>>(mut self, n: S, v: V) -> NodeDef {
        self.mut_attr().insert(n.to_string(), v.into());
        self
    }
}

impl From<types::DataType> for AttrValue {
    fn from(t: types::DataType) -> AttrValue {
        let mut dt = AttrValue::new();
        dt.set_field_type(t);
        dt
    }
}

impl<'a> From<&'a str> for AttrValue {
    fn from(t: &'a str) -> AttrValue {
        let mut value = attr_value::AttrValue::new();
        value.set_s(t.to_string().into_bytes());
        value
    }
}

impl From<Vec<i64>> for AttrValue {
    fn from(t: Vec<i64>) -> AttrValue {
        let mut list = attr_value::AttrValue_ListValue::new();
        list.set_i(t);
        let mut value = attr_value::AttrValue::new();
        value.set_list(list);
        value
    }
}

impl<'a> From<tensor::TensorProto> for AttrValue {
    fn from(t: tensor::TensorProto) -> AttrValue {
        let mut value = attr_value::AttrValue::new();
        value.set_tensor(t);
        value
    }
}

