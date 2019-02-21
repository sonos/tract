//! Generated protobuf codec for SharedTensor models, plus a handful of helper for
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

pub mod attr_value;
pub mod function;
pub mod graph;
pub mod node_def;
pub mod op_def;
pub mod resource_handle;
pub mod tensor;
pub mod tensor_shape;
pub mod types;
pub mod versions;

use self::node_def::NodeDef;
use self::attr_value::AttrValue;

use tract_core::{ TractResult, ToTract };
use tract_core::model::TVec;

pub fn graph() -> graph::GraphDef {
    graph::GraphDef::new()
}

pub fn node() -> NodeDef {
    node_def::NodeDef::new()
}

pub fn tensor_f32(dim:Vec<usize>, values:Vec<f32>) -> tensor::TensorProto {
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
    pub fn save_to<P: AsRef<::std::path::Path>>(self, p: P) -> TractResult<()> {
        use protobuf::Message;
        use std::io::Write;
        ::std::fs::File::create(p)?.write(&*self.write_to_bytes().map_err(|e| format!("{:?}", e))?)?;
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

impl node_def::NodeDef {
    pub fn get_attr_raw_str(&self, name: &str) -> TractResult<&[u8]> {
        Ok(self.get_attr_opt_raw_str(name)?
            .ok_or_else(|| format!("Node {} ({}) expected string attribute '{}'", self.get_name(), self.get_op(), name))?)
    }

    pub fn get_attr_opt_raw_str(&self, name: &str) -> TractResult<Option<&[u8]>> {
        Ok(self.get_attr().get(name).map(|v| v.get_s()))
    }

    pub fn get_attr_str(&self, name: &str) -> TractResult<String> {
        Ok(self.get_attr_opt_str(name)?
            .ok_or_else(|| format!("Node {} ({}) expected UTF-8 string attribute '{}'", self.get_name(), self.get_op(), name))?)
    }

    pub fn get_attr_opt_str(&self, name: &str) -> TractResult<Option<String>> {
        if let Some(s) = self.get_attr_opt_raw_str(name)? {
            Ok(Some(String::from_utf8(s.to_vec())
                .map_err(|_| format!("Node {} ({}) expected an UTF-8 string for attribute '{}'", self.get_name(), self.get_op(), name))?))
        } else {
            Ok(None)
        }
    }

    pub fn get_attr_bool(&self, name: &str) -> TractResult<bool> {
        Ok(self.get_attr_opt_bool(name)?
            .ok_or_else(|| format!("Node {} ({}) expected bool attribute '{}'", self.get_name(), self.get_op(), name))?)
    }

    pub fn get_attr_opt_bool(&self, name: &str) -> TractResult<Option<bool>> {
        if let Some(t) = self.get_attr().get(name) {
            Ok(Some(t.get_b()))
        } else {
            Ok(None)
        }
    }

    pub fn get_attr_datum_type(&self, name: &str) -> TractResult<tract_core::DatumType> {
        Ok(self.get_attr_opt_datum_type(name)?
            .ok_or_else(|| format!("Node {} ({}) expected datum_type attribute '{}'", self.get_name(), self.get_op(), name))?)
    }

    pub fn get_attr_opt_datum_type(&self, name: &str) -> TractResult<Option<tract_core::DatumType>> {
        if let Some(t) = self.get_attr().get(name) {
            Ok(Some(t.get_field_type().tractify()?))
        } else {
            Ok(None)
        }
    }

    pub fn get_attr_shape(&self, name: &str) -> TractResult<TVec<usize>> {
        Ok(self.get_attr_opt_shape(name)?
            .ok_or_else(|| format!("Node {} ({}) expected shape attribute '{}'", self.get_name(), self.get_op(), name))?)
    }

    pub fn get_attr_opt_shape(&self, name: &str) -> TractResult<Option<TVec<usize>>> {
        if let Some(t) = self.get_attr().get(name).map(|v| v.get_shape()) {
            Ok(Some(t.tractify()?))
        } else {
            Ok(None)
        }
    }

    pub fn get_attr_tensor(&self, name: &str) -> TractResult<tract_core::Tensor> {
        Ok(self.get_attr_opt_tensor(name)?
            .ok_or_else(|| format!("Node {} ({}) expected tensor attribute '{}'", self.get_name(), self.get_op(), name))?)
    }

    pub fn get_attr_opt_tensor(&self, name: &str) -> TractResult<Option<tract_core::Tensor>> {
        if let Some(t) = self.get_attr().get(name).map(|v| v.get_tensor()) {
            Ok(Some(t.tractify()?))
        } else {
            Ok(None)
        }
    }

    pub fn get_attr_int<T: ::num_traits::FromPrimitive>(&self, name: &str) -> TractResult<T> {
        Ok(self.get_attr_opt_int(name)?
            .ok_or_else(|| format!("Node {} ({}) expected int attribute '{}'", self.get_name(), self.get_op(), name))?)
    }

    pub fn get_attr_opt_int<T: ::num_traits::FromPrimitive>(&self, name: &str) -> TractResult<Option<T>> {
        if let Some(i) = self.get_attr().get(name) {
            Ok(Some(T::from_i64(i.get_i())
                .ok_or_else(|| format!("Node {} ({}) expected int attribute '{}'", self.get_name(), self.get_op(), name))?))
        } else {
            Ok(None)
        }
    }

    pub fn get_attr_float<T: ::num_traits::FromPrimitive>(&self, name: &str) -> TractResult<T> {
        Ok(self.get_attr_opt_float(name)?
            .ok_or_else(|| format!("Node {} ({}) expected int attribute '{}'", self.get_name(), self.get_op(), name))?)
    }

    pub fn get_attr_opt_float<T: ::num_traits::FromPrimitive>(&self, name: &str) -> TractResult<Option<T>> {
        if let Some(i) = self.get_attr().get(name) {
            Ok(Some(T::from_f32(i.get_f())
                .ok_or_else(|| format!("Node {} ({}) expected int attribute '{}'", self.get_name(), self.get_op(), name))?))
        } else {
            Ok(None)
        }
    }

    pub fn get_attr_list_int<T: ::num_traits::FromPrimitive>(&self, name: &str) -> TractResult<Vec<T>> {
        Ok(self.get_attr_opt_list_int(name)?
            .ok_or_else(|| format!("Node {} ({}) expected list<int> attribute '{}'", self.get_name(), self.get_op(), name))?)
    }

    pub fn get_attr_opt_list_int<T: ::num_traits::FromPrimitive>(&self, name: &str) -> TractResult<Option<Vec<T>>> {
        if let Some(list) = self.get_attr().get(name) {
            Ok(Some(list.get_list().get_i().iter().map(|i| T::from_i64(*i)
                .ok_or_else(|| format!("Node {} ({}) expected list<int> attribute '{}'", self.get_name(), self.get_op(), name).into()))
                .collect::<TractResult<Vec<T>>>()?))
        } else {
            Ok(None)
        }
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

impl From<i64> for AttrValue {
    fn from(t: i64) -> AttrValue {
        let mut value = attr_value::AttrValue::new();
        value.set_i(t);
        value
    }
}

impl From<f32> for AttrValue {
    fn from(t: f32) -> AttrValue {
        let mut value = attr_value::AttrValue::new();
        value.set_f(t);
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

impl<'a> From<tensor_shape::TensorShapeProto> for AttrValue {
    fn from(t: tensor_shape::TensorShapeProto) -> AttrValue {
        let mut value = attr_value::AttrValue::new();
        value.set_shape(t);
        value
    }
}

