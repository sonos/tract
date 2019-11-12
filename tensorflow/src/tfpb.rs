use std::fs;

mod google {
    mod protobuf {
        include!(concat!(env!("OUT_DIR"), "/prost/google.protobuf.rs"));
    }
}

pub mod tensorflow {
    include!(concat!(env!("OUT_DIR"), "/prost/tensorflow.rs"));
}

use self::tensorflow::attr_value::ListValue;
use self::tensorflow::attr_value::Value;
use self::tensorflow::{AttrValue, DataType, GraphDef, NodeDef, TensorProto, TensorShapeProto};

use std::convert::TryInto;

use tract_core::internal::*;

pub fn graph() -> GraphDef {
    GraphDef { library: None, node: vec![], version: 0, versions: None }
}

pub fn node() -> NodeDef {
    NodeDef {
        name: String::new(),
        op: String::new(),
        input: vec![],
        device: String::new(),
        attr: HashMap::new(),
    }
}

impl GraphDef {
    pub fn node(mut self, n: NodeDef) -> Self {
        self.node.push(n);
        self
    }
    pub fn write_to_bytes(&self) -> TractResult<Vec<u8>> {
        use prost::Message;
        let mut buf = vec![];
        self.encode(&mut buf).map_err(|e| format!("Prost/Protobuf encoding error : {:?}", e))?;
        Ok(buf)
    }
    pub fn save_to<P: AsRef<::std::path::Path>>(self, p: P) -> TractResult<()> {
        let buf = self.write_to_bytes()?;
        fs::write(p, &buf)?;
        Ok(())
    }
}

impl NodeDef {
    pub fn name<S: ToString>(mut self, n: S) -> NodeDef {
        self.name = n.to_string();
        self
    }
    pub fn op<S: ToString>(mut self, n: S) -> NodeDef {
        self.op = n.to_string();
        self
    }
    pub fn input<S: ToString>(mut self, n: S) -> NodeDef {
        self.input.push(n.to_string());
        self
    }
    pub fn attr<S: ToString, V: Into<AttrValue>>(mut self, n: S, v: V) -> NodeDef {
        self.attr.insert(n.to_string(), v.into());
        self
    }
}

impl NodeDef {
    pub fn get_attr_raw_str(&self, name: &str) -> TractResult<&[u8]> {
        Ok(self.get_attr_opt_raw_str(name)?.ok_or_else(|| {
            format!("Node {} ({}) expected string attribute '{}'", self.name, self.op, name)
        })?)
    }

    pub fn get_attr_opt_raw_str(&self, name: &str) -> TractResult<Option<&[u8]>> {
        if let Some(a) = self.attr.get(name) {
            if let Value::S(bytes) = a.value.as_ref().unwrap() {
                return Ok(Some(&bytes));
            }
        };
        Ok(None)
    }

    pub fn get_attr_str(&self, name: &str) -> TractResult<String> {
        Ok(self.get_attr_opt_str(name)?.ok_or_else(|| {
            format!("Node {} ({}) expected UTF-8 string attribute '{}'", self.name, self.op, name)
        })?)
    }

    pub fn get_attr_opt_str(&self, name: &str) -> TractResult<Option<String>> {
        if let Some(s) = self.get_attr_opt_raw_str(name)? {
            Ok(Some(String::from_utf8(s.to_vec()).map_err(|_| {
                format!(
                    "Node {} ({}) expected an UTF-8 string for attribute '{}'",
                    self.name, self.op, name
                )
            })?))
        } else {
            Ok(None)
        }
    }

    pub fn get_attr_bool(&self, name: &str) -> TractResult<bool> {
        Ok(self.get_attr_opt_bool(name)?.ok_or_else(|| {
            format!("Node {} ({}) expected bool attribute '{}'", self.name, self.op, name)
        })?)
    }

    pub fn get_attr_opt_bool(&self, name: &str) -> TractResult<Option<bool>> {
        if let Some(a) = self.attr.get(name) {
            if let Value::B(v) = a.value.as_ref().unwrap() {
                return Ok(Some(*v));
            }
        };
        Ok(None)
    }

    pub fn get_attr_datum_type(&self, name: &str) -> TractResult<DatumType> {
        Ok(self.get_attr_opt_datum_type(name)?.ok_or_else(|| {
            format!("Node {} ({}) expected datum_type attribute '{}'", self.name, self.op, name)
        })?)
    }

    pub fn get_attr_opt_datum_type(&self, name: &str) -> TractResult<Option<DatumType>> {
        if let Some(a) = self.attr.get(name) {
            if let Value::Type(v) = a.value.as_ref().unwrap() {
                return Ok(Some(DataType::from_i32(*v).unwrap().try_into()?));
            }
        };
        Ok(None)
    }

    pub fn get_attr_shape(&self, name: &str) -> TractResult<TVec<isize>> {
        Ok(self.get_attr_opt_shape(name)?.ok_or_else(|| {
            format!("Node {} ({}) expected shape attribute '{}'", self.name, self.op, name)
        })?)
    }

    pub fn get_attr_opt_shape(&self, name: &str) -> TractResult<Option<TVec<isize>>> {
        if let Some(a) = self.attr.get(name) {
            if let Value::Shape(ref shape) = a.value.as_ref().unwrap() {
                return Ok(Some(shape.try_into()?));
            }
        };
        Ok(None)
    }

    pub fn get_attr_tensor(&self, name: &str) -> TractResult<tract_core::internal::Tensor> {
        Ok(self.get_attr_opt_tensor(name)?.ok_or_else(|| {
            format!("Node {} ({}) expected tensor attribute '{}'", self.name, self.op, name)
        })?)
    }

    pub fn get_attr_opt_tensor(
        &self,
        name: &str,
    ) -> TractResult<Option<tract_core::internal::Tensor>> {
        if let Some(a) = self.attr.get(name) {
            if let Value::Tensor(ref t) = a.value.as_ref().unwrap() {
                return Ok(Some(t.try_into()?));
            }
        };
        Ok(None)
    }

    pub fn get_attr_int<T: ::num_traits::FromPrimitive>(&self, name: &str) -> TractResult<T> {
        Ok(self.get_attr_opt_int(name)?.ok_or_else(|| {
            format!("Node {} ({}) expected int attribute '{}'", self.name, self.op, name)
        })?)
    }

    pub fn get_attr_opt_int<T: ::num_traits::FromPrimitive>(
        &self,
        name: &str,
    ) -> TractResult<Option<T>> {
        if let Some(a) = self.attr.get(name) {
            if let Value::I(i) = a.value.as_ref().unwrap() {
                return Ok(Some(T::from_i64(*i).unwrap()));
            }
        };
        Ok(None)
    }

    pub fn get_attr_float<T: ::num_traits::FromPrimitive>(&self, name: &str) -> TractResult<T> {
        Ok(self.get_attr_opt_float(name)?.ok_or_else(|| {
            format!("Node {} ({}) expected int attribute '{}'", self.name, self.op, name)
        })?)
    }

    pub fn get_attr_opt_float<T: ::num_traits::FromPrimitive>(
        &self,
        name: &str,
    ) -> TractResult<Option<T>> {
        if let Some(a) = self.attr.get(name) {
            if let Value::F(i) = a.value.as_ref().unwrap() {
                return Ok(Some(T::from_f32(*i).unwrap()));
            }
        };
        Ok(None)
    }

    pub fn get_attr_list_int<T: ::num_traits::FromPrimitive>(
        &self,
        name: &str,
    ) -> TractResult<Vec<T>> {
        Ok(self.get_attr_opt_list_int(name)?.ok_or_else(|| {
            format!("Node {} ({}) expected list<int> attribute '{}'", self.name, self.op, name)
        })?)
    }

    pub fn get_attr_opt_list_int<T: ::num_traits::FromPrimitive>(
        &self,
        name: &str,
    ) -> TractResult<Option<Vec<T>>> {
        if let Some(a) = self.attr.get(name) {
            if let Value::List(list) = a.value.as_ref().unwrap() {
                return Ok(Some(list.i.iter().map(|&i| T::from_i64(i).unwrap()).collect()));
            }
        };
        Ok(None)
    }
}

impl From<DataType> for AttrValue {
    fn from(t: DataType) -> AttrValue {
        AttrValue { value: Some(Value::Type(t.into())) }
    }
}

impl<'a> From<&'a str> for AttrValue {
    fn from(t: &'a str) -> AttrValue {
        AttrValue { value: Some(Value::S(t.as_bytes().to_vec())) }
    }
}

impl From<i32> for AttrValue {
    fn from(t: i32) -> AttrValue {
        AttrValue::from(t as i64)
    }
}

impl From<i64> for AttrValue {
    fn from(t: i64) -> AttrValue {
        AttrValue { value: Some(Value::I(t)) }
    }
}

impl From<f32> for AttrValue {
    fn from(t: f32) -> AttrValue {
        AttrValue { value: Some(Value::F(t)) }
    }
}

impl From<Vec<i64>> for AttrValue {
    fn from(t: Vec<i64>) -> AttrValue {
        AttrValue {
            value: Some(Value::List(ListValue {
                s: vec![],
                i: t,
                f: vec![],
                b: vec![],
                r#type: vec![],
                shape: vec![],
                tensor: vec![],
                func: vec![],
            })),
        }
    }
}

impl From<TensorProto> for AttrValue {
    fn from(t: TensorProto) -> AttrValue {
        AttrValue { value: Some(Value::Tensor(t.into())) }
    }
}

impl From<TensorShapeProto> for AttrValue {
    fn from(t: TensorShapeProto) -> AttrValue {
        AttrValue { value: Some(Value::Shape(t.into())) }
    }
}
