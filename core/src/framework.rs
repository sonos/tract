use std::io::Read;
use std::fmt::Debug;
use std::path::Path;
use crate::ops::prelude::*;
use crate::model::Model;

type OpBuilder<ProtoOp> = fn(&ProtoOp) -> TractResult<Box<Op>>;

pub struct Framework<ProtoOp: Debug, ProtoModel: Debug> {
    pub ops: HashMap<String, OpBuilder<ProtoOp>>,
    pub model_loader: Box<Fn(&mut Read) -> TractResult<ProtoModel>>,
    pub model_builder: Box<Fn(&ProtoModel, &Framework<ProtoOp, ProtoModel>) -> TractResult<Model>>,
}

impl<ProtoOp: Debug, ProtoModel: Debug> Framework<ProtoOp, ProtoModel> {
    pub fn build_op(&self, name: &str, payload: &ProtoOp) -> TractResult<Box<Op>> {
        match self.ops.get(name) {
            Some(builder) => builder(payload),
            None => Ok(Box::new(crate::ops::unimpl::UnimplementedOp::new(
                name,
                format!("{:?}", payload),
            ))),
        }
    }

    pub fn insert(&mut self, name: &str, builder: fn(&ProtoOp) -> TractResult<Box<Op>>) {
        self.ops.insert(name.to_string(), builder);
    }

    pub fn proto_model_for_reader(&self, r: &mut Read) -> TractResult<ProtoModel> {
        (self.model_loader)(r)
    }

    pub fn proto_model_for_path(&self, p: impl AsRef<Path>) -> TractResult<ProtoModel> {
        let mut r = std::fs::File::open(p)?;
        (self.model_loader)(&mut r)
    }

    pub fn model_for_proto_model(&self, proto_model: &ProtoModel) -> TractResult<Model> {
        (self.model_builder)(&proto_model, &self)
    }

    pub fn model_for_reader(&self, r: &mut Read) -> TractResult<Model> {
        let proto_model = self.proto_model_for_reader(r)?;
        self.model_for_proto_model(&proto_model)
    }

    pub fn model_for_path(&self, p: impl AsRef<Path>) -> TractResult<Model> {
        let proto_model = self.proto_model_for_path(p)?;
        self.model_for_proto_model(&proto_model)
    }
}
