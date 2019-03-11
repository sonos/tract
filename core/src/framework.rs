use std::io::Read;
use std::fmt::Debug;
use std::path::Path;
use crate::ops::prelude::*;
use crate::model::Model;
use crate::ops::unimpl::UnimplementedOp;

pub type OpBuilder<ProtoOp> = fn(&ProtoOp) -> TractResult<Box<Op>>;

#[derive(Default)]
pub struct OpRegister<ProtoOp>(HashMap<String, OpBuilder<ProtoOp>>);

impl<ProtoOp> OpRegister<ProtoOp> {
    pub fn get(&self, name: &str) -> Option<&OpBuilder<ProtoOp>> {
        self.0.get(name)
    }
    pub fn insert(&mut self, name: impl AsRef<str>, b: OpBuilder<ProtoOp>) {
        self.0.insert(name.as_ref().to_string(), b);
    }
    pub fn names(&self) -> impl Iterator<Item=&str> {
        self.0.keys().map(|s| &**s)
    }
}

pub trait Framework<ProtoOp: Debug, ProtoModel: Debug> {
    fn op_builder_for_name(&self, name: &str) -> Option<&OpBuilder<ProtoOp>>;
    fn proto_model_for_read(&self, reader: &mut Read) -> TractResult<ProtoModel>;
    fn model_for_proto_model(&self, proto: &ProtoModel) -> TractResult<Model>;

    fn proto_model_for_path(&self, p: impl AsRef<Path>) -> TractResult<ProtoModel> {
        let mut r = std::fs::File::open(p)?;
        self.proto_model_for_read(&mut r)
    }

    fn model_for_read(&self, r: &mut Read) -> TractResult<Model> {
        let proto_model = self.proto_model_for_read(r)?;
        self.model_for_proto_model(&proto_model)
    }

    fn model_for_path(&self, p: impl AsRef<Path>) -> TractResult<Model> {
        let mut r = std::fs::File::open(p)?;
        self.model_for_read(&mut r)
    }

    fn build_op(&self, name: &str, payload: &ProtoOp) -> TractResult<Box<Op>> {
        match self.op_builder_for_name(name) {
            Some(builder) => builder(payload),
            None => Ok(Box::new(UnimplementedOp::new(
                name,
                format!("{:?}", payload),
            ))),
        }
    }
}

