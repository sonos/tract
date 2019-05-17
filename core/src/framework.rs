//! Enforce consistent API between the implemented Frameworks importers.
use crate::internal::*;
use crate::model::InferenceModel;
use crate::ops::unimpl::UnimplementedOp;
use std::fmt::Debug;
use std::io::Read;
use std::path::Path;

/// Build an Op from its proto representation (not necessarily protobuf).
pub type OpBuilder<ProtoOp, O> = fn(&ProtoOp) -> TractResult<O>;

/// An index of OpBuilder by name.
pub struct OpRegister<ProtoOp, O>(HashMap<String, OpBuilder<ProtoOp, O>>);

impl<ProtoOp, O> Default for OpRegister<ProtoOp, O> {
    fn default() -> OpRegister<ProtoOp, O> {
        OpRegister(HashMap::new())
    }
}

impl<ProtoOp, O> OpRegister<ProtoOp, O> {
    pub fn get(&self, name: &str) -> Option<&OpBuilder<ProtoOp, O>> {
        self.0.get(name)
    }
    pub fn insert(&mut self, name: impl AsRef<str>, b: OpBuilder<ProtoOp, O>) {
        self.0.insert(name.as_ref().to_string(), b);
    }
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.0.keys().map(|s| &**s)
    }
}

/// A Framework that translate its model to tract core model.
///
/// The ProtoModel is the parsed representation of the imported model. It does
/// not have to be Protobuf based.
pub trait Framework<ProtoOp: Debug, O: From<UnimplementedOp>, ProtoModel: Debug> {
    /// Find the OpBuilder for an operation name.
    fn op_builder_for_name(&self, name: &str) -> Option<&OpBuilder<ProtoOp, O>>;

    /// Parse a proto model from a reader.
    fn proto_model_for_read(&self, reader: &mut Read) -> TractResult<ProtoModel>;

    /// Translate a proto model into a model.
    fn model_for_proto_model(&self, proto: &ProtoModel) -> TractResult<InferenceModel>;

    /// Read a proto model from a filename.
    fn proto_model_for_path(&self, p: impl AsRef<Path>) -> TractResult<ProtoModel> {
        let mut r = std::fs::File::open(p.as_ref())
            .map_err(|e| format!("Could not open {:?}: {}", p.as_ref(), e))?;
        self.proto_model_for_read(&mut r)
    }

    /// Read a model from a reader
    fn model_for_read(&self, r: &mut Read) -> TractResult<InferenceModel> {
        let proto_model = self.proto_model_for_read(r)?;
        self.model_for_proto_model(&proto_model)
    }

    /// Build a model from a filename.
    fn model_for_path(&self, p: impl AsRef<Path>) -> TractResult<InferenceModel> {
        let mut r = std::fs::File::open(p.as_ref())
            .map_err(|e| format!("Could not open {:?}: {}", p.as_ref(), e))?;
        self.model_for_read(&mut r)
    }

    /// Build an op from its representation in the ProtoModel.
    ///
    /// This method stub wraps unknown operations in UnimplementedOp.
    fn build_op(&self, name: &str, payload: &ProtoOp) -> TractResult<O> {
        match self.op_builder_for_name(name) {
            Some(builder) => builder(payload),
            None => Ok(UnimplementedOp::new(name, format!("{:?}", payload)).into()),
        }
    }

}
