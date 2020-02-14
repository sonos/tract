//! Enforce consistent API between the implemented Frameworks importers.
use crate::internal::*;
use crate::infer::InferenceModel;
use std::fmt::Debug;
use std::io::Read;
use std::path::Path;

/// A Framework that translate its model to tract core model.
///
/// The ProtoModel is the parsed representation of the imported model. It does
/// not have to be Protobuf based.
pub trait Framework<ProtoModel>
where
    ProtoModel: Debug,
{
    /// Parse a proto model from a reader.
    fn proto_model_for_read(&self, reader: &mut dyn Read) -> TractResult<ProtoModel>;

    /// Translate a proto model into a model.
    fn model_for_proto_model(&self, proto: &ProtoModel) -> TractResult<InferenceModel>;

    /// Read a proto model from a filename.
    fn proto_model_for_path(&self, p: impl AsRef<Path>) -> TractResult<ProtoModel> {
        let mut r = std::fs::File::open(p.as_ref())
            .map_err(|e| format!("Could not open {:?}: {}", p.as_ref(), e))?;
        self.proto_model_for_read(&mut r)
    }

    /// Read a model from a reader
    fn model_for_read(&self, r: &mut dyn Read) -> TractResult<InferenceModel> {
        let proto_model = self.proto_model_for_read(r)?;
        self.model_for_proto_model(&proto_model)
    }

    /// Build a model from a filename.
    fn model_for_path(&self, p: impl AsRef<Path>) -> TractResult<InferenceModel> {
        let mut r = std::fs::File::open(p.as_ref())
            .map_err(|e| format!("Could not open {:?}: {}", p.as_ref(), e))?;
        self.model_for_read(&mut r)
    }
}
