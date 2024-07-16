//! Enforce consistent API between the implemented Frameworks importers.
use crate::internal::*;
use std::fmt::Debug;
use std::io::Read;
use std::path::Path;

/// A Framework that translate its model to tract core model.
///
/// The ProtoModel is the parsed representation of the imported model. It does
/// not have to be Protobuf based.
pub trait Framework<ProtoModel, Model>: Send + Sync
where
    ProtoModel: Debug,
    Model: Default,
{
    /// Parse a proto model from a reader.
    fn proto_model_for_read(&self, reader: &mut dyn Read) -> TractResult<ProtoModel>;

    /// Translate a proto model into a model.
    fn model_for_proto_model(&self, proto: &ProtoModel) -> TractResult<Model> {
        self.model_for_proto_model_with_model_template(proto, Model::default())
    }

    /// Translate a proto model into a model, with some symbols already listed.
    fn model_for_proto_model_with_model_template(
        &self,
        proto: &ProtoModel,
        template: Model,
    ) -> TractResult<Model>;

    /// Read a proto model from a filename.
    fn proto_model_for_path(&self, p: impl AsRef<Path>) -> TractResult<ProtoModel> {
        let mut r = std::fs::File::open(p.as_ref())
            .with_context(|| format!("Could not open {:?}", p.as_ref()))?;
        self.proto_model_for_read(&mut r)
    }

    /// Read a model from a reader
    fn model_for_read(&self, r: &mut dyn Read) -> TractResult<Model> {
        let proto_model = self.proto_model_for_read(r).context("Reading proto model")?;
        self.model_for_proto_model(&proto_model).context("Translating proto model to model")
    }

    /// Build a model from a filename.
    fn model_for_path(&self, p: impl AsRef<Path>) -> TractResult<Model> {
        let mut r = std::fs::File::open(p.as_ref())
            .with_context(|| format!("Could not open {:?}", p.as_ref()))?;
        self.model_for_read(&mut r)
    }
}
