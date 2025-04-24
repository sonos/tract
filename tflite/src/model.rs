use std::collections::hash_map::Entry;
use std::fmt::Debug;

use flatbuffers::FlatBufferBuilder;
use tract_core::internal::*;

use crate::registry::Registry;
use crate::tensors::{flat_tensor_to_tract_fact, flat_tensor_uses_per_axis_q};
use crate::tflite;
use crate::tflite::{Buffer, BufferArgs};

pub struct Tflite(Registry);

impl Debug for Tflite {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "tract-TfLite-framework")
    }
}

impl Default for Tflite {
    fn default() -> Self {
        let mut registry = Registry::default();
        crate::ops::register_all(&mut registry);
        Tflite(registry)
    }
}

#[derive(Clone, Debug)]
pub struct TfliteProtoModel(Vec<u8>);

impl TfliteProtoModel {
    fn new(buf: Vec<u8>) -> TractResult<TfliteProtoModel> {
        let _ = tflite::root_as_model(&buf)?;
        Ok(TfliteProtoModel(buf))
    }

    pub fn root(&self) -> tflite::Model {
        unsafe { tflite::root_as_model_unchecked(&self.0) }
    }
}

fn write_model<'fb>(
    registry: &Registry,
    model: &TypedModel,
) -> TractResult<FlatBufferBuilder<'fb>> {
    let mut model = model.clone();
    crate::rewriter::rewrite_for_tflite(&mut model).context("Pre-dump rewrite")?;
    let mut builder = flatbuffers::FlatBufferBuilder::new();
    let mut op_codes = vec![];
    let sentinel = Buffer::create(&mut builder, &BufferArgs { data: None });
    let mut buffers = vec![sentinel];
    crate::ser::ModelBuilder {
        registry,
        builder: &mut builder,
        op_codes: &mut op_codes,
        buffers: &mut buffers,
    }
    .write_model(&model)?;
    Ok(builder)
}

impl Tflite {
    pub fn write(&self, model: &TypedModel, mut w: impl std::io::Write) -> TractResult<()> {
        let builder = write_model(&self.0, model)?;
        w.write_all(builder.finished_data())?;
        Ok(())
    }
}

impl Framework<TfliteProtoModel, TypedModel> for Tflite {
    fn proto_model_for_read(
        &self,
        reader: &mut dyn std::io::Read,
    ) -> tract_core::prelude::TractResult<TfliteProtoModel> {
        let mut buf = vec![];
        reader.read_to_end(&mut buf)?;
        TfliteProtoModel::new(buf)
    }

    fn model_for_proto_model_with_model_template(
        &self,
        proto: &TfliteProtoModel,
        mut target: TypedModel,
    ) -> TractResult<TypedModel> {
        let root = proto.root();
        let main = &root.subgraphs().context("No subgraphs in Tflite model")?.get(0);
        let mut mapping = HashMap::new();
        for input in main.inputs().context("No inputs in Tflite model")? {
            if !flat_tensor_uses_per_axis_q(main, input) {
                let (fact, name) = flat_tensor_to_tract_fact(&root, main, input)?;
                let it = target.add_source(name, fact)?;
                mapping.insert(input, it);
            }
        }
        for op in main.operators().context("No operators in Tflite model")? {
            for input in op.inputs().context("No input in Tflite  operator")? {
                if let Entry::Vacant(slot) = mapping.entry(input) {
                    let (fact, name) = flat_tensor_to_tract_fact(&root, main, input)?;
                    let value = fact.konst.with_context(|| format!("Error in TF file for operator {op:?}. No prior computation nor constant for input {input}"))?;
                    let konst = target.add_const(name, value)?;
                    slot.insert(konst);
                }
            }
            self.0.deser_op(&root, main, &op, &mut target, &mut mapping).with_context(|| {
                format!("Translating proto-op from Tflite into tract op: {op:#?}")
            })?;
        }
        let outputs: TVec<_> = main
            .outputs()
            .context("No outputs in Tflite model")?
            .iter()
            .map(|o| mapping[&o])
            .collect();
        target.set_output_outlets(&outputs)?;
        Ok(target)
    }
}
