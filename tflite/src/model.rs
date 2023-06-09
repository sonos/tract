use tract_hir::internal::*;

use crate::registry::Registry;
use crate::tflite;

pub struct Tflite(Registry);

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
        //        tflite::model::Model::from_buffer(&self.0).context("Failed to read flat buffer model")
    }
}

impl Framework<TfliteProtoModel, TypedModel> for Tflite {
    fn proto_model_for_read(
        &self,
        reader: &mut dyn std::io::Read,
    ) -> tract_hir::prelude::TractResult<TfliteProtoModel> {
        let mut buf = vec![];
        reader.read_to_end(&mut buf)?;
        TfliteProtoModel::new(buf)
    }

    fn model_for_proto_model_with_symbols(
        &self,
        proto: &TfliteProtoModel,
        _symbols: &SymbolTable,
    ) -> TractResult<TypedModel> {
        let root = proto.root();
        let main = &root.subgraphs().context("No subgraphs in Tflite model")?.get(0);
        let tensors = main.tensors().unwrap();
        let mut target = TypedModel::default();
        let mut mapping = HashMap::new();
        for input in main.inputs().unwrap() {
            let (fact, name) = crate::tensors::tensor_to_fact(&root, main, input)?;
            let it = target.add_source(name, fact)?;
            mapping.insert(input, it);
        }
        for op in main.operators().unwrap() {
            for input in op.inputs().unwrap() {
                if !mapping.contains_key(&input) {
                    let (fact, name) = crate::tensors::tensor_to_fact(&root, main, input)?;
                    let konst = target.add_const(name, fact.konst.unwrap())?;
                    mapping.insert(input, konst);
                }
            }
            self.0.op(&root, main, &op, &mut target, &mut mapping)?;
        }
        for output in main.outputs().unwrap() {
            dbg!(output);
            dbg!(tensors.get(output as _));
        }
        dbg!(&target);
        Ok(target)
    }
}
