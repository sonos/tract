use std::collections::HashMap;
use std::{fs, path};

use tract_core::model::{InletId, Model, OutletId};
use tract_core::*;

use pb;

/// Load a ONNX protobul model from a file.
pub fn for_path<P: AsRef<path::Path>>(p: P) -> TfdResult<Model> {
    for_reader(fs::File::open(p)?)
}

/// Load a ONNX model from a reader.
pub fn for_reader<R: ::std::io::Read>(r: R) -> TfdResult<Model> {
    model_proto_for_reader(r)?.tractify()
}

/// Load a ONNX protobuf graph def from a path
pub fn model_proto_for_path<P: AsRef<path::Path>>(p: P) -> TfdResult<pb::ModelProto> {
    model_proto_for_reader(fs::File::open(p)?)
}

/// Load a ONNX protobuf graph def from a reader.
pub fn model_proto_for_reader<R: ::std::io::Read>(mut r: R) -> TfdResult<pb::ModelProto> {
    Ok(::protobuf::parse_from_reader(&mut r).map_err(|e| format!("{:?}", e))?)
}

impl Tractify<pb::ModelProto> for Model {
    fn tractify(proto: &pb::ModelProto) -> TfdResult<Model> {
        let mut model = Model::default();
        let op_builder = super::ops::OpBuilder::new();
        let graph = proto.get_graph();
        let mut initializers: HashMap<&str, Tensor> = graph
            .get_initializer()
            .iter()
            .map(|init| Ok((init.get_name(), init.tractify()?)))
            .collect::<TfdResult<_>>()?;
        let mut outlets_by_name = HashMap::<String, OutletId>::new();
        for input in graph.get_input().iter() {
            if let Some(init) = initializers.remove(input.get_name()) {
                let id = model.add_node(
                    input.get_name().to_owned(),
                    Box::new(::tract_core::ops::konst::Const::new(init.into())),
                )?;
                outlets_by_name.insert(input.get_name().to_owned(), OutletId::new(id, 0));
            } else {
                let fact = input.get_field_type().get_tensor_type().tractify()?;
                let id = model.add_node(
                    input.get_name().to_owned(),
                    Box::new(::tract_core::ops::source::Source::new(fact)),
                )?;
                outlets_by_name.insert(input.get_name().to_owned(), OutletId::new(id, 0));
            }
        }
        for pbnode in graph.get_node().iter() {
            let name = if pbnode.get_name() != "" {
                pbnode.get_name().to_string()
            } else if pbnode.get_output().len() > 0 && pbnode.get_output()[0] != "" {
                pbnode.get_output()[0].to_owned()
            } else {
                format!("{}-{}", model.nodes().len(), pbnode.get_op_type())
            };
            let id = model.add_node(name, op_builder.build(pbnode)?)?;
            for (ix, output) in pbnode.get_output().iter().enumerate() {
                outlets_by_name.insert(output.to_owned(), OutletId::new(id, ix));
            }
            for (ix, input) in pbnode.get_input().iter().enumerate() {
                model.add_edge(outlets_by_name[&*input], InletId::new(id, ix))?;
            }
        }
        let mut outputs = vec![];
        for output in graph.get_output().iter() {
            let fact = output.get_field_type().get_tensor_type().tractify()?;
            outputs.push(outlets_by_name[output.get_name()]);
            model.set_fact(outlets_by_name[output.get_name()], fact)?;
        }
        model.set_outputs_outlets(&outputs)?;
        Ok(model)
    }
}
