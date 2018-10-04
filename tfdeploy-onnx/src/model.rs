use std::collections::HashMap;
use std::sync::Arc;
use std::{fs, path};

use tfdeploy::model::{Model, Node, OutletId, RawModel};
use tfdeploy::*;

use pb;

/// Load a ONNX protobul model from a file.
pub fn for_path<P: AsRef<path::Path>>(p: P) -> TfdResult<Model> {
    for_reader(fs::File::open(p)?)
}

/// Load a ONNX model from a reader.
pub fn for_reader<R: ::std::io::Read>(r: R) -> TfdResult<Model> {
    model_proto_for_reader(r)?.to_tfd()
}

/// Load a ONNX protobuf graph def from a path
pub fn model_proto_for_path<P: AsRef<path::Path>>(p: P) -> TfdResult<pb::ModelProto> {
    model_proto_for_reader(fs::File::open(p)?)
}

/// Load a ONNX protobuf graph def from a reader.
pub fn model_proto_for_reader<R: ::std::io::Read>(mut r: R) -> TfdResult<pb::ModelProto> {
    Ok(::protobuf::parse_from_reader(&mut r).map_err(|e| format!("{:?}", e))?)
}

impl TfdFrom<pb::ModelProto> for Model {
    fn tfd_from(proto: &pb::ModelProto) -> TfdResult<Model> {
        let mut nodes = vec![];
        let mut model_inputs = vec![];
        let mut outlets_index: HashMap<String, OutletId> = HashMap::new();
        let mut nodes_by_name: HashMap<String, usize> = HashMap::new();
        let op_builder = super::ops::OpBuilder::new();
        let graph = proto.get_graph();
        for input in graph.get_input().iter() {
            outlets_index.insert(input.get_name().to_owned(), OutletId::new(nodes.len(), 0));
            let fact = input.get_field_type().get_tensor_type().to_tfd()?;
            let source = Node {
                id: nodes.len(),
                name: input.get_name().to_owned(),
                op: Box::new(::tfdeploy::ops::source::Source::new(fact)),
                op_name: "Source".to_string(),
                inputs: vec![],
            };
            nodes_by_name.insert(input.get_name().to_owned(), nodes.len());
            model_inputs.push(nodes.len());
            nodes.push(source);
        }
        for input in graph.get_initializer() {
            let node = nodes_by_name[input.get_name()];
            let node = &mut nodes[node];
            let tensor:Tensor = input.to_tfd()?;
            if node.op_name == "Source" {
                node.op_name = "Const".to_string();
                node.op = Box::new(::tfdeploy::ops::konst::Const::new(tensor.into()));
            }
        }
        for pbnode in graph.get_node().iter() {
            let name = if pbnode.get_name() != "" {
                pbnode.get_name().to_string()
            } else if pbnode.get_output().len() > 0 && pbnode.get_output()[0] != "" {
                pbnode.get_output()[0].to_owned()
            } else {
                format!("{}-{}", nodes.len(), pbnode.get_op_type())
            };
            for (ix, output) in pbnode.get_output().iter().enumerate() {
                outlets_index.insert(output.to_string(), OutletId::new(nodes.len(), ix));
            }
            let op_name = pbnode.get_op_type().to_owned();
            let node = Node {
                id: nodes.len(),
                name: name.clone(),
                op: op_builder.build(pbnode)?,
                op_name,
                inputs: vec![],
            };
            nodes_by_name.insert(name, nodes.len());
            nodes.push(node)
        }
        for (pbnode, mut node) in graph
            .get_node()
            .iter()
            .zip(&mut nodes.iter_mut().skip(graph.get_input().len()))
        {
            for pbinput in pbnode.get_input() {
                node.inputs.push(
                    outlets_index
                        .get(pbinput)
                        .ok_or_else(|| format!("Can not find matching outlet for {}", pbinput))?
                        .clone(),
                )
            }
        }
        for output in graph.get_output().iter() {
            let fact = output.get_field_type().get_tensor_type().to_tfd()?;
            let outlet = outlets_index[output.get_name()];
            let source = Node {
                id: nodes.len(),
                name: format!("Sink-{}", output.get_name()),
                op: Box::new(::tfdeploy::ops::sink::Sink::new(fact)),
                op_name: "Sink".to_string(),
                inputs: vec![outlet],
            };
            nodes_by_name.insert(format!("Output-{}", output.get_name()), nodes.len());
            nodes.push(source);
        }
        Ok(Model(Arc::new(RawModel::new(nodes, nodes_by_name))))
    }
}
