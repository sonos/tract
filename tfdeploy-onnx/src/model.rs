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

#[cfg(test)]
mod tests {
    use std::{fs, path};
    use super::*;
    use pb::TensorProto;

    pub fn load_half_dataset(prefix: &str, path: &path::Path) -> TVec<Tensor> {
        let mut vec = tvec!();
        let len = fs::read_dir(path)
            .unwrap()
            .filter(|d| {
                d.as_ref()
                    .unwrap()
                    .file_name()
                    .to_str()
                    .unwrap()
                    .starts_with(prefix)
            })
            .count();
        for i in 0..len {
            let filename = path.join(format!("{}_{}.pb", prefix, i));
            let mut file = fs::File::open(filename).unwrap();
            let tensor: TensorProto = ::protobuf::parse_from_reader(&mut file).unwrap();
            vec.push(tensor.to_tfd().unwrap())
        }
        vec
    }

    pub fn load_dataset(path: &path::Path) -> (TVec<Tensor>, TVec<Tensor>) {
        (
            load_half_dataset("input", path),
            load_half_dataset("output", path),
        )
    }

    #[test]
    #[ignore]
    fn onnx_abs() {
        let root = path::PathBuf::from("test_abs");
        let model = for_path(root.join("model.onnx")).unwrap();
        let inputs: Vec<&str> = model.guess_inputs().iter().map(|n| &*n.name).collect();
        let outputs: Vec<&str> = model.guess_outputs().iter().map(|n| &*n.name).collect();
        let plan = SimplePlan::new(&model, &*inputs, &*outputs).unwrap();
        for d in fs::read_dir(root).unwrap() {
            let d = d.unwrap();
            if d.metadata().unwrap().is_dir()
                && d.file_name()
                    .to_str()
                    .unwrap()
                    .starts_with("test_data_set_")
            {
                let (inputs, expected) = load_dataset(&d.path());
                let computed = plan.run(inputs).unwrap().remove(0);
                assert_eq!(computed.len(), expected.len());
                computed
                    .iter()
                    .zip(expected.iter())
                    .for_each(|(a, b)| assert!(a.close_enough(b, true)));
            }
        }
    }
}
