use std::collections::HashMap;
use std::sync::Arc;

use std::{fs, path};

use tfdeploy::model::{Model, Node, OutletId, RawModel};
use tfpb::graph::GraphDef;
use tfdeploy::{TfdFrom, ToTfd, TfdResult };

/// Load a Tensorflow protobul model from a file.
pub fn for_path<P: AsRef<path::Path>>(p: P) -> TfdResult<Model> {
    for_reader(fs::File::open(p)?)
}

/// Load a Tfdeploy model from a reader.
pub fn for_reader<R: ::std::io::Read>(r: R) -> TfdResult<Model> {
    graphdef_for_reader(r)?.to_tfd()
}

/// Load a Tensorflow protobuf graph def from a reader.
pub fn graphdef_for_reader<R: ::std::io::Read>(mut r: R) -> TfdResult<GraphDef> {
    Ok(::protobuf::parse_from_reader::<GraphDef>(
        &mut r,
    ).map_err(|e| format!("{:?}", e))?)
}

/// Load a Tensorflow protobuf graph def from a path
pub fn graphdef_for_path<P: AsRef<path::Path>>(p: P) -> TfdResult<GraphDef> {
    graphdef_for_reader(fs::File::open(p)?)
}

impl TfdFrom<GraphDef> for Model {
    fn tfd_from(graph: &GraphDef) -> TfdResult<Model> {
    let mut nodes = vec![];
    let mut nodes_by_name: HashMap<String, usize> = HashMap::new();
    let op_builder = ::ops::OpBuilder::new();
    for pbnode in graph.get_node().iter() {
        let name = pbnode.get_name().to_string();

        // From the node_def.proto documentation:
        // Each input is "node:src_output" with "node" being a string name and
        // "src_output" indicating which output tensor to use from "node". If
        // "src_output" is 0 the ":0" suffix can be omitted. Regular inputs may
        // optionally be followed by control inputs that have the format "^node".
        let inputs: Vec<OutletId> = pbnode
            .get_input()
            .iter()
            .map(|i| {
                let input: (usize, usize) = if i.starts_with("^") {
                    (
                        nodes_by_name
                            .get(&*i.replace("^", ""))
                            .ok_or(format!("No node {} found", i))?
                            .clone(),
                        0,
                    )
                } else {
                    let splits: Vec<_> = i.splitn(2, ':').collect();
                    (
                        nodes_by_name
                            .get(splits[0])
                            .ok_or(format!("No node {} found", i))?
                            .clone(),
                        if splits.len() > 1 {
                            splits[1].parse::<usize>()?
                        } else {
                            0
                        },
                    )
                };
                Ok(OutletId::new(input.0, input.1))
            })
            .collect::<TfdResult<Vec<_>>>()
            .map_err(|e| format!("While building node {}, {}", name, e.description()))?;
        let node = Node {
            id: nodes.len(),
            name: name.to_string(),
            op_name: pbnode.get_op().to_string(),
            inputs: inputs,
            op: op_builder
                .build(&pbnode)
                .map_err(|e| format!("While building node {}, {}", name, e.description()))?,
        };
        nodes_by_name.insert(name, nodes.len());
        nodes.push(node)
    }

    Ok(Model(Arc::new(RawModel::new(nodes, nodes_by_name))))
}
}
