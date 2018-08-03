use std::{fs, path, str};
use std::sync::Arc;
use std::collections::HashMap;
use std::ops::Deref;

use bit_set;

use {ops, tfpb, Result};

#[cfg_attr(feature = "serialize", derive(Serialize))]
#[derive(Debug, Clone)]
pub struct Node {
    pub id: usize,
    pub name: String,
    pub op_name: String,
    pub inputs: Vec<(usize, usize)>,
    pub op: Box<ops::Op>,
}

impl Node {
    pub fn op(&self) -> &ops::Op {
        &*self.op
    }
}

/// Model is Tfdeploy workhouse. It wraps a protobuf tensorflow model,
/// and runs the inference interpreter.
#[derive(Clone,Debug)]
pub struct RawModel {
    pub nodes: Vec<Node>,
    pub nodes_by_name: HashMap<String, usize>,
}

impl RawModel {
    pub fn new(graph: tfpb::graph::GraphDef) -> Result<Model> {
        let mut nodes = vec![];
        let mut nodes_by_name: HashMap<String, usize> = HashMap::new();
        let op_builder = ops::OpBuilder::new();
        for pbnode in graph.get_node().iter() {
            let name = pbnode.get_name().to_string();

            // From the node_def.proto documentation:
            // Each input is "node:src_output" with "node" being a string name and
            // "src_output" indicating which output tensor to use from "node". If
            // "src_output" is 0 the ":0" suffix can be omitted. Regular inputs may
            // optionally be followed by control inputs that have the format "^node".
            let inputs: Vec<(usize, usize)> = pbnode
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
                    Ok((input.0, input.1))
                })
                .collect::<Result<Vec<_>>>()
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
        Ok(Model(Arc::new(RawModel {
            nodes,
            nodes_by_name,
        })))
    }

    pub fn node_by_name(&self, name: &str) -> Result<&Node> {
        let id:&usize = self.nodes_by_name
            .get(name)
            .ok_or_else(|| format!("Node named {} not found", name))?;
        Ok(&self.nodes[*id])
    }

    pub fn node_names(&self) -> Vec<&str> {
        self.nodes.iter().map(|s| &*s.name).collect()
    }

    pub fn nodes(&self) -> &[Node] {
        &*self.nodes
    }
}

#[derive(Debug, Clone)]
pub struct Model(pub Arc<RawModel>);

impl Model {
    /// Load a Tensorflow protobul model from a file.
    pub fn for_path<P: AsRef<path::Path>>(p: P) -> Result<Model> {
        Self::for_reader(fs::File::open(p)?)
    }

    /// Load a Tfdeploy model from a reader.
    pub fn for_reader<R: ::std::io::Read>(r: R) -> Result<Model> {
        RawModel::new(Model::graphdef_for_reader(r)?)
    }

    /// Load a Tensorflow protobuf graph def from a reader.
    pub fn graphdef_for_reader<R: ::std::io::Read>(mut r: R) -> Result<::tfpb::graph::GraphDef> {
        Ok(::protobuf::parse_from_reader::<::tfpb::graph::GraphDef>(
            &mut r,
        )?)
    }

    /// Load a Tensorflow protobuf graph def from a path
    pub fn graphdef_for_path<P: AsRef<path::Path>>(p: P) -> Result<::tfpb::graph::GraphDef> {
        Self::graphdef_for_reader(fs::File::open(p)?)
    }

    pub fn analyser(&self, output: &str) -> Result<::analyser::Analyser> {
        ::analyser::Analyser::new(&self, output)
    }
}

impl Deref for Model {
    type Target = RawModel;
    fn deref(&self) -> &RawModel {
        &*self.0
    }
}

pub fn eval_order_for_nodes(nodes: &[Node], targets: &[usize]) -> Result<Vec<usize>> {
    let mut order: Vec<usize> = Vec::new();
    let mut done = bit_set::BitSet::with_capacity(nodes.len());
    let mut needed = bit_set::BitSet::with_capacity(nodes.len());
    for &t in targets {
        needed.insert(t);
    }
    loop {
        let mut done_something = false;
        let mut missing = needed.clone();
        missing.difference_with(&done);
        for node_id in missing.iter() {
            let mut computable = true;
            let node = &nodes[node_id];
            for i in node.inputs.iter() {
                if !done.contains(i.0) {
                    computable = false;
                    done_something = true;
                    needed.insert(i.0.clone());
                }
            }
            if computable {
                done_something = true;
                order.push(node_id);
                done.insert(node_id);
            }
        }
        if !done_something {
            break;
        }
    }
    for &t in targets {
        if !done.contains(t) {
            let node = &nodes[t];
            Err(format!("Could not plan for node {}", node.name))?
        }
    }
    Ok(order)
}

