use std::{fs, path, str};
use std::sync::Arc;
use std::collections::HashMap;

use {ops, tfpb, Result, ModelState, Node, Plan, Tensor };
use std::ops::Deref;

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
            let inputs: Vec<(usize, Option<usize>)> = pbnode
                .get_input()
                .iter()
                .map(|i| {
                    let input: (usize, Option<usize>) = if i.starts_with("^") {
                        (
                            nodes_by_name
                                .get(&*i.replace("^", ""))
                                .ok_or(format!("No node {} found", i))?
                                .clone(),
                            None,
                        )
                    } else {
                        let splits: Vec<_> = i.splitn(2, ':').collect();
                        (
                            nodes_by_name
                                .get(splits[0])
                                .ok_or(format!("No node {} found", i))?
                                .clone(),
                            if splits.len() > 1 {
                                Some(splits[1].parse::<usize>()?)
                            } else {
                                Some(0)
                            },
                        )
                    };
                    Ok((input.0.clone(), input.1))
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

    pub fn node_id_by_name(&self, name: &str) -> Result<usize> {
        self.nodes_by_name
            .get(name)
            .cloned()
            .ok_or(format!("Node named {} not found", name).into())
    }

    pub fn node_names(&self) -> Vec<&str> {
        self.nodes.iter().map(|s| &*s.name).collect()
    }

    /// Get a tfdeploy Node by name.
    pub fn get_node(&self, name: &str) -> Result<&Node> {
        Ok(&self.nodes[self.node_id_by_name(name)?])
    }

    /// Get a tfdeploy Node by id.
    pub fn get_node_by_id(&self, id: usize) -> Result<&Node> {
        if id >= self.nodes.len() {
            Err(format!("Invalid node id {}", id))?
        } else {
            Ok(&self.nodes[id])
        }
    }

    pub fn nodes(&self) -> &[Node] {
        &*self.nodes
    }
}

#[derive(Debug, Clone)]
pub struct Model(pub Arc<RawModel>);

impl Model {
    pub fn state(&self) -> ModelState {
        ModelState {
            model: self.clone(),
            outputs: vec![None; self.nodes.len()],
        }
    }

    pub fn run(&self, inputs: Vec<(usize, Tensor)>, output: usize) -> Result<Vec<Tensor>> {
        self.state().run(inputs, output)
    }


    pub fn run_with_names(&self, inputs: Vec<(&str, Tensor)>, output: &str) -> Result<Vec<Tensor>> {
        let inputs = inputs
            .into_iter()
            .map(|(name, mat)| -> Result<(usize, Tensor)> {
                Ok((self.node_id_by_name(name)?, mat))
            })
            .collect::<Result<_>>()?;
        self.run(inputs, self.node_id_by_name(output)?)
    }

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

    pub fn plan_for_one(&self, node: usize) -> Result<Plan> {
        Plan::for_model(&self, &[node])
    }

    pub fn analyser(&self, output: usize) -> Result<::analyser::Analyser> {
        ::analyser::Analyser::new(&self, output)
    }
}

impl Deref for Model {
    type Target = RawModel;
    fn deref(&self) -> &RawModel {
        &*self.0
    }
}

