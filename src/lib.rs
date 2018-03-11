//! # Tensorflow Deploy
//!
//! Tiny, no-nonsense, self contained, portable Tensorflow inference.
//!
//! ## Example
//!
//! ```
//! # extern crate tfdeploy;
//! # extern crate ndarray;
//! # fn main() {
//! // load a simple model that just add 3 to each input component
//! let graph = tfdeploy::for_path("tests/plus3.pb").unwrap();
//! 
//! // "input" and "output" are tensorflow graph node names.
//! // we need to map these names to ids
//! let input_id = graph.node_id_by_name("input").unwrap();
//! let output_id = graph.node_id_by_name("output").unwrap();
//!
//! // run the computation.
//! let input = ndarray::arr1(&[1.0f32, 2.5, 5.0]);
//! let mut outputs = graph.run(vec![(input_id,input.into())], output_id).unwrap();
//!
//! // grab the first (and only) tensor of the result, and unwrap it as array of f32
//! let output = outputs.remove(0).take_f32s().unwrap();
//! assert_eq!(output, ndarray::arr1(&[4.0, 5.5, 8.0]).into_dyn());
//! # }
//! ```
//!
//! For a more serious example, see [inception v3 example](https://github.com/kali/tensorflow-deploy-rust/blob/master/examples/inceptionv3.rs).

#[macro_use]
extern crate downcast_rs;
#[macro_use]
extern crate error_chain;
extern crate bit_set;
extern crate image;
#[allow(unused_imports)]
#[macro_use]
extern crate log;
#[allow(unused_imports)]
#[macro_use]
extern crate ndarray;
extern crate num_traits;
extern crate protobuf;
#[cfg(test)]
#[macro_use]
#[allow(unused_imports)]
extern crate proptest;
#[cfg(feature = "tensorflow")]
extern crate tensorflow;

pub mod errors;
pub mod tfpb;
pub mod matrix;
pub mod ops;

#[cfg(feature = "tensorflow")]
pub mod tf;

use std::{fs, path, str};
use std::collections::{HashMap, HashSet};
use ops::{ Input, Op };
use errors::*;

pub use matrix::Matrix;

#[derive(Debug)]
pub struct Node {
    pub id: usize,
    pub name: String,
    pub op_name: String,
    inputs: Vec<(usize, Option<usize>)>,
    op: Box<Op>,
}

impl Node {
    pub fn dump_eval_tree(&self, model:&Model) -> String {
        self._dump_eval_tree(model, 0, &mut HashSet::new())
    }

    fn _dump_eval_tree(&self, model:&Model, depth: usize, dups: &mut HashSet<String>) -> String {
        let pad: String = ::std::iter::repeat("  ").take(depth).collect();
        let mut s = format!("{}{}\n", pad, self.name);
        for i in &self.inputs {
            let node = &model.nodes[i.0];
            s.push_str(&*format!("{}", node._dump_eval_tree(&model, depth + 1, dups)));
        }
        s
    }

    pub fn eval_order(&self, model: &Model) -> Result<Vec<usize>> {
        let mut order: Vec<usize> = Vec::new();
        let mut done = bit_set::BitSet::with_capacity(model.nodes.len());
        let mut needed = bit_set::BitSet::with_capacity(model.nodes.len());
        needed.insert(self.id);
        loop {
            let mut done_something = false;
            let mut missing = needed.clone();
            missing.difference_with(&done);
            for node_id in missing.iter() {
                let mut computable = true;
                let node = &model.nodes[node_id];
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
            };
            if !done_something {
                break;
            }
        }
        if done.contains(self.id) {
            Ok(order)
        } else {
            Err(format!("Could not compute node {}", self.name).into())
        }
    }
}

/// Load a Tensorflow protobul model from a file.
pub fn for_path<P: AsRef<path::Path>>(p: P) -> Result<Model> {
    Model::for_path(p)
}

pub struct Plan {
    order: Vec<usize>,
}

impl Plan {
    fn for_node(model: &Model, target: usize) -> Result<Plan> {
        Self::for_nodes(model, &[target])
    }

    fn for_nodes(model: &Model, targets: &[usize]) -> Result<Plan> {
        let mut order: Vec<usize> = Vec::new();
        let mut done = bit_set::BitSet::with_capacity(model.nodes.len());
        let mut needed = bit_set::BitSet::with_capacity(model.nodes.len());
        for &t in targets {
            needed.insert(t);
        }
        loop {
            let mut done_something = false;
            let mut missing = needed.clone();
            missing.difference_with(&done);
            for node_id in missing.iter() {
                let mut computable = true;
                let node = &model.nodes[node_id];
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
            };
            if !done_something {
                break;
            }
        }
        for &t in targets {
            if !done.contains(t) {
                let node = &model.nodes[t];
                Err(format!("Could not plan for node {}", node.name))?
            }
        }
        Ok(Plan { order })
    }

    pub fn run(&self, state: &mut ModelState) -> Result<()> {
        for &n in &self.order {
            if state.outputs[n].is_none() {
                state.compute_one(n)?;
            }
        }
        Ok(())
    }
}

/// Model is Tfdeploy workhouse. It wraps a protobuf tensorflow model,
/// and runs the inference interpreter.
///
pub struct Model {
    nodes: Vec<Node>,
    nodes_by_name: HashMap<String, usize>,
}

impl Model {
    pub fn new(graph: tfpb::graph::GraphDef) -> Result<Model> {
        let mut nodes = vec!();
        let mut nodes_by_name: HashMap<String, usize> = HashMap::new();
        let op_builder = ops::OpBuilder::new();
        for pbnode in graph.get_node().iter() {
            let name = pbnode.get_name().to_string();
            let inputs: Vec<(usize, Option<usize>)> = pbnode
                .get_input()
                .iter()
                .map(|i| {
                    let input: (usize, Option<usize>) = if i.starts_with("^") {
                        (
                            nodes_by_name.get(&*i.replace("^", "")).ok_or(format!(
                                "No node {} found",
                                i
                            ))?.clone(),
                            None,
                        )
                    } else {
                        (
                            nodes_by_name.get(i).ok_or(
                                format!("No node {} found", i),
                            )?.clone(),
                            Some(0usize),
                        )
                    };
                    Ok((input.0.clone(), input.1))
                })
                .collect::<Result<Vec<_>>>()
                .map_err(|e| {
                    format!("While building node {}, {}", name, e.description())
                })?;
            let node = Node {
                id: nodes.len(),
                name: name.to_string(),
                op_name: pbnode.get_op().to_string(),
                inputs: inputs,
                op: op_builder.build(&pbnode).map_err(|e| {
                    format!("While building node {}, {}", name, e.description())
                })?,
            };
            nodes_by_name.insert(name, nodes.len());
            nodes.push(node)
        }
        Ok(Model { nodes, nodes_by_name })
    }

    pub fn node_id_by_name(&self, name:&str) -> Result<usize> {
        self.nodes_by_name.get(name).cloned().ok_or(format!("Node named {} not found", name).into())
    }

    pub fn state(&self) -> ModelState {
        ModelState {
            model: self,
            outputs: vec!(None; self.nodes.len())
        }
    }

    /// Load a Tensorflow protobul model from a file.
    pub fn for_path<P: AsRef<path::Path>>(p: P) -> Result<Model> {
        Self::for_reader(fs::File::open(p)?)
    }

    /// Load a Tensorflow protobul model from a reader.
    pub fn for_reader<R: ::std::io::Read>(mut r: R) -> Result<Model> {
        let loaded = ::protobuf::core::parse_from_reader::<::tfpb::graph::GraphDef>(&mut r)?;
        Model::new(loaded)
    }

    pub fn node_names(&self) -> Vec<&str> {
        self.nodes.iter().map(|s| &*s.name).collect()
    }

    /// Build a tfdeploy Node by name.
    pub fn get_node(&self, name: &str) -> Result<&Node> {
        Ok(&self.nodes[self.node_id_by_name(name)?])
    }

    pub fn plan_for_one(&self, node: usize) -> Result<Plan> {
        Plan::for_node(&self, node)
    }

    pub fn run(&self, inputs: Vec<(usize, Matrix)>, output: usize) -> Result<Vec<Matrix>> {
        self.state().run(inputs, output)
    }
}

pub struct ModelState<'a> {
    model: &'a Model,
    outputs: Vec<Option<Vec<Input>>>,
}

impl<'a> ModelState<'a> {
    /// Reset internal state.
    pub fn reset(&mut self) -> Result<()> {
        self.outputs = vec!(None; self.model.nodes.len());
        Ok(())
    }

    pub fn set_outputs(&mut self, id:usize, values: Vec<Matrix>) -> Result<()> {
        self.outputs[id] = Some(values.into_iter().map(Input::Owned).collect());
        Ok(())
    }

    pub fn set_value(&mut self, id: usize, value: Matrix) -> Result<()> {
        self.set_outputs(id, vec![value])
    }

    fn compute_one(&mut self, node:usize) -> Result<()> {
        let node: &Node = &self.model.nodes[node];
        let mut inputs: Vec<Input> = vec![];
        for i in &node.inputs {
            let prec = self.outputs[i.0].as_ref().ok_or("Unsatisfied node dep")?;
            inputs.push(prec[i.1.ok_or("no output found")?].clone().into())
        }
        let outputs = node.op.eval(inputs)?;
        self.outputs[node.id] = Some(outputs);
        Ok(())
    }

    pub fn take_by_name(&mut self, name: &str) -> Result<Vec<Matrix>> {
        let id = self.model.node_id_by_name(name)?;
        Self::take(self, id)
    }

    pub fn take(&mut self, id: usize) -> Result<Vec<Matrix>> {
        Ok(self.outputs[id].take().ok_or("Value is not computed")?.into_iter().map(Input::into_matrix).collect())
    }

    /// Main entrypoint for running a network.
    ///
    /// Clears the internal state.
    pub fn run(&mut self, inputs: Vec<(usize, Matrix)>, output: usize) -> Result<Vec<Matrix>> {
        self.reset()?;
        for input in inputs {
            self.set_value(input.0, input.1)?;
        }
        Plan::for_node(self.model, output)?.run(self)?;
        Ok(self.take(output)?)
    }
}
