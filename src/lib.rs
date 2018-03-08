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
//! // run the computation. "input" and "output" are tensorflow graph node names.
//! let input = ndarray::arr1(&[1.0f32, 2.5, 5.0]);
//! let mut outputs = graph.run(vec![("input",input.into())], "output").unwrap();
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

use std::{fs, path, rc, str};
use std::collections::{HashMap, HashSet};
use ops::Op;
use errors::*;

pub use matrix::Matrix;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Node(pub std::sync::Arc<RawNode>);

impl ::std::ops::Deref for Node {
    type Target = RawNode;
    fn deref(&self) -> &RawNode {
        &*self.0
    }
}

#[derive(Debug)]
pub struct RawNode {
    pub name: String,
    pub op_name: String,
    inputs: Vec<(Node, Option<usize>)>,
    op: Box<Op>,
}

impl ::std::hash::Hash for RawNode {
    fn hash<H: ::std::hash::Hasher>(&self, h: &mut H) {
        self.name.hash(h)
    }
}
impl PartialEq for RawNode {
    fn eq(&self, other: &RawNode) -> bool {
        self.name.eq(&other.name)
    }
}
impl Eq for RawNode {}

impl Node {
    pub fn dump_eval_tree(&self) -> String {
        self._dump_eval_tree(0, &mut HashSet::new())
    }

    fn _dump_eval_tree(&self, depth: usize, dups: &mut HashSet<String>) -> String {
        let pad: String = ::std::iter::repeat("  ").take(depth).collect();
        let mut s = format!("{}{}\n", pad, self.name);
        for i in &self.inputs {
            s.push_str(&*format!("{}", i.0._dump_eval_tree(depth + 1, dups)));
        }
        s
    }

    pub fn eval_order(&self) -> Result<Vec<Node>> {
        let mut order: Vec<Node> = Vec::new();
        let mut done: HashSet<Node> = HashSet::new();
        let mut needed: HashSet<Node> = HashSet::new();
        let mut more: HashSet<Node> = HashSet::new();
        needed.insert(self.clone());
        loop {
            let mut done_something = false;
            more.clear();
            needed.retain(|node| {
                let mut computable = true;
                for i in node.inputs.iter() {
                    if !done.contains(&i.0) {
                        computable = false;
                        done_something = true;
                        more.insert(i.0.clone());
                    }
                }
                if computable {
                    done_something = true;
                    order.push(node.clone());
                    done.insert(node.clone());
                    false
                } else {
                    true
                }
            });
            needed.extend(more.iter().cloned());
            if !done_something {
                break;
            }
        }
        if done.contains(self) {
            Ok(order)
        } else {
            Err(format!("Could not compute node {}", self.name).into())
        }
    }

    fn _eval_order<'a: 'b, 'b: 'c, 'c>(
        &'a self,
        entered: &'b mut HashSet<Node>,
        done: &'b mut HashMap<Node, Vec<Node>>,
    ) -> Result<&'c Vec<Node>> {
        let mut result: Vec<Node> = vec![];
        if !done.contains_key(&self) {
            entered.insert(self.clone());
            for i in &self.inputs {
                if entered.contains(&i.0) {
                    Err(format!(
                        "Loop detected {} - {}! Need more code!",
                        self.name,
                        i.0.name,
                    ))?
                }
                let v = i.0._eval_order(entered, done)?;
                result.extend(v.into_iter().cloned());
            }
            result.push(self.clone());
            entered.remove(&self);
            println!("Eval order done for {}", self.name);
            done.insert(self.clone(), result);
        }
        Ok(done.get(self).unwrap())
    }
}

/// Load a Tensorflow protobul model from a file.
pub fn for_path<P: AsRef<path::Path>>(p: P) -> Result<Model> {
    Model::for_path(p)
}

/// Model is Tfdeploy workhouse. It wraps a protobuf tensorflow model,
/// and runs the inference interpreter.
///
pub struct Model {
    nodes: HashMap<String, Node>,
}

impl Model {
    pub fn new(graph: tfpb::graph::GraphDef) -> Result<Model> {
        let mut nodes: HashMap<String, Node> = HashMap::new();
        let op_builder = ops::OpBuilder::new();
        for pbnode in graph.get_node().iter() {
            let name = pbnode.get_name().to_string();
            let inputs: Vec<(Node, Option<usize>)> = pbnode
                .get_input()
                .iter()
                .map(|i| {
                    let input: (&Node, Option<usize>) = if i.starts_with("^") {
                        (
                            nodes.get(&*i.replace("^", "")).ok_or(format!(
                                "No node {} found",
                                i
                            ))?,
                            None,
                        )
                    } else {
                        (
                            nodes.get(&*i.to_string()).ok_or(
                                format!("No node {} found", i),
                            )?,
                            Some(0usize),
                        )
                    };
                    Ok((input.0.clone(), input.1))
                })
                .collect::<Result<Vec<_>>>()
                .map_err(|e| {
                    format!("While building node {}, {}", name, e.description())
                })?;
            let node = Node(std::sync::Arc::new(RawNode {
                name: name.to_string(),
                op_name: pbnode.get_op().to_string(),
                inputs: inputs,
                op: op_builder.build(&pbnode).map_err(|e| {
                    format!("While building node {}, {}", name, e.description())
                })?,
            }));
            nodes.insert(name, node);
        }
        Ok(Model { nodes })
    }

    pub fn state(&self) -> ModelState {
        ModelState {
            model: self,
            outputs: HashMap::new(),
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
        self.nodes.keys().map(|s| &**s).collect()
    }

    /// Build a tfdeploy Node by name.
    pub fn get_node(&self, name: &str) -> Result<&Node> {
        Ok(self.nodes.get(name).ok_or(
            format!("node {} do not exist", name),
        )?)
    }

    pub fn run(&self, inputs: Vec<(&str, Matrix)>, output_name: &str) -> Result<Vec<Matrix>> {
        self.state().run(inputs, output_name)
    }
}

pub struct ModelState<'a> {
    model: &'a Model,
    outputs: HashMap<String, Vec<Matrix>>,
}

impl<'a> ModelState<'a> {
    /// Reset internal state.
    pub fn reset(&mut self) -> Result<()> {
        self.outputs.clear();
        Ok(())
    }

    pub fn set_outputs(&mut self, name: &str, values: Vec<Matrix>) -> Result<()> {
        self.outputs.insert(name.to_string(), values);
        Ok(())
    }

    pub fn set_value(&mut self, name: &str, value: Matrix) -> Result<()> {
        self.set_outputs(name, vec![value])
    }

    fn compute_node(&mut self, node:Node) -> Result<()> {
        let mut inputs: Vec<Matrix> = vec![];
        for i in &node.inputs {
            inputs.push(self.outputs.get(&i.0.name).ok_or("Unsatisfied node dep")?[i.1.ok_or("no output found")?].clone())
        }
        let outputs = node.op.eval(inputs)?;
        self.outputs.insert(node.name.to_string(), outputs);
        Ok(())
    }

    fn compute(&mut self, name: &str) -> Result<()> {
        let node: &Node = self.model.get_node(name)?;
        for dep in node.eval_order()? {
            if !self.outputs.contains_key(&dep.name) {
                self.compute_node(dep)?;
            }
        }
        Ok(())
    }

    /// Trigger evaluation of the specified node and return the cache value.
    pub fn eval(&mut self, name: &str) -> Result<&Vec<Matrix>> {
        self.compute(name)?;
        Ok(self.outputs.get(name).expect(
            "node found but was not computed",
        ))
    }

    /// Trigger evaluation of the specified node and consume the value from the
    /// cache.
    pub fn take(&mut self, name: &str) -> Result<Vec<Matrix>> {
        self.compute(name)?;
        Ok(self.outputs.remove(name).ok_or(
            format!("{} does not exits", name),
        )?)
    }

    /// Main entrypoint for running a network.
    ///
    /// Clears the internal state.
    pub fn run(&mut self, inputs: Vec<(&str, Matrix)>, output_name: &str) -> Result<Vec<Matrix>> {
        self.reset()?;
        for input in inputs {
            self.set_value(input.0, input.1)?;
        }
        Ok(self.take(output_name)?)
    }
}
