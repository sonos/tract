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
//! let graph = tfdeploy::for_path("tests/models/plus3.pb").unwrap();
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

// TODO: show Plan-based API in doc instead of shortcut

extern crate bit_set;
#[cfg(feature = "blis")]
extern crate blis_src;
#[macro_use]
extern crate derive_new;
#[macro_use]
extern crate error_chain;
#[cfg(feature = "image_ops")]
extern crate image;
extern crate itertools;
#[allow(unused_imports)]
#[macro_use]
extern crate log;
#[allow(unused_imports)]
#[macro_use]
extern crate ndarray;
extern crate num_traits;
extern crate protobuf;
#[macro_use]
extern crate maplit;
#[macro_use]
extern crate objekt;

#[cfg(feature = "serialize")]
extern crate serde;
#[cfg(test)]
extern crate simplelog;
#[cfg(feature = "serialize")]
#[macro_use]
extern crate serde_derive;
#[macro_use]
extern crate downcast_rs;

#[macro_use]
pub mod analyser;
pub mod errors;
pub mod model;
pub mod ops;
pub mod streaming;
pub mod tensor;
pub mod tfpb;

use std::collections::{HashMap, HashSet, VecDeque};
use std::{path, str};

// use analyser::prelude::*;
use analyser::helpers::tensor_to_fact;
pub use errors::*;
use ops::{Op, OpBuffer, TensorView};
pub use tensor::{DataType, Tensor};
pub use model::Model;

#[cfg_attr(feature = "serialize", derive(Serialize))]
#[derive(Debug, Clone)]
pub struct Node {
    pub id: usize,
    pub name: String,
    pub op_name: String,
    pub inputs: Vec<(usize, Option<usize>)>,
    pub op: Box<Op>,
}

impl Node {
    pub fn dump_eval_tree(&self, model: &Model) -> String {
        self._dump_eval_tree(model, 0, &mut HashSet::new())
    }

    fn _dump_eval_tree(&self, model: &Model, depth: usize, dups: &mut HashSet<String>) -> String {
        let pad: String = ::std::iter::repeat("  ").take(depth).collect();
        let mut s = format!("{}{}\n", pad, self.name);
        for i in &self.inputs {
            let node = &model.nodes[i.0];
            s.push_str(&*format!(
                "{}",
                node._dump_eval_tree(&model, depth + 1, dups)
            ));
        }
        s
    }

    pub fn eval_order(&self, model: &Model) -> Result<Vec<usize>> {
        Ok(Plan::for_model(model, &[self.id])?.order)
    }

    pub fn op(&self) -> &Op {
        &*self.op
    }
}

/// Load a Tensorflow protobul model from a file.
pub fn for_path<P: AsRef<path::Path>>(p: P) -> Result<Model> {
    Ok(Model::for_path(p)?)
}

#[derive(Debug)]
pub struct Plan {
    pub order: Vec<usize>,
}

impl Plan {
    pub fn for_model(model: &Model, targets: &[usize]) -> Result<Plan> {
        Self::for_nodes(&model.nodes, targets)
    }

    fn for_nodes(nodes: &Vec<Node>, targets: &[usize]) -> Result<Plan> {
        Self::for_node_provider(nodes.len(), &|i|&nodes[i], targets)
    }

    fn for_node_provider<'a, N>(nodes_count:usize, get_node: &'a N, targets: &[usize]) -> Result<Plan> 
        where N: Fn(usize) -> &'a Node
    {
        let mut order: Vec<usize> = Vec::new();
        let mut done = bit_set::BitSet::with_capacity(nodes_count);
        let mut needed = bit_set::BitSet::with_capacity(nodes_count);
        for &t in targets {
            needed.insert(t);
        }
        loop {
            let mut done_something = false;
            let mut missing = needed.clone();
            missing.difference_with(&done);
            for node_id in missing.iter() {
                let mut computable = true;
                let node = get_node(node_id);
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
                let node = get_node(t);
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

#[derive(Clone)]
pub struct ModelState {
    model: Model,
    pub outputs: Vec<Option<Vec<TensorView>>>,
}

impl ModelState {
    /// Reset internal state.
    pub fn reset(&mut self) -> Result<()> {
        self.outputs = vec![None; self.model.nodes.len()];
        Ok(())
    }

    pub fn set_outputs(&mut self, id: usize, values: Vec<Tensor>) -> Result<()> {
        self.outputs[id] = Some(values.into_iter().map(TensorView::Owned).collect());
        Ok(())
    }

    pub fn set_value(&mut self, id: usize, value: Tensor) -> Result<()> {
        self.set_outputs(id, vec![value])
    }

    pub fn set_values(&mut self, values: Vec<(&str, Tensor)>) -> Result<()> {
        for (name, mat) in values {
            let id = self.model.node_id_by_name(name)?;
            self.set_value(id, mat)?;
        }

        Ok(())
    }

    pub fn compute_one(&mut self, node: usize) -> Result<()> {
        let node: &Node = &self.model.nodes[node];
        let mut inputs: Vec<TensorView> = vec![];
        for i in &node.inputs {
            let prec_node = &self.model.nodes[i.0];
            let prec = self.outputs[i.0].as_ref().ok_or(format!(
                "Computing {}, precursor {} not done:",
                node.name, prec_node.name
            ))?;
            inputs.push(prec[i.1.unwrap_or(0)].clone().into())
        }
        let outputs = node.op.eval(inputs)?;
        self.outputs[node.id] = Some(outputs);
        Ok(())
    }

    pub fn take_by_name(&mut self, name: &str) -> Result<Vec<Tensor>> {
        let id = self.model.node_id_by_name(name)?;
        Self::take(self, id)
    }

    pub fn take(&mut self, id: usize) -> Result<Vec<Tensor>> {
        Ok(self.outputs[id]
            .take()
            .ok_or("Value is not computed")?
            .into_iter()
            .map(TensorView::into_tensor)
            .collect())
    }

    /// Main entrypoint for running a network.
    ///
    /// Clears the internal state.
    pub fn run(&mut self, inputs: Vec<(usize, Tensor)>, output: usize) -> Result<Vec<Tensor>> {
        self.reset()?;
        for input in inputs {
            self.set_value(input.0, input.1)?;
        }
        Plan::for_model(&self.model, &[output])?.run(self)?;
        Ok(self.take(output)?)
    }

    pub fn model(&self) -> &Model {
        &self.model
    }
}

#[cfg(test)]
#[allow(dead_code)]
fn setup_test_logger() {
    use simplelog::{Config, LevelFilter, TermLogger};
    TermLogger::init(LevelFilter::Trace, Config::default()).unwrap()
}
