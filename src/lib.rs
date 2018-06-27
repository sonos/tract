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
#[cfg(feature = "serialize")]
#[macro_use]
extern crate serde_derive;

#[macro_use]
pub mod analyser;
pub mod errors;
pub mod ops;
pub mod tensor;
pub mod tfpb;

use std::collections::{HashMap, HashSet, VecDeque};
use std::{fs, path, str};

use analyser::helpers::tensor_to_fact;
use analyser::Analyser;
pub use errors::*;
use ops::{Op, TensorView, Buffer};
pub use tensor::Tensor;

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
}

/// Load a Tensorflow protobul model from a file.
pub fn for_path<P: AsRef<path::Path>>(p: P) -> Result<Model> {
    Model::for_path(p)
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
#[derive(Clone)]
pub struct Model {
    pub nodes: Vec<Node>,
    pub nodes_by_name: HashMap<String, usize>,
}

impl Model {
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
                            }
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
        Ok(Model {
            nodes,
            nodes_by_name,
        })
    }

    pub fn node_id_by_name(&self, name: &str) -> Result<usize> {
        self.nodes_by_name
            .get(name)
            .cloned()
            .ok_or(format!("Node named {} not found", name).into())
    }

    pub fn state(&self) -> ModelState {
        ModelState {
            model: self,
            outputs: vec![None; self.nodes.len()],
        }
    }

    /// Load a Tensorflow protobul model from a file.
    pub fn for_path<P: AsRef<path::Path>>(p: P) -> Result<Model> {
        Self::for_reader(fs::File::open(p)?)
    }

    /// Load a Tfdeploy model from a reader.
    pub fn for_reader<R: ::std::io::Read>(r: R) -> Result<Model> {
        Model::new(Self::graphdef_for_reader(r)?)
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

    pub fn plan_for_one(&self, node: usize) -> Result<Plan> {
        Plan::for_model(&self, &[node])
    }

    pub fn run(&self, inputs: Vec<(usize, Tensor)>, output: usize) -> Result<Vec<Tensor>> {
        self.state().run(inputs, output)
    }

    pub fn nodes(&self) -> &[Node] {
        &*self.nodes
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
}

pub struct ModelState<'a> {
    model: &'a Model,
    pub outputs: Vec<Option<Vec<TensorView>>>,
}

impl<'a> ModelState<'a> {
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
            self.set_value(self.model.node_id_by_name(name)?, mat)?;
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
            inputs.push(prec[i.1.ok_or("no output found")?].clone().into())
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
        Plan::for_model(self.model, &[output])?.run(self)?;
        Ok(self.take(output)?)
    }

    pub fn model(&self) -> &Model {
        self.model
    }
}

/// The state of a model during streaming evaluation.
#[derive(Clone)]
pub struct StreamingState {
    model: Model,
    output: usize,
    mapping: Vec<Option<usize>>,
    buffers: Vec<Buffer>,
    dimensions: HashMap<(usize, usize), usize>,
    successors: Vec<Vec<(usize, usize)>>,
}

/// The type of an input during streaming evaluation.
#[derive(Debug, Clone)]
pub enum StreamingInput {
    // The input is being streamed along some dimension.
    Streamed(usize),

    // The input will remain constant during the evaluation.
    Constant(Tensor),
}

impl StreamingState {
    /// Initializes the streaming evaluation of a model.
    ///
    /// For each input in the model, you must either provide a constant
    /// value or specify the dimension along which to stream.
    ///
    /// You will only be able to fetch the results of the evaluation step
    /// for the output node. If `output` is None, the output node will be
    /// guessed automatically.
    pub fn start(
        model: Model,
        inputs: Vec<(usize, StreamingInput)>,
        output: Option<usize>,
    ) -> Result<StreamingState> {
        use StreamingInput::*;

        let output = output
            .or(analyser::detect_output(&model)?)
            .ok_or("Unable to auto-detect output node.")?;

        let mut analyser = Analyser::new(model, output)?;

        // Pre-compute the constant part of the graph using the analyser.
        let mut streaming_inputs = vec![];
        for input in inputs {
            match input {
                (i, Streamed(d)) => streaming_inputs.push((i, d)),
                (i, Constant(tensor)) => analyser.hint(i, &tensor_to_fact(tensor))?,
            }
        }

        analyser.run()?;
        analyser.propagate_constants()?;

        // Keep track of the relation between old and new node indexes, as the
        // analyser replaces the constant parts of the graph with Const nodes.
        let mapping = analyser.prune_unused();
        let output = mapping[output].ok_or("The output node doesn't exist in the streaming graph.")?;

        let successors = analyser.next_edges.iter()
            .map(|s| {
                s.iter()
                    .filter_map(|&e| {
                        let e = &analyser.edges[e];
                        e.to_node.map(|dest| (e.from_out, dest))
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let buffers = vec![Buffer::new(); analyser.nodes.len()];

        let mut dimensions = HashMap::with_capacity(analyser.edges.len());
        for edge in &analyser.edges {
            let source = &analyser.nodes[edge.from_node.unwrap()];

            // FIXME(liautaud): Get the actual streamed dimension.
            if source.op_name != "Const" {
                dimensions.insert((source.id, edge.from_out), 0);
            }
        }

        let model = analyser.into_model();

        Ok(StreamingState {model, output, mapping, buffers, dimensions, successors})
    }

    /// Runs one streaming evaluation step.
    ///
    /// The step starts by feeding a new chunk of data into one of the
    /// non-constant inputs of the model, which gets propagated to all
    /// the nodes in the graph in breadth-first ordering.
    ///
    /// The method will return a Vec<Vec<Tensor>>, which will contain
    /// a Vec<Tensor> for every chunk that was produced by the output
    /// during the evaluation step, with one Tensor per output port.
    pub fn step(&mut self, input: usize, input_chunk: Tensor) -> Result<Vec<Vec<Tensor>>> {
        let mut queue = VecDeque::new();
        let mut outputs = vec![];

        let input = self.mapping[input].ok_or("The input node doesn't exist in the streaming graph.")?;
        let input_view = Into::<TensorView>::into(input_chunk).into_shared();

        for &(port, target) in &self.successors[input] {
            queue.push_back((input, port, target, input_view.clone()));
        }

        while let Some((source, port, target, chunk)) = queue.pop_front() {
            debug!(
                "Executing new edge: source={:?}, port={:?}, target={:?}, chunk={:?}",
                source, port, target, chunk
            );

            let target = self.model.get_node_by_id(target)?;
            let mut inputs = vec![];

            // We wrap chunk in an option because we want to capture
            // its value in one and only one of the iterations, but
            // the borrow checker doesn't know that.
            let mut chunk = Some(chunk);

            for &(k, kp) in &target.inputs {
                let pred = self.model.get_node_by_id(k)?;
                let dimension = self.dimensions.get(&(k, kp.unwrap_or(0))).map(|i| *i);

                let value = if pred.op_name == "Const" {
                    // The input is not streamed, and so was turned into a constant
                    // node by the analyser when performing StreamingState::start.
                    Some(pred.op.eval(vec![])?.pop().unwrap())
                } else if k == source && kp.is_some() && kp.unwrap() == port {
                    // The input is streamed, and we've got a new chunk to give it.
                    // FIXME(liautaud): This doesn't work well if there are multiple
                    // edges from node source to node k, because the condition above
                    // will get verified for all edges but only one actually "holds"
                    // the chunk. The others will be None, and the unwrap will fail.
                    let chunk = chunk.take().unwrap();

                    // We only allow chunks of size 1 along the streaming dimension.
                    if chunk.as_tensor().shape()[dimension.unwrap()] != 1 {
                        bail!("Trying to consume a chunk of size != 1 along the streaming dimension.");
                    }

                    Some(chunk)
                } else {
                    // The input is streamed, but we don't have anything to give it yet.
                    None
                };

                inputs.push((dimension, value));
            }

            let buffer = &mut self.buffers[target.id];

            if let Some(mut output_chunks) = target.op.step(inputs, buffer)? {
                if target.id == self.output {
                    // If we've reached the output, just save the chunks.
                    outputs.push(output_chunks.clone());
                }

                // Propagate the chunks to the successors.
                for &(port, successor) in &self.successors[target.id] {
                    queue.push_back((target.id, port, successor, output_chunks[port].share()));
                }
            }
        }

        // Convert the output TensorViews to Tensors.
        let outputs = outputs
            .into_iter()
            .map(|chunks| chunks
                .into_iter()
                .map(|tv| tv.into_tensor())
                .collect()
            )
            .collect();

        Ok(outputs)
    }

    /// Resets the model state.
    pub fn reset(&mut self) -> Result<()> {
        unimplemented!()
    }
}
