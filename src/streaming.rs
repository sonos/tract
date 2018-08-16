use std::collections::{HashMap, VecDeque};
use std::ops::Deref;
use std::sync::Arc;

use super::*;
use analyser::interface::*;
use ops::{ StepValue, StreamInfo, Stream };

#[derive(Clone,Copy,Debug,PartialEq,Eq,Hash)]
pub struct OutletId {
    pub node: usize,
    pub outlet: usize,
}

impl OutletId {
    pub fn new(node: usize, outlet: usize) -> OutletId {
        OutletId { node, outlet }
    }
}

#[derive(Clone,Copy,Debug,PartialEq,Eq,Hash)]
pub struct InletId {
    pub node: usize,
    pub inlet: usize,
}

impl InletId {
    pub fn new(node: usize, inlet: usize) -> InletId {
        InletId { node, inlet }
    }
}

#[derive(Clone, Debug)]
pub struct RawStreamingPlan {
    model: Model,
    input_nodes: Vec<(OutletId, StreamInfo)>,
    output_node: usize,
    stream_infos: HashMap<OutletId, StreamInfo>,
    // successors[outlet.node][outlet.outlet]
    successors: Vec<Vec<Vec<InletId>>>,
}

impl RawStreamingPlan {
    pub fn new(
        model: &Model,
        inputs: Vec<(&str, TensorFact)>,
        output: Option<&str>,
    ) -> Result<RawStreamingPlan> {
        let output_node = match output {
            Some(name) => model.node_by_name(name)?,
            None => analyser::detect_inputs(&model)?
                .pop()
                .ok_or("Unable to auto-detect output node.")?,
        };

        let mut analyser = Analyser::new(&model, &output_node.name)?;
        let mut input_nodes = vec![];

        // Pre-compute the constant part of the graph using the analyser.
        for input in inputs {
            analyser.hint(input.0, &input.1)?;
            input_nodes.push((OutletId::new(model.node_by_name(input.0)?.id, 0), input.1.stream_info()?
                              .ok_or_else(|| format!("No streaming dim for {:?}", input))?));
        }
        analyser.analyse()?;

        let mut successors = vec![vec![]; model.nodes.len()];
        model.nodes.iter().for_each(|node| {
            node.inputs
                .iter()
                .enumerate()
                .for_each(|(dst_inlet, (src_node, src_outlet))| {
                    while successors[*src_node].len() <= *src_outlet {
                        successors[*src_node].push(vec![]);
                    }
                    successors[*src_node][*src_outlet].push(InletId::new(node.id, dst_inlet));
                });
        });

        let mut stream_infos = HashMap::with_capacity(analyser.edges.len());
        for edge in &analyser.edges {
            let source = &analyser.nodes[edge.from_node.unwrap()];
            if source.op_name == "Const" {
                continue;
            }

            let stream_info = edge.fact.stream_info()?;
            if let Some(stream) = stream_info {
                debug!(
                    "Found streaming dimension {:?} for ({}, {:?}).",
                    stream,
                    source.name,
                    edge.from_out,
                );

                stream_infos.insert(OutletId::new(source.id, edge.from_out), stream);
            }
        }

        Ok(RawStreamingPlan {
            model: analyser.finalize_model()?,
            stream_infos,
            successors,
            output_node: output_node.id,
            input_nodes,
        })
    }

    pub fn output_stream_info(&self) -> Result<StreamInfo> {
        Ok(self.stream_infos[&OutletId::new(self.output_node, 0)])
    }

    pub fn successors(&self, edge: OutletId) -> &[InletId] {
        &self.successors[edge.node][edge.outlet]
    }
}

#[derive(Clone, Debug)]
pub struct StreamingPlan(Arc<RawStreamingPlan>);

impl StreamingPlan {
    /// Initializes the streaming evaluation of a model.
    ///
    /// For each input in the model, you must either provide a constant
    /// value or specify the dimension along which to stream.
    ///
    /// You will only be able to fetch the results of the evaluation step
    /// for the output node. If `output` is None, the output node will be
    /// guessed automatically.
    pub fn new(
        model: &Model,
        inputs: Vec<(&str, TensorFact)>,
        output: Option<&str>,
    ) -> Result<StreamingPlan> {
        Ok(StreamingPlan(Arc::new(RawStreamingPlan::new(
            model, inputs, output,
        )?)))
    }

    pub fn state(&self) -> Result<StreamingModelState> {
        let mut state = StreamingModelState {
            plan: self.clone(),
            inlets_offset: HashMap::new(),
            buffers: vec![],
        };
        state.reset()?;
        Ok(state)
    }

    pub fn model(&self) -> &Model {
        &self.model
    }
}

impl Deref for StreamingPlan {
    type Target = RawStreamingPlan;
    fn deref(&self) -> &RawStreamingPlan {
        &*self.0
    }
}

#[derive(Clone, Debug)]
pub struct StreamingModelState {
    plan: StreamingPlan,
    inlets_offset: HashMap<InletId, u64>,
    buffers: Vec<Box<ops::OpBuffer>>,
}

impl StreamingModelState {
    /// Runs one streaming evaluation step.
    ///
    /// The step starts by feeding a new chunk of data into one of the
    /// non-constant inputs of the model, which gets propagated to all
    /// the nodes in the graph in breadth-first ordering.
    ///
    /// The method will return a Vec<Vec<Tensor>>, which will contain
    /// a Vec<Tensor> for every chunk that was produced by the output
    /// during the evaluation step, with one Tensor per output port.
    pub fn step(&mut self, input_id: usize, input_chunk: Tensor) -> Result<Vec<Vec<Tensor>>> {
        self.step_wrapping_ops(input_id, input_chunk, |node, inputs, buffers| {
            node.op.step(inputs, buffers)
        })
    }

    // This function is not part of the public API, it's public to allow
    // instrumentation and auditing from cli.
    #[inline]
    #[doc(hidden)]
    pub fn step_wrapping_ops<W>(
        &mut self,
        input_id: usize,
        input_chunk: Tensor,
        mut node_step: W,
    ) -> Result<Vec<Vec<Tensor>>>
    where
        W: FnMut(&Node, Vec<StepValue>, &mut Box<ops::OpBuffer>) -> Result<Option<Vec<ops::Value>>>,
    {
        let mut queue:VecDeque<(InletId, ops::Value)> = VecDeque::new();
        let mut outputs = vec![];

        let input_view = ops::Value::from(input_chunk).into_shared();

        let (input_outlet, _input_stream_info) = self.plan.input_nodes[input_id];
        for inlet in self.plan.successors(input_outlet) {
            queue.push_back((*inlet, input_view.clone()));
        }

        while let Some((inlet, chunk)) = queue.pop_front() {
            let node = &self.plan.model.nodes[inlet.node];
            debug!(
                "Feeding node: {} {:?} ({}), chunk={:?} inlet:{:?}",
                node.id, node.name, node.op_name, chunk, inlet,
            );

            // We wrap chunk in an option because we want to capture
            // its value in one and only one of the iterations, but
            // the borrow checker doesn't know that.
            let mut chunk = Some(chunk);

            let mut inputs:Vec<StepValue> = vec!();
            for (ix,input) in node.inputs.iter().enumerate() {
                let input = OutletId::new(input.0, input.1);
                debug!("making input {}", ix);
                if let Some(&info) = self.plan.stream_infos.get(&input) {
                    let chunk = if inlet.inlet == ix { chunk.take() } else { None };

                    let offset_ref = self.inlets_offset.entry(InletId::new(inlet.node, ix)).or_insert(0u64);
                    let offset = *offset_ref;
                    *offset_ref += chunk.as_ref().map(|t| t.shape()[info.axis]).unwrap_or(0) as u64;

                    inputs.push(StepValue::Stream(Stream { info, offset, chunk}))

                } else {
                    let pred = &self.plan.model.nodes[input.node];
                    // The input is not streamed, and so was turned into a constant
                    // node by the analyser when performing StreamingState::start.
                    inputs.push(StepValue::Const(
                        pred.op
                            .const_value()
                            .ok_or("Streaming mode should have only const or streamable edges.")?
                            .into(),
                    ))
                };
            };

            debug!(
                "Pushing to {} {:?} ({}), inputs: {:?}",
                node.id, node.name, node.op_name, inputs
            );
            let output = node_step(node, inputs, &mut self.buffers[node.id])?;
            debug!(
                "Node: {} {:?} ({}), generated chunk={:?}",
                node.id, node.name, node.op_name, &output
            );
            if let Some(mut output_chunks) = output {
                if node.id == self.plan.output_node {
                    // If we've reached the output, just save the chunks.
                    outputs.push(output_chunks.clone());
                }
                for (port, tensor) in output_chunks.into_iter().enumerate() {
                    let outlet = OutletId::new(node.id, port);
                    let info = self.plan.stream_infos[&outlet];
                    for chunk in tensor.axis_chunks(info.axis, 1)? {
                        let mut value:ops::Value = chunk.into();
                        if let Some(dst) = self.plan.successors[node.id].get(port) {
                            for dst in dst.iter() {
                                queue.push_back((*dst, value.share()));
                            }
                        }
                    }
                }
            }
        }


        // Convert the output Values to Tensors.
        let outputs = outputs
            .into_iter()
            .map(|chunks| chunks.into_iter().map(|tv| tv.into_tensor()).collect())
            .collect();

        Ok(outputs)
    }

    pub fn model(&self) -> &Model {
        &self.plan.model
    }

    /// Resets the model state.
    pub fn reset(&mut self) -> Result<()> {
        self.buffers = self
            .model()
            .nodes
            .iter()
            .map(|n| n.op.new_buffer())
            .collect::<Vec<_>>();
        Ok(())
    }
}
