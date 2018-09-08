use std::collections::VecDeque;
use std::ops::Deref;
use std::sync::Arc;

use super::*;
use analyser::prelude::*;
use model::*;
use ops::{StepValue, Stream, StreamInfo, Value};

#[derive(Clone, Debug)]
pub struct RawStreamingPlan {
    model: Model,
    input_nodes: Vec<(OutletId, StreamInfo)>,
    output: OutletId,
    proto_inputs: Vec<TVec<StepValue>>,
    stream_infos: Vec<TVec<Option<StreamInfo>>>,
    successors: Vec<TVec<TVec<InletId>>>,
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
                .ok_or_else(|| "Unable to auto-detect output node.")?,
        };

        let mut analyser = Analyser::new(&model, &output_node.name)?;
        let mut input_nodes = vec![];

        // Pre-compute the constant part of the graph using the analyser.
        for input in inputs {
            analyser.hint(input.0, &input.1)?;
            input_nodes.push((
                OutletId::new(model.node_by_name(input.0)?.id, 0),
                input
                    .1
                    .stream_info()?
                    .ok_or_else(|| format!("No streaming dim for {:?}", input))?,
            ));
        }
        analyser.analyse()?;

        let mut stream_infos = Vec::with_capacity(model.nodes.len());
        let mut proto_inputs = Vec::with_capacity(model.nodes.len());
        let mut successors: Vec<TVec<TVec<InletId>>> = vec![tvec![]; model.nodes.len()];
        for node in model.nodes.iter() {
            let mut inputs = tvec!();
            for ix in 0..node.inputs.len() {
                let edge_id = analyser.prev_edges[node.id][ix];
                let edge = &analyser.edges[edge_id];
                if let Some(info) = edge.fact.stream_info()? {
                    inputs.push(StepValue::Stream(Stream {
                        info,
                        offset: 0,
                        chunk: None,
                    }));
                } else {
                    let value: Value = edge
                        .fact
                        .concretize()
                        .ok_or_else(|| "Failed analysis")?
                        .into();
                    inputs.push(StepValue::Const(value.into_shared()))
                }
                let from = edge.from.unwrap(); //checked
                while successors[from.node].len() <= from.slot {
                    successors[from.node].push(tvec!())
                }
                successors[from.node][from.slot].push(InletId::new(node.id, ix));
            }
            proto_inputs.push(inputs);
            let mut outputs = tvec!();
            for edge_id in &analyser.next_edges[node.id] {
                let edge = &analyser.edges[*edge_id];
                outputs.push(edge.fact.stream_info()?);
            }
            stream_infos.push(outputs);
        }

        Ok(RawStreamingPlan {
            model: analyser.finalize_model()?,
            stream_infos,
            proto_inputs,
            successors,
            output: OutletId::new(output_node.id, 0),
            input_nodes,
        })
    }

    pub fn stream_info(&self, outlet: &OutletId) -> Option<StreamInfo> {
        *self.stream_infos.get(outlet.node)?.get(outlet.slot)?
    }

    pub fn output_stream_info(&self) -> ::Result<StreamInfo> {
        self.stream_info(&self.output)
            .ok_or_else(|| "Output is not a stream".into())
    }

    pub fn successors(&self, edge: OutletId) -> &[InletId] {
        self.successors[edge.node]
            .get(edge.slot)
            .map(|v| &v[..])
            .unwrap_or(&[])
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
            inlets_offset: vec![],
            buffers: vec![],
            queue: VecDeque::new(),
            outputs: vec![],
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
    inlets_offset: Vec<TVec<u64>>,
    buffers: Vec<Box<ops::OpBuffer>>,
    queue: VecDeque<(InletId, ops::Value)>,
    outputs: Vec<TVec<Tensor>>,
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
    pub fn step(&mut self, input_id: usize, input_chunk: Tensor) -> Result<Vec<TVec<Tensor>>> {
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
    ) -> Result<Vec<TVec<Tensor>>>
    where
        W: FnMut(&Node, TVec<StepValue>, &mut Box<ops::OpBuffer>)
            -> Result<Option<TVec<ops::Value>>>,
    {
        let (input_outlet, _input_stream_info) = self.plan.input_nodes[input_id];
        self.enqueue(input_chunk.into(), input_outlet);

        while let Some((inlet, chunk)) = self.queue.pop_front() {
            let output = {
                let node = &self.plan.model.nodes[inlet.node];
                debug!(
                    "Feeding node: {} {:?} ({}), chunk={:?} inlet:{:?}",
                    node.id, node.name, node.op_name, chunk, inlet,
                );

                let mut inputs: TVec<StepValue> = self.plan.proto_inputs[node.id].clone();
                debug!("proto input: {:?}", inputs);
                if let StepValue::Stream(ref mut stream) = inputs[inlet.inlet] {
                    let offset = self.inlets_offset[inlet.node][inlet.inlet];
                    self.inlets_offset[inlet.node][inlet.inlet] +=
                        chunk.shape()[stream.info.axis] as u64;
                    stream.offset = offset;
                    stream.chunk = Some(chunk);
                }

                debug!(
                    "Pushing to {} {:?} ({}), inputs: {:?}",
                    node.id, node.name, node.op_name, inputs
                );
                let output = node_step(node, inputs, &mut self.buffers[node.id])?;

                let node = &self.plan.model.nodes[inlet.node];
                debug!(
                    "Node: {} {:?} ({}), generated chunk={:?}",
                    node.id, node.name, node.op_name, &output
                );
                output
            };

            if let Some(mut output_chunks) = output {
                if inlet.node == self.plan.output.node {
                    // If we've reached the output, just save the chunks.
                    self.outputs.push(
                        output_chunks
                            .iter()
                            .map(|tv| tv.as_tensor().clone())
                            .collect(),
                    );
                }
                for (port, value) in output_chunks.into_iter().enumerate() {
                    let tensor = value.into_tensor();
                    let outlet = OutletId::new(inlet.node, port);
                    let info = self
                        .plan
                        .stream_info(&outlet)
                        .ok_or_else(|| "Expected a stream")?;

                    if tensor.shape()[info.axis] > 1 {
                        for chunk in tensor.axis_chunks(info.axis, 1)? {
                            self.enqueue(chunk.into(), outlet);
                        }
                    } else {
                        self.enqueue(tensor.into(), outlet);
                    }
                }
            }
        }

        let mut outputs = vec![];
        std::mem::swap(&mut outputs, &mut self.outputs);
        Ok(outputs)
    }

    fn enqueue(&mut self, value: Value, outlet: OutletId) {
        let dst = self.plan.successors(outlet);
        if dst.len() == 1 {
            self.queue.push_back((dst[0], value));
        } else {
            let value = value.into_shared();
            for dst in dst.iter() {
                self.queue.push_back((*dst, value.clone()));
            }
        }
    }

    pub fn model(&self) -> &Model {
        &self.plan.model
    }

    /// Resets the model state.
    pub fn reset(&mut self) -> Result<()> {
        self.inlets_offset = self
            .model()
            .nodes
            .iter()
            .map(|node| tvec!(0; node.inputs.len()))
            .collect();
        self.buffers = self
            .model()
            .nodes
            .iter()
            .map(|n| n.op.new_buffer())
            .collect::<Vec<_>>();
        self.queue.clear();
        Ok(())
    }
}
