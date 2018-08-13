use std::collections::{HashMap, VecDeque};
use std::ops::Deref;
use std::sync::Arc;

use super::*;
use analyser::interface::*;
use ops::StepValue;

#[derive(Clone, Debug)]
pub struct RawStreamingPlan {
    model: Model,
    // inputs nodes in the form (id, streaming dim)
    input_nodes: Vec<(usize, usize)>,
    output_node: usize,
    streaming_dimensions: HashMap<(usize, usize), usize>,
    // successors[src_node][src_port] -> vec<(dst_input, dst_ports)>
    successors: Vec<Vec<Vec<(usize, usize)>>>,
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
            input_nodes.push((model.node_by_name(input.0)?.id, input.1.streaming_dim()?));
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
                    successors[*src_node][*src_outlet].push((node.id, dst_inlet));
                });
        });

        let mut streaming_dimensions = HashMap::with_capacity(analyser.edges.len());
        for edge in &analyser.edges {
            let source = &analyser.nodes[edge.from_node.unwrap()];
            let streamed = edge.fact.shape.dims.iter().position(|d| d.is_streamed());

            if source.op_name != "Const" && streamed.is_some() {
                debug!(
                    "Found streaming dimension {:?} for ({}, {:?}).",
                    streamed.unwrap(),
                    source.name,
                    edge.from_out,
                );

                streaming_dimensions.insert((source.id, edge.from_out), streamed.unwrap());
            }
        }

        Ok(RawStreamingPlan {
            model: model.clone(),
            streaming_dimensions,
            successors,
            output_node: output_node.id,
            input_nodes,
        })
    }

    pub fn output_streaming_dim(&self) -> Result<usize> {
        Ok(self.streaming_dimensions[&(self.output_node, 0)])
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
        let mut queue = VecDeque::new();
        let mut outputs = vec![];

        let input_view = ops::Value::from(input_chunk).into_shared();

        let input = self.plan.input_nodes[input_id];
        for dst in &self.plan.successors[input.0][input.1] {
            queue.push_back((*dst, input_view.clone()));
        }

        while let Some((dst, chunk)) = queue.pop_front() {
            let node = &self.plan.model.nodes[dst.0];
            debug!(
                "Feeding node: {} {:?} ({}), chunk={:?} (stream dim: {})",
                node.id, node.name, node.op_name, chunk, dst.1
            );

            let mut inputs: Vec<StepValue> = vec![];

            // We wrap chunk in an option because we want to capture
            // its value in one and only one of the iterations, but
            // the borrow checker doesn't know that.
            let mut chunk = Some(chunk);

            for (ix, input) in node.inputs.iter().enumerate() {
                let pred = &self.plan.model.nodes[input.0];

                let value = if let Some(dimension) =
                    self.plan.streaming_dimensions.get(&input).map(|i| *i)
                {
                    if ix == dst.1 {
                        // The input is streamed, and we've got a new chunk to give it.
                        // FIXME(liautaud): This doesn't work well if there are multiple
                        // edges from node source to node k, because the condition above
                        // will get verified for all edges but only one actually "holds"
                        // the chunk. The others will be None, and the unwrap will fail.
                        let chunk = chunk.take().ok_or("streamable edge used twice")?;

                        // We only allow chunks of size 1 along the streaming dimension.
                        if chunk.as_tensor().shape()[dimension] != 1 {
                            bail!(
                                "Trying to consume a chunk of size != 1 along the streaming dimension."
                            );
                        }

                        StepValue::Stream(dimension, Some(chunk))
                    } else {
                        // The input is streamed, but this chunk is for another input.
                        StepValue::Stream(dimension, None)
                    }
                } else {
                    // The input is not streamed, and so was turned into a constant
                    // node by the analyser when performing StreamingState::start.
                    StepValue::Const(
                        pred.op
                            .const_value()
                            .ok_or("Streaming mode should have only const or streamable edges.")?
                            .into(),
                    )
                };

                inputs.push(value);
            }

            let output = node_step(node, inputs, &mut self.buffers[node.id])?;
            if let Some(mut output_chunks) = output {
                if node.id == self.plan.output_node {
                    // If we've reached the output, just save the chunks.
                    outputs.push(output_chunks.clone());
                }
                output_chunks
                    .into_iter()
                    .enumerate()
                    .for_each(|(port, mut tensor)| {
                        if let Some(dst) = self.plan.successors[node.id].get(port) {
                            for dst in dst.iter() {
                                queue.push_back((*dst, tensor.share()));
                            }
                        }
                    });
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
