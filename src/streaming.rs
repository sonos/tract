use std::ops::Deref;
use std::sync::Arc;

use super::*;
use analyser::interface::*;

/// The type of an input during streaming evaluation.
#[derive(Debug, Clone)]
pub enum StreamingInput {
    // The input is being streamed. We pass its datatype and shape along,
    // using None to denote the streaming dimension.
    Streamed(DataType, Vec<Option<usize>>),

    // The input will remain constant during the evaluation.
    Constant(Tensor),
}

#[derive(Clone, Debug)]
struct RawStreamingPlan {
    model: Model,
    output_node: usize,
    dimensions: HashMap<(usize, usize), usize>,
    // successors[src_node][src_port] -> vec<(dst_input, dst_ports)>
    successors: Vec<Vec<Vec<(usize, usize)>>>,
}

impl RawStreamingPlan {
    pub fn new(
        model: &Model,
        inputs: Vec<(usize, StreamingInput)>,
        output: Option<usize>,
    ) -> Result<RawStreamingPlan> {
        use self::StreamingInput::*;

        let output_node = output
            .or(analyser::detect_output(&model)?)
            .ok_or("Unable to auto-detect output node.")?;

        let mut analyser = Analyser::new(&model, output_node)?;

        // Pre-compute the constant part of the graph using the analyser.
        for input in inputs {
            match input {
                (i, Streamed(dt, shape)) => analyser.hint(
                    i,
                    &TensorFact {
                        datatype: typefact!(dt),
                        shape: shape.iter().cloned().collect(),
                        value: valuefact!(_),
                    },
                )?,
                (i, Constant(tensor)) => analyser.hint(i, &tensor_to_fact(tensor))?,
            }
        }
        analyser.analyse()?;

        let successors = vec!(vec!(); model.nodes.len());
        model.nodes.iter().for_each(|node| {
            node.inputs.iter().enumerate().for_each(|(dst_inlet, (src_node, src_outlet))| {
                while successors[*src_node].len() <= src_outlet.unwrap_or(0) {
                    successors[*src_node].push(vec!());
                }
                successors[*src_node][src_outlet.unwrap_or(0)].push((node.id, dst_inlet));
            });
        });

        let mut dimensions = HashMap::with_capacity(analyser.edges.len());
        for edge in &analyser.edges {
            let source = &analyser.get_node(edge.from_node.unwrap());
            let streamed = edge.fact.shape.dims.iter().position(|d| d.is_streamed());

            if source.op_name != "Const" && streamed.is_some() {
                debug!(
                    "Found streaming dimension {:?} for ({}, {:?}).",
                    streamed.unwrap(),
                    source.name,
                    edge.from_out,
                );

                dimensions.insert((source.id, edge.from_out), streamed.unwrap());
            }
        }

        Ok(RawStreamingPlan {
            model: model.clone(),
            dimensions,
            successors,
            output_node,
        })
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
        inputs: Vec<(usize, StreamingInput)>,
        output: Option<usize>,
    ) -> Result<StreamingPlan> {
        Ok(StreamingPlan(Arc::new(RawStreamingPlan::new(model, inputs, output)?)))
    }

    pub fn state(&self) -> Result<StreamingModelState> {
        let mut state = StreamingModelState {
            plan: self.clone(),
            buffers: vec![],
        };
        state.reset()?;
        Ok(state)
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
    buffers: Vec<Box<OpBuffer>>,
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
    pub fn step(&mut self, input: (usize, usize), input_chunk: Tensor) -> Result<Vec<Vec<Tensor>>> {
        self.step_wrapping_ops(input, input_chunk, |node, inputs, buffers| {
            node.op.step(inputs, buffers)
        })
    }

    // This function is not part of the public API, it's public to allow
    // instrumentation and auditing from cli.
    #[inline]
    #[doc(hidden)]
    pub fn step_wrapping_ops<W>(
        &mut self,
        input: (usize, usize),
        input_chunk: Tensor,
        mut node_step: W,
    ) -> Result<Vec<Vec<Tensor>>>
    where
        W: FnMut(&Node, Vec<(Option<usize>, Option<TensorView>)>, &mut Box<OpBuffer>)
            -> Result<Option<Vec<TensorView>>>,
    {
        let mut queue = VecDeque::new();
        let mut outputs = vec![];

        let input_view = Into::<TensorView>::into(input_chunk).into_shared();

        for dst in &self.plan.successors[input.0][input.1] {
            queue.push_back((dst, input_view.clone()));
        }

        while let Some((dst, chunk)) = queue.pop_front() {
            debug!("Executing new edge: dst={:?}, chunk={:?}", dst, chunk);

            let node = self.plan.model.get_node_by_id(dst.0)?;
            let mut inputs = vec![];

            // We wrap chunk in an option because we want to capture
            // its value in one and only one of the iterations, but
            // the borrow checker doesn't know that.
            let mut chunk = Some(chunk);

            for (ix, input) in node.inputs.iter().enumerate() {
                let pred = self.plan.model.get_node_by_id(input.0)?;
                let dimension = self.plan.dimensions.get(&(input.0, input.1.unwrap_or(0))).map(|i| *i);

                let value = if let Some(v) = pred.op.const_value() {
                    // The input is not streamed, and so was turned into a constant
                    // node by the analyser when performing StreamingState::start.
                    v
                } else if ix == dst.1 {
                    // The input is streamed, and we've got a new chunk to give it.
                    // FIXME(liautaud): This doesn't work well if there are multiple
                    // edges from node source to node k, because the condition above
                    // will get verified for all edges but only one actually "holds"
                    // the chunk. The others will be None, and the unwrap will fail.
                    let chunk = chunk.take().unwrap();

                    // We only allow chunks of size 1 along the streaming dimension.
                    if chunk.as_tensor().shape()[dimension.unwrap()] != 1 {
                        bail!(
                            "Trying to consume a chunk of size != 1 along the streaming dimension."
                        );
                    }

                    Some(chunk)
                } else {
                    // The input is streamed, but we don't have anything to give it yet.
                    None
                };

                inputs.push((dimension, value));
            }

            let buffer = &mut self.buffers[node.id];

            if let Some(mut output_chunks) = node_step(node, inputs, buffer)? {
                if node.id == self.plan.output_node {
                    // If we've reached the output, just save the chunks.
                    outputs.push(output_chunks.clone());
                }
                output_chunks.into_iter().enumerate().for_each(|(port, tensor)| {
                    for dst in self.plan.successors[node.id][port].iter() {
                        queue.push_back((dst, tensor.share()));
                    }
                });
            }
        }

        // Convert the output TensorViews to Tensors.
        let outputs = outputs
            .into_iter()
            .map(|chunks| chunks.into_iter().map(|tv| tv.into_tensor()).collect())
            .collect();

        Ok(outputs)
    }

    pub fn model(&self) -> &Model {
        &self.model
    }

    /// Resets the model state.
    pub fn reset(&mut self) -> Result<()> {
        self.buffers = self.model
            .model
            .nodes
            .iter()
            .map(|n| n.op.new_buffer())
            .collect::<Vec<_>>();
        Ok(())
    }
}
