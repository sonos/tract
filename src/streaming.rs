use super::*;
use analyser::interface::*;

/// The type of an input during streaming evaluation.
#[derive(Debug, Clone)]
pub enum StreamingInput {
    // The input is being streamed. We pass its datatype and shape along,
    // using None to denote the streaming dimension.
    Streamed(tfpb::types::DataType, Vec<Option<usize>>),

    // The input will remain constant during the evaluation.
    Constant(Tensor),
}

/// The state of a model during streaming evaluation.
#[derive(Clone)]
pub struct StreamingModel {
    model: Model,
    output: usize,
    mapping: Vec<Option<usize>>,
    dimensions: HashMap<(usize, usize), usize>,
    successors: Vec<Vec<(usize, usize)>>,
}

impl StreamingModel {
    /// Initializes the streaming evaluation of a model.
    ///
    /// For each input in the model, you must either provide a constant
    /// value or specify the dimension along which to stream.
    ///
    /// You will only be able to fetch the results of the evaluation step
    /// for the output node. If `output` is None, the output node will be
    /// guessed automatically.
    pub fn new(
        model: Model,
        inputs: Vec<(usize, StreamingInput)>,
        output: Option<usize>,
    ) -> Result<StreamingModel> {
        use self::StreamingInput::*;

        let output = output
            .or(analyser::detect_output(&model)?)
            .ok_or("Unable to auto-detect output node.")?;

        let mut analyser = Analyser::new(model, output)?;

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

        analyser.run()?;
        analyser.propagate_constants()?;

        // Keep track of the relation between old and new node indexes, as the
        // analyser replaces the constant parts of the graph with Const nodes.
        let mapping = analyser.prune_unused();
        let output =
            mapping[output].ok_or("The output node doesn't exist in the streaming graph.")?;

        let successors = analyser
            .next_edges
            .iter()
            .map(|s| {
                s.iter()
                    .filter_map(|&e| {
                        let e = &analyser.edges[e];
                        e.to_node.map(|dest| (e.from_out, dest))
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut dimensions = HashMap::with_capacity(analyser.edges.len());
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

                dimensions.insert((source.id, edge.from_out), streamed.unwrap());
            }
        }

        let model = analyser.into_model();
        Ok(StreamingModel {
            model,
            output,
            mapping,
            dimensions,
            successors,
        })
    }

    pub fn state(&self) -> StreamingModelState {
        let mut state = StreamingModelState {
            model: &self,
            buffers: vec![],
        };
        state.reset();
        state
    }

    /// Access the simplified Model for streaming records.
    /// This is not the original model from which the StreamingModel has been
    /// generated.
    pub fn inner_model(&self) -> &Model {
        &self.model
    }
}

pub struct StreamingModelState<'a> {
    model: &'a StreamingModel,
    buffers: Vec<Box<OpBuffer>>,
}

impl<'a> StreamingModelState<'a> {
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
        input: usize,
        input_chunk: Tensor,
        mut node_step: W,
    ) -> Result<Vec<Vec<Tensor>>>
    where
        W: FnMut(&Node, Vec<(Option<usize>, Option<TensorView>)>, &mut Box<OpBuffer>)
            -> Result<Option<Vec<TensorView>>>,
    {
        let mut queue = VecDeque::new();
        let mut outputs = vec![];

        let input = self.model.mapping[input]
            .ok_or("The input node doesn't exist in the streaming graph.")?;
        let input_view = Into::<TensorView>::into(input_chunk).into_shared();

        for &(port, target) in &self.model.successors[input] {
            queue.push_back((input, port, target, input_view.clone()));
        }

        while let Some((source, port, target, chunk)) = queue.pop_front() {
            debug!(
                "Executing new edge: source={:?}, port={:?}, target={:?}, chunk={:?}",
                source, port, target, chunk
            );

            let target = self.model.model.get_node_by_id(target)?;
            let mut inputs = vec![];

            // We wrap chunk in an option because we want to capture
            // its value in one and only one of the iterations, but
            // the borrow checker doesn't know that.
            let mut chunk = Some(chunk);

            for &(k, kp) in &target.inputs {
                let pred = self.model.model.get_node_by_id(k)?;
                let dimension = self.model.dimensions.get(&(k, kp.unwrap_or(0))).map(|i| *i);

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

            let buffer = &mut self.buffers[target.id];

            if let Some(mut output_chunks) = node_step(target, inputs, buffer)? {
                if target.id == self.model.output {
                    // If we've reached the output, just save the chunks.
                    outputs.push(output_chunks.clone());
                }

                // Propagate the chunks to the successors.
                for &(port, successor) in &self.model.successors[target.id] {
                    queue.push_back((target.id, port, successor, output_chunks[port].share()));
                }
            }
        }

        // Convert the output TensorViews to Tensors.
        let outputs = outputs
            .into_iter()
            .map(|chunks| chunks.into_iter().map(|tv| tv.into_tensor()).collect())
            .collect();

        Ok(outputs)
    }

    pub fn streaming_model(&self) -> &StreamingModel {
        &self.model
    }

    /// Resets the model state.
    pub fn reset(&mut self) {
        self.buffers = self.model
            .model
            .nodes
            .iter()
            .map(|n| n.op.new_buffer())
            .collect::<Vec<_>>();
    }
}
