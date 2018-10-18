use std::borrow::Borrow;
use std::collections::VecDeque;
use std::marker::PhantomData;

use analyser::prelude::*;
use model::*;
use ops::prelude::*;

pub mod types;
pub mod values;

pub mod prelude {
    pub use streaming::types::OpBuffer;
    pub use streaming::values::{StepValue, Stream};
}

use streaming::values::StepValue;

#[derive(Clone, Debug)]
pub struct StreamingPlan<M: Borrow<Model>> {
    model: M,
    //    input_nodes: Vec<(OutletId, StreamInfo)>,
    proto_inputs: Vec<TVec<StepValue>>,
    //    stream_infos: Vec<TVec<Option<StreamInfo>>>,
}

impl<M: Borrow<Model>> StreamingPlan<M> {
    pub fn new(model: M) -> TfdResult<StreamingPlan<M>> {
        let mut proto_inputs = Vec::with_capacity(model.borrow().nodes().len());
        for node in model.borrow().nodes() {
            let mut inputs = tvec!();
            for ix in 0..node.inputs.len() {
                let edge_id = node.inputs[ix];
                let fact = model.borrow().fact(edge_id)?;
                if let Some(info) = fact.stream_info()? {
                    inputs.push(StepValue::Stream(Stream {
                        info,
                        offset: 0,
                        chunk: None,
                    }));
                } else {
                    let value: Value = fact.concretize().ok_or_else(|| "Failed analysis")?.into();
                    inputs.push(StepValue::Const(value.into_shared()))
                }
            }
            proto_inputs.push(inputs);
            /*
            let mut outputs = tvec!();
            for edge_id in &analyser.next_edges[node.id] {
                let edge = &analyser.edges[*edge_id];
                outputs.push(edge.fact.stream_info()?);
            }
            stream_infos.push(outputs);
            */
        }

        Ok(StreamingPlan {
            model: model,
            // stream_infos,
            proto_inputs,
            // input_nodes,
        })
    }

    pub fn stream_info(&self, outlet: OutletId) -> TfdResult<Option<StreamInfo>> {
        self.model().fact(outlet)?.stream_info()
    }

    pub fn output_stream_info(&self) -> TfdResult<StreamInfo> {
        self.stream_info(self.model().outputs()?[0])?
            .ok_or_else(|| "Output is not a stream".into())
    }

    pub fn successors(&self, edge: OutletId) -> impl Iterator<Item = InletId> + '_ {
        self.model().nodes()[edge.node].outputs[edge.slot]
            .successors
            .iter()
            .cloned()
    }

    pub fn model(&self) -> &Model {
        &self.model.borrow()
    }
}

/*
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
    ) -> TfdResult<StreamingPlan> {
        Ok(StreamingPlan(Arc::new(RawStreamingPlan::new(
            model, inputs, output,
        )?)))
    }

    pub fn state(&self) -> TfdResult<StreamingModelState> {
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

}

impl Deref for StreamingPlan {
    type Target = RawStreamingPlan;
    fn deref(&self) -> &RawStreamingPlan {
        &*self.0
    }
}
*/

#[derive(Clone, Debug)]
pub struct StreamingModelState<M: Borrow<Model>, P: Borrow<StreamingPlan<M>>> {
    plan: P,
    inlets_offset: Vec<TVec<u64>>,
    buffers: Vec<Box<OpBuffer>>,
    queue: VecDeque<(InletId, Value)>,
    outputs: Vec<TVec<Tensor>>,
    _phantom: PhantomData<M>,
}

impl<M: Borrow<Model>, P: Borrow<StreamingPlan<M>>> StreamingModelState<M, P> {
    pub fn new(plan: P) -> TfdResult<StreamingModelState<M, P>> {
        let mut state = StreamingModelState {
            plan,
            inlets_offset: vec![],
            buffers: vec![],
            queue: VecDeque::new(),
            outputs: vec![],
            _phantom: PhantomData,
        };
        state.reset()?;
        Ok(state)
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
    pub fn step(&mut self, input_chunk: Tensor) -> TfdResult<Vec<TVec<Tensor>>> {
        self.step_wrapping_ops(input_chunk, |node, inputs, buffers| {
            node.op.step(inputs, buffers)
        })
    }

    // This function is not part of the public API, it's public to allow
    // instrumentation and auditing from cli.
    #[inline]
    #[doc(hidden)]
    pub fn step_wrapping_ops<W>(
        &mut self,
        input_chunk: Tensor,
        mut node_step: W,
    ) -> TfdResult<Vec<TVec<Tensor>>>
    where
        W: FnMut(&Node, TVec<StepValue>, &mut Box<OpBuffer>) -> TfdResult<Option<TVec<Value>>>,
    {
        let input_outlet = self.model().inputs()?[0];
        self.enqueue(input_chunk.into(), input_outlet);

        while let Some((inlet, chunk)) = self.queue.pop_front() {
            let output = {
                let node = &self.plan.borrow().model().nodes()[inlet.node];
                debug!(
                    "Feeding node: {} {:?} ({}), chunk={:?} inlet:{:?}",
                    node.id,
                    node.name,
                    node.op.name(),
                    chunk,
                    inlet,
                );

                let mut inputs: TVec<StepValue> = self.plan().proto_inputs[node.id].clone();
                debug!("proto input: {:?}", inputs);
                if let StepValue::Stream(ref mut stream) = inputs[inlet.slot] {
                    let offset = self.inlets_offset[inlet.node][inlet.slot];
                    self.inlets_offset[inlet.node][inlet.slot] +=
                        chunk.shape()[stream.info.axis] as u64;
                    stream.offset = offset;
                    stream.chunk = Some(chunk);
                }

                debug!(
                    "Pushing to {} {:?} ({}), inputs: {:?}",
                    node.id,
                    node.name,
                    node.op.name(),
                    inputs
                );
                let output = node_step(node, inputs, &mut self.buffers[node.id])?;

                let node = &self.model().nodes()[inlet.node];
                debug!(
                    "Node: {} {:?} ({}), generated chunk={:?}",
                    node.id,
                    node.name,
                    node.op.name(),
                    &output
                );
                output
            };

            if let Some(mut output_chunks) = output {
                if inlet.node == self.model().outputs()?[0].node {
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
                        .plan()
                        .stream_info(outlet)?
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
        ::std::mem::swap(&mut outputs, &mut self.outputs);
        Ok(outputs)
    }

    fn enqueue(&mut self, value: Value, outlet: OutletId) {
        let mut dst = self.plan.borrow().successors(outlet);
        if dst.size_hint() == (1, Some(1)) {
            self.queue.push_back((dst.next().unwrap(), value));
        } else {
            let value = value.into_shared();
            for dst in dst {
                self.queue.push_back((dst, value.clone()));
            }
        }
    }

    pub fn plan(&self) -> &StreamingPlan<M> {
        &self.plan.borrow()
    }

    pub fn model(&self) -> &Model {
        &self.plan().model()
    }

    /// Resets the model state.
    pub fn reset(&mut self) -> TfdResult<()> {
        self.inlets_offset = self
            .model()
            .nodes()
            .iter()
            .map(|node| tvec!(0; node.inputs.len()))
            .collect();
        self.buffers = self
            .model()
            .nodes()
            .iter()
            .map(|n| n.op.new_buffer())
            .collect::<Vec<_>>();
        self.queue.clear();
        Ok(())
    }
}
