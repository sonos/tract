use crate::kernels::conv::{ConvGeneric, ConvKernel, ConvKernelScratch};
use crate::kernels::conv_cudnn::ConvCudnn;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use tract_core::internal::*;
use tract_core::ops::cnn::Conv;
use tract_gpu::ops::change_axes::GpuAxisOp;
use tract_gpu::tensor::DeviceTensorExt;

pub fn wire_cuda_conv(
    source: &TypedModel,
    node: &TypedNode,
    target: &mut TypedModel,
    inputs: &[OutletId],
    op: &Conv,
) -> TractResult<TVec<OutletId>> {
    let facts = source.node_input_facts(node.id)?;
    let data_shape = op.pool_spec.data_format.shape(&facts[0].shape)?;
    let hw_rank = data_shape.hw_rank();
    let is_f16 = facts[0].datum_type.is::<f16>();
    if facts.iter().all(|f| f.datum_type.is::<f32>() || f.datum_type.is::<f16>())
        && hw_rank <= if is_f16 { 2 } else { 6 }
        && op
            .pool_spec
            .computed_padding(data_shape.hw_dims())
            .iter()
            .all(|paddings| paddings.pad_before == paddings.pad_after)
    {
        let prefix = &node.name;
        let bias = &facts[2];
        let need_bias = !(bias.konst.is_some() && bias.konst.as_ref().unwrap().is_all_zero()?);
        let conv_name = format!("{prefix}.conv");
        let mut conv_wire = target.wire_node(
            if need_bias { &conv_name } else { &node.name },
            CudaConv { op: op.clone(), kernel: Box::new(ConvCudnn) },
            &inputs[0..2],
        )?[0];
        if need_bias {
            let mut needed_shape = tvec![1.to_dim(); node.outputs[0].fact.rank()];
            needed_shape[data_shape.c_axis()] = op.pool_spec.output_channels.to_dim();
            let reshaped = target.wire_node(
                format!("{prefix}.bias_reshaped"),
                GpuAxisOp::new(AxisOp::Reshape(0, bias.shape.to_tvec(), needed_shape)),
                &[inputs[2]],
            )?[0];
            conv_wire = target.wire_node(
                prefix,
                crate::kernels::binary::cuda_bin_op(Box::new(tract_core::ops::math::Add)),
                &[conv_wire, reshaped],
            )?[0];
        }
        Ok(tvec!(conv_wire))
    } else {
        target.wire_node(
            &node.name,
            CudaConv { op: op.clone(), kernel: Box::new(ConvGeneric) },
            inputs,
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaConv {
    op: Conv,
    kernel: Box<dyn ConvKernel>,
}

impl Op for CudaConv {
    fn name(&self) -> StaticName {
        "CudaConv".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        let mut info = self.op.info()?;
        info.push(format!("kernel: {}", self.kernel.name()));
        Ok(info)
    }

    op_as_typed_op!();
}

impl TypedOp for CudaConv {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        tract_gpu::utils::facts_to_device_facts(inputs, |facts| {
            let zero = facts[0].datum_type.scalar_fact();
            let mut facts: TVec<&TypedFact> = facts.into();
            if facts.len() == 2 {
                facts.push(&zero);
            }
            self.op.output_facts(&facts)
        })
        .with_context(|| format!("Error while computing facts for Conv/{:?}", self.kernel.name()))
    }
}

// cudnn descriptors are not Send, so they cannot live in TurnState::scratch. They
// stay in a thread-local keyed by an id, and TurnState::scratch holds only the
// node -> id map, which is Send. Dropping the map -- i.e. dropping the state that
// owns it -- evicts that plan's descriptors from the cache.
thread_local! {
    static CUDA_CONV_SCRATCH: RefCell<HashMap<usize, Box<dyn ConvKernelScratch>>> =
        RefCell::new(HashMap::new());
}

static NEXT_CUDA_CONV_SCRATCH_ID: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug, Default)]
struct CudaConvScratchIds(HashMap<usize, usize>);

impl CudaConvScratchIds {
    fn id_for(&mut self, node_id: usize) -> usize {
        *self
            .0
            .entry(node_id)
            .or_insert_with(|| NEXT_CUDA_CONV_SCRATCH_ID.fetch_add(1, Ordering::Relaxed))
    }
}

impl Drop for CudaConvScratchIds {
    fn drop(&mut self) {
        CUDA_CONV_SCRATCH.with_borrow_mut(|cache| {
            for id in self.0.values() {
                cache.remove(id);
            }
        });
    }
}

impl EvalOp for CudaConv {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_turn(
        &self,
        node_id: usize,
        turn: &mut TurnState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let inputs =
            inputs.iter().map(|it| it.to_device_tensor()).collect::<TractResult<TVec<_>>>()?;
        let output_shape = self.op.pool_spec.output_shape(inputs[0].shape())?;
        let scratch_id = turn.scratch.entry::<CudaConvScratchIds>().or_default().id_for(node_id);
        let output = tract_gpu::turn_handler::make_tensor_for_node(
            turn,
            node_id,
            inputs[0].datum_type(),
            &output_shape.shape,
        )?;

        if output.len() > 0 {
            crate::with_cuda_stream(|stream| {
                CUDA_CONV_SCRATCH.with_borrow_mut(|cache| {
                    let scratch = cache.entry(scratch_id).or_insert_with(|| self.kernel.state());
                    self.kernel.dispatch(
                        &mut **scratch,
                        node_id,
                        &self.op,
                        stream,
                        inputs[0],
                        inputs[1],
                        inputs.get(2).cloned(),
                        &output,
                    )
                })
            })?;
        }
        Ok(tvec!(output.into_tensor().into_tvalue()))
    }
}
