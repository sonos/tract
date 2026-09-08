use crate::kernels::pool::{PoolKind, wgpu_pool_dispatch, wgpu_pool_supported};
use tract_core::internal::*;
use tract_core::ops::cnn::{MaxPool, OptMaxPool, OptSumPool, PoolSpec, SumPool};
use tract_gpu::tensor::DeviceTensorExt;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WgpuPool {
    pub pool_spec: PoolSpec,
    pub kind: PoolKindOp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PoolKindOp {
    Max,
    Sum { count_include_pad: bool, normalize: bool },
}

impl From<PoolKindOp> for PoolKind {
    fn from(k: PoolKindOp) -> Self {
        match k {
            PoolKindOp::Max => PoolKind::Max,
            PoolKindOp::Sum { count_include_pad, normalize } => {
                PoolKind::Sum { count_include_pad, normalize }
            }
        }
    }
}

impl Op for WgpuPool {
    fn name(&self) -> StaticName {
        match self.kind {
            PoolKindOp::Max => "WgpuMaxPool".into(),
            PoolKindOp::Sum { .. } => "WgpuSumPool".into(),
        }
    }
    fn info(&self) -> TractResult<Vec<String>> {
        Ok(self.pool_spec.info())
    }
    op_as_typed_op!();
}

impl EvalOp for WgpuPool {
    op_out_of_plan!();

    fn eval(&self, ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input = inputs[0].to_device_tensor()?;
        let output_shape = self.pool_spec.output_shape(input.shape())?;
        let output = tract_gpu::turn_handler::make_tensor_for_node(
            ctx,
            input.datum_type(),
            &output_shape.shape,
        )?;
        if output.len() > 0 {
            wgpu_pool_dispatch(&self.pool_spec, self.kind.into(), input, &output)?;
        }
        Ok(tvec!(output.into_tensor().into_tvalue()))
    }
}

impl TypedOp for WgpuPool {
    as_op!();
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        tract_gpu::utils::facts_to_device_facts(inputs, |facts| {
            let shape = self.pool_spec.output_shape(&facts[0].shape)?;
            Ok(tvec!(facts[0].datum_type.fact(shape.shape)))
        })
        .with_context(|| "Error while computing facts for WgpuPool")
    }
}

crate::register_wgpu_op!(OptMaxPool, |source, node, op| {
    let facts = source.node_input_facts(node.id)?;
    if op.with_index_outputs.is_some() || !wgpu_pool_supported(&op.pool_spec, facts[0]) {
        return Ok(None);
    }
    Ok(Some(Box::new(WgpuPool { pool_spec: op.pool_spec.clone(), kind: PoolKindOp::Max })
        as Box<dyn TypedOp>))
});

crate::register_wgpu_op!(MaxPool, |source, node, op| {
    let facts = source.node_input_facts(node.id)?;
    if op.with_index_outputs.is_some() || !wgpu_pool_supported(&op.pool_spec, facts[0]) {
        return Ok(None);
    }
    Ok(Some(Box::new(WgpuPool { pool_spec: op.pool_spec.clone(), kind: PoolKindOp::Max })
        as Box<dyn TypedOp>))
});

crate::register_wgpu_op!(OptSumPool, |source, node, op| {
    let facts = source.node_input_facts(node.id)?;
    if !wgpu_pool_supported(&op.pool_spec, facts[0]) {
        return Ok(None);
    }
    Ok(Some(Box::new(WgpuPool {
        pool_spec: op.pool_spec.clone(),
        kind: PoolKindOp::Sum { count_include_pad: op.count_include_pad, normalize: op.normalize },
    }) as Box<dyn TypedOp>))
});

crate::register_wgpu_op!(SumPool, |source, node, op| {
    let facts = source.node_input_facts(node.id)?;
    if !wgpu_pool_supported(&op.pool_spec, facts[0]) {
        return Ok(None);
    }
    Ok(Some(Box::new(WgpuPool {
        pool_spec: op.pool_spec.clone(),
        kind: PoolKindOp::Sum { count_include_pad: op.count_include_pad, normalize: op.normalize },
    }) as Box<dyn TypedOp>))
});

pub fn link_translators() {}
