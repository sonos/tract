use crate::kernels::nn::pool::{PoolKind, dispatch_metal_pool, metal_pool_supported};
use tract_core::internal::*;
use tract_core::ops::cnn::{MaxPool, OptMaxPool, OptSumPool, PoolSpec, SumPool};
use tract_gpu::tensor::DeviceTensorExt;

/// Metal device op for `OptMaxPool` / `OptSumPool` over a channels-last tensor.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MetalPool {
    pub pool_spec: PoolSpec,
    pub kind: PoolKindOp,
}

/// `PoolKind` without the borrowed geometry, so the op stays hashable.
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

impl Op for MetalPool {
    fn name(&self) -> StaticName {
        match self.kind {
            PoolKindOp::Max => "MetalMaxPool".into(),
            PoolKindOp::Sum { .. } => "MetalSumPool".into(),
        }
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(self.pool_spec.info())
    }

    op_as_typed_op!();
}

impl EvalOp for MetalPool {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        node_id: usize,
        session: &TurnState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let input = inputs[0].to_device_tensor()?;
        let output_shape = self.pool_spec.output_shape(input.shape())?;
        let output = tract_gpu::session_handler::make_tensor_for_node(
            session,
            node_id,
            input.datum_type(),
            &output_shape.shape,
        )?;
        if output.len() > 0 {
            crate::with_metal_stream(|stream| {
                dispatch_metal_pool(stream, &self.pool_spec, self.kind.into(), input, &output)
            })?;
        }
        Ok(tvec!(output.into_tensor().into_tvalue()))
    }
}

impl TypedOp for MetalPool {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        tract_gpu::utils::facts_to_device_facts(inputs, |facts| {
            let shape = self.pool_spec.output_shape(&facts[0].shape)?;
            Ok(tvec!(facts[0].datum_type.fact(shape.shape)))
        })
        .with_context(|| "Error while computing facts for MetalPool")
    }
}

crate::register_metal_op!(OptMaxPool, |source, node, op| {
    let facts = source.node_input_facts(node.id)?;
    // an index output would need a second buffer the kernel does not write
    if op.with_index_outputs.is_some()
        || !metal_pool_supported(&op.pool_spec, PoolKind::Max, facts[0])
    {
        return Ok(None);
    }
    Ok(Some(Box::new(MetalPool { pool_spec: op.pool_spec.clone(), kind: PoolKindOp::Max })
        as Box<dyn TypedOp>))
});

crate::register_metal_op!(OptSumPool, |source, node, op| {
    let facts = source.node_input_facts(node.id)?;
    let kind = PoolKind::Sum { count_include_pad: op.count_include_pad, normalize: op.normalize };
    if !metal_pool_supported(&op.pool_spec, kind, facts[0]) {
        return Ok(None);
    }
    Ok(Some(Box::new(MetalPool {
        pool_spec: op.pool_spec.clone(),
        kind: PoolKindOp::Sum { count_include_pad: op.count_include_pad, normalize: op.normalize },
    }) as Box<dyn TypedOp>))
});

// The metal transform runs before optimization, so a model still carries the
// unoptimized pools; the Opt* forms are registered too for callers that
// optimize first.
crate::register_metal_op!(MaxPool, |source, node, op| {
    let facts = source.node_input_facts(node.id)?;
    if op.with_index_outputs.is_some()
        || !metal_pool_supported(&op.pool_spec, PoolKind::Max, facts[0])
    {
        return Ok(None);
    }
    Ok(Some(Box::new(MetalPool { pool_spec: op.pool_spec.clone(), kind: PoolKindOp::Max })
        as Box<dyn TypedOp>))
});

crate::register_metal_op!(SumPool, |source, node, op| {
    let facts = source.node_input_facts(node.id)?;
    let kind = PoolKind::Sum { count_include_pad: op.count_include_pad, normalize: op.normalize };
    if !metal_pool_supported(&op.pool_spec, kind, facts[0]) {
        return Ok(None);
    }
    Ok(Some(Box::new(MetalPool {
        pool_spec: op.pool_spec.clone(),
        kind: PoolKindOp::Sum { count_include_pad: op.count_include_pad, normalize: op.normalize },
    }) as Box<dyn TypedOp>))
});

/// Referenced from the transform so the registrations below survive linking.
pub fn link_translators() {}
