use crate::kernels::matmul::RoutedQ40InputMode;
use crate::kernels::matmul::RoutedSwigluAct;
use crate::kernels::matmul::dispatch_routed_q40_f32;
use crate::kernels::matmul::dispatch_routed_q40_swiglu_f32;
use crate::kernels::moe::dispatch_route_select_f32;
use crate::kernels::moe::dispatch_routed_combine;
use crate::kernels::moe::dispatch_routed_postamble_f32;
use tract_core::internal::*;
use tract_core::tract_linalg::block_quant::Q4_0;
use tract_gpu::ops::routing::GpuRouteSelect;
use tract_gpu::ops::routing::GpuRoutedCombine;
use tract_gpu::ops::routing::GpuRoutedLinear;
use tract_gpu::ops::routing::RoutedLinearKernels;
use tract_gpu::ops::routing::RoutingKernels;
use tract_gpu::tensor::DeviceTensor;
use tract_gpu::utils::as_quant_fact;
use tract_transformers::ops::moe_ffn::MoeFfn;
use tract_transformers::ops::moe_ffn::RouteSelect;
use tract_transformers::ops::moe_ffn::RoutedCombine;
use tract_transformers::ops::moe_ffn::RoutedLinear;

const METAL_ROUTING_KERNELS: RoutingKernels = RoutingKernels {
    select: |scores, bias, k, gate, token_ids, expert_ids, weights| {
        crate::with_metal_stream(|stream| {
            dispatch_route_select_f32(stream, scores, bias, k, gate, token_ids, expert_ids, weights)
        })
    },
    combine: |values, weights, k, out| {
        crate::with_metal_stream(|stream| dispatch_routed_combine(stream, values, weights, k, out))
    },
};

crate::register_metal_op!(RouteSelect, |source, node, op| {
    Ok(GpuRouteSelect::from_tract_core(source, node, op, "Metal", METAL_ROUTING_KERNELS)?
        .map(|op| Box::new(op) as _))
});

crate::register_metal_op!(RoutedCombine, |source, node, op| {
    Ok(GpuRoutedCombine::from_tract_core(source, node, op, "Metal", METAL_ROUTING_KERNELS)?
        .map(|op| Box::new(op) as _))
});

/// Expert weights the routed kernels read: Q4_0, `[E, N, K]`.
fn supports(fact: &TypedFact) -> bool {
    fact.rank() == 3 && as_quant_fact(fact, &Q4_0).is_some()
}

/// The kernels' row source: `rows` when routes read token rows, else the route's own row, for
/// which the kernels never read `token_ids`.
fn input_mode<'a>(
    rows: Option<&'a DeviceTensor>,
    expert_ids: &'a DeviceTensor,
) -> (&'a DeviceTensor, RoutedQ40InputMode) {
    match rows {
        Some(rows) => (rows, RoutedQ40InputMode::TokenRows),
        None => (expert_ids, RoutedQ40InputMode::RouteRows),
    }
}

/// Submits the command buffer after a routed projection over many routes, so the GPU starts on
/// the expert matmuls while the rest of the layer encodes.
fn commit_after_many_routes(stream: &crate::MetalStream, routes: usize) -> TractResult<()> {
    if routes > crate::tuning::MOE_COMMIT_MIN_ROUTES {
        stream.commit_current()?;
    }
    Ok(())
}

const METAL_ROUTED_LINEAR_KERNELS: RoutedLinearKernels = RoutedLinearKernels {
    supports,
    project: |input, weights, rows, expert_ids, out| {
        crate::with_metal_stream(|stream| {
            let (token_ids, mode) = input_mode(rows, expert_ids);
            dispatch_routed_q40_f32(stream, input, weights, token_ids, expert_ids, mode, out)?;
            commit_after_many_routes(stream, expert_ids.len())
        })
    },
    postamble: |gate, up, bias, bias_up, expert_ids, activation, out| {
        crate::with_metal_stream(|stream| {
            dispatch_routed_postamble_f32(
                stream, gate, up, bias, bias_up, expert_ids, activation, out,
            )
        })
    },
    fused_pair: Some(|input, weights, weights_up, biases, rows, expert_ids, activation, out| {
        let act = match activation {
            (1, _, _) => RoutedSwigluAct::Plain,
            (4, alpha, limit) => RoutedSwigluAct::Clamped { alpha, limit },
            _ => return Ok(false),
        };
        crate::with_metal_stream(|stream| {
            let (token_ids, mode) = input_mode(rows, expert_ids);
            dispatch_routed_q40_swiglu_f32(
                stream, input, weights, weights_up, biases, token_ids, expert_ids, mode, act, out,
            )?;
            commit_after_many_routes(stream, expert_ids.len())
        })?;
        Ok(true)
    }),
};

crate::register_metal_op!(RoutedLinear, |source, node, op| {
    Ok(GpuRoutedLinear::from_tract_core(source, node, op, "Metal", METAL_ROUTED_LINEAR_KERNELS)?
        .map(|op| Box::new(op) as _))
});

/// [`tract_gpu::ops::routing::lower_moe_ffn`] for the weights Metal projects.
pub fn lower_moe_ffn(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    _name: &str,
    op: &MoeFfn,
) -> TractResult<Option<TypedModelPatch>> {
    tract_gpu::ops::routing::lower_moe_ffn(model, node, op, supports)
}
