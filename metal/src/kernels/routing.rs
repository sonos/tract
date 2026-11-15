use crate::kernels::matmul::RoutedQ40InputMode;
use crate::kernels::matmul::RoutedSwigluAct;
use crate::kernels::matmul::dispatch_routed_f32;
use crate::kernels::matmul::dispatch_routed_swiglu_f32;
use crate::kernels::matmul::routed_pair_supported;
use crate::kernels::matmul::routed_weights_supported;
use crate::kernels::moe::dispatch_route_select_f32;
use crate::kernels::moe::dispatch_routed_combine;
use crate::kernels::moe::dispatch_routed_postamble_f32;
use tract_core::internal::*;
use tract_gpu::ops::routing::FusePairPolicy;
use tract_gpu::ops::routing::GpuRouteSelect;
use tract_gpu::ops::routing::GpuRoutedCombine;
use tract_gpu::ops::routing::GpuRoutedLinear;
use tract_gpu::ops::routing::RoutedLinearKernels;
use tract_gpu::ops::routing::RoutingKernels;
use tract_gpu::tensor::DeviceTensor;
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
    supports: routed_weights_supported,
    project: |input, weights, rows, expert_ids, out| {
        crate::with_metal_stream(|stream| {
            let (token_ids, mode) = input_mode(rows, expert_ids);
            dispatch_routed_f32(stream, input, weights, token_ids, expert_ids, mode, out)?;
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
    fused_pair: Some((
        FusePairPolicy::Always,
        |input, weights, weights_up, biases, rows, expert_ids, activation, out| {
            let act = match activation {
                (1, _, _) => RoutedSwigluAct::Plain,
                (4, alpha, limit) => RoutedSwigluAct::Clamped { alpha, limit },
                _ => return Ok(false),
            };
            if !routed_pair_supported(weights) {
                return Ok(false);
            }
            crate::with_metal_stream(|stream| {
                let (token_ids, mode) = input_mode(rows, expert_ids);
                dispatch_routed_swiglu_f32(
                    stream, input, weights, weights_up, biases, token_ids, expert_ids, mode, act,
                    out,
                )?;
                commit_after_many_routes(stream, expert_ids.len())
            })?;
            Ok(true)
        },
    )),
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
    tract_gpu::ops::routing::lower_moe_ffn(model, node, op, routed_weights_supported)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{MetalRuntime, MetalTransform};
    use tract_core::internal::DefaultRuntime;
    use tract_core::ops::konst::Const;
    use tract_core::tract_linalg::block_quant::{
        BlockQuant, BlockQuantFact, BlockQuantStorage, Q4_0,
    };
    use tract_core::transform::ModelTransform;
    use tract_transformers::ops::moe_ffn::{ExpertLayout, GateMode, MoeFfn};

    fn add_q40_const(
        model: &mut TypedModel,
        name: &str,
        shape: &[usize],
        data: &[f32],
    ) -> TractResult<OutletId> {
        let k = *shape.last().context("Q40 tensor has no last axis")?;
        ensure!(k % Q4_0.block_len() == 0);
        let rows = shape[..shape.len() - 1].iter().product::<usize>();
        let quant = Q4_0.quant_f32(data)?;
        let storage = BlockQuantStorage::new(Box::new(Q4_0), rows, k, Arc::new(quant))?;
        let tensor = Arc::new(storage.into_tensor_with_shape(f32::datum_type(), shape));
        let fact = BlockQuantFact::new(Box::new(Q4_0), shape.iter().copied().collect());
        Ok(model.wire_node(name, Const::new_with_exotic_fact(tensor, Box::new(fact))?, &[])?[0])
    }

    fn tensor_from_f32(shape: &[usize], data: &[f32], dt: DatumType) -> TractResult<Tensor> {
        match dt {
            DatumType::F32 => Tensor::from_shape(shape, data),
            DatumType::F16 => {
                let data = data.iter().map(|&v| f16::from_f32(v)).collect::<Vec<_>>();
                Tensor::from_shape(shape, &data)
            }
            _ => bail!("unsupported test datum type: {dt:?}"),
        }
    }

    fn check_q40_moe_ffn_lowers_to_metal_primitives(
        rank3_input: bool,
        input_dt: DatumType,
        router_dt: DatumType,
    ) -> TractResult<()> {
        let experts = 4;
        let tokens = 3;
        let d_model = 32;
        let d_hidden = 64;
        let input_shape =
            if rank3_input { vec![1, tokens, d_model] } else { vec![tokens, d_model] };
        let mut rng_state: u64 = 20260702;
        let mut next_f32 = || -> f32 {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng_state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        };
        let input_data = (0..tokens * d_model).map(|_| next_f32()).collect::<Vec<_>>();
        let wg_data = (0..experts * d_model).map(|_| next_f32()).collect::<Vec<_>>();
        let w1_data = (0..experts * d_hidden * d_model).map(|_| next_f32()).collect::<Vec<_>>();
        let w2_data = (0..experts * d_model * d_hidden).map(|_| next_f32()).collect::<Vec<_>>();
        let w3_data = (0..experts * d_hidden * d_model).map(|_| next_f32()).collect::<Vec<_>>();

        let mut model = TypedModel::default();
        let x = model.add_source("x", input_dt.fact(&input_shape))?;
        let wg =
            model.add_const("wg", tensor_from_f32(&[experts, d_model], &wg_data, router_dt)?)?;
        let w1 = add_q40_const(&mut model, "w1", &[experts, d_hidden, d_model], &w1_data)?;
        let w2 = add_q40_const(&mut model, "w2", &[experts, d_model, d_hidden], &w2_data)?;
        let w3 = add_q40_const(&mut model, "w3", &[experts, d_hidden, d_model], &w3_data)?;
        let op = MoeFfn {
            k: 2,
            activation: "silu".to_string(),
            gate: GateMode::SoftmaxTopk,
            has_w3: true,
            has_wg_bias: false,
            has_w1_bias: false,
            has_w3_bias: false,
            has_w2_bias: false,
            act_alpha_bits: None,
            act_limit_bits: None,
            expert_layout: ExpertLayout::Linear,
        };
        let y = model.wire_node("moe", op, &[x, wg, w1, w2, w3])?;
        model.select_output_outlets(&y)?;

        let mut transformed = model.clone();
        MetalTransform::default().transform(&mut transformed)?;
        check_lowered_to_shared_routing_ops(&transformed)?;

        let input = tensor_from_f32(&input_shape, &input_data, input_dt)?;
        let expected =
            DefaultRuntime.prepare(model.clone())?.run(tvec![input.clone().into_tvalue()])?;
        let actual = MetalRuntime.prepare(model)?.run(tvec![input.into_tvalue()])?;
        actual[0]
            .clone()
            .into_tensor()
            .close_enough(&expected[0].clone().into_tensor(), Approximation::Approximate)
    }

    fn check_lowered_to_shared_routing_ops(transformed: &TypedModel) -> TractResult<()> {
        let count =
            |name: &str| transformed.nodes().iter().filter(|n| n.op().name() == name).count();
        let ops = || {
            transformed
                .nodes()
                .iter()
                .map(|node| format!("{}: {}", node.name, node.op().name()))
                .collect::<Vec<_>>()
                .join("\n")
        };
        for (name, expected) in
            [("MetalRouteSelect", 1), ("MetalRoutedLinear", 2), ("MetalRoutedCombine", 1)]
        {
            ensure!(count(name) == expected, "expected {expected} {name}\n{}", ops());
        }
        Ok(())
    }

    #[test]
    fn graph_q40_moe_ffn_lowers_to_metal_primitives() -> TractResult<()> {
        check_q40_moe_ffn_lowers_to_metal_primitives(false, DatumType::F32, DatumType::F32)
    }

    #[test]
    fn graph_q40_moe_ffn_lowers_rank3_f16_to_metal_primitives() -> TractResult<()> {
        check_q40_moe_ffn_lowers_to_metal_primitives(true, DatumType::F16, DatumType::F16)
    }

    #[test]
    fn graph_gpt_oss_q40_moe_ffn_lowers_to_metal_primitives() -> TractResult<()> {
        let experts = 4;
        let tokens = 3;
        let d_model = 32;
        let d_hidden = 64;
        let input_shape = vec![tokens, d_model];
        let mut rng_state: u64 = 20260709;
        let mut next_f32 = || -> f32 {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng_state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        };
        let input_data = (0..tokens * d_model).map(|_| next_f32()).collect::<Vec<_>>();
        let wg_data = (0..experts * d_model).map(|_| next_f32()).collect::<Vec<_>>();
        let w1_data = (0..experts * d_hidden * d_model).map(|_| next_f32()).collect::<Vec<_>>();
        let w2_data = (0..experts * d_model * d_hidden).map(|_| next_f32()).collect::<Vec<_>>();
        let w3_data = (0..experts * d_hidden * d_model).map(|_| next_f32()).collect::<Vec<_>>();
        let wg_bias_data = (0..experts).map(|_| next_f32()).collect::<Vec<_>>();
        let w1_bias_data = (0..experts * d_hidden).map(|_| next_f32()).collect::<Vec<_>>();
        let w3_bias_data = (0..experts * d_hidden).map(|_| next_f32()).collect::<Vec<_>>();
        let w2_bias_data = (0..experts * d_model).map(|_| next_f32()).collect::<Vec<_>>();

        let mut model = TypedModel::default();
        let x = model.add_source("x", f16::datum_type().fact(&input_shape))?;
        let wg = model
            .add_const("wg", tensor_from_f32(&[experts, d_model], &wg_data, DatumType::F16)?)?;
        let w1 = add_q40_const(&mut model, "w1", &[experts, d_hidden, d_model], &w1_data)?;
        let w2 = add_q40_const(&mut model, "w2", &[experts, d_model, d_hidden], &w2_data)?;
        let w3 = add_q40_const(&mut model, "w3", &[experts, d_hidden, d_model], &w3_data)?;
        let wg_bias = model
            .add_const("wg_bias", tensor_from_f32(&[experts], &wg_bias_data, DatumType::F16)?)?;
        let w1_bias = model.add_const(
            "w1_bias",
            tensor_from_f32(&[experts, d_hidden], &w1_bias_data, DatumType::F16)?,
        )?;
        let w3_bias = model.add_const(
            "w3_bias",
            tensor_from_f32(&[experts, d_hidden], &w3_bias_data, DatumType::F16)?,
        )?;
        let w2_bias = model.add_const(
            "w2_bias",
            tensor_from_f32(&[experts, d_model], &w2_bias_data, DatumType::F16)?,
        )?;
        let op = MoeFfn {
            k: 2,
            activation: "swiglu".to_string(),
            gate: GateMode::SoftmaxTopk,
            has_w3: true,
            has_wg_bias: true,
            has_w1_bias: true,
            has_w3_bias: true,
            has_w2_bias: true,
            act_alpha_bits: Some(1.702f32.to_bits()),
            act_limit_bits: Some(7.0f32.to_bits()),
            expert_layout: ExpertLayout::Linear,
        };
        let y =
            model.wire_node("moe", op, &[x, wg, w1, w2, w3, wg_bias, w1_bias, w3_bias, w2_bias])?;
        model.select_output_outlets(&y)?;

        let mut transformed = model.clone();
        MetalTransform::default().transform(&mut transformed)?;
        check_lowered_to_shared_routing_ops(&transformed)?;

        let input = tensor_from_f32(&input_shape, &input_data, DatumType::F16)?;
        let expected =
            DefaultRuntime.prepare(model.clone())?.run(tvec![input.clone().into_tvalue()])?;
        let actual = MetalRuntime.prepare(model)?.run(tvec![input.into_tvalue()])?;
        actual[0]
            .clone()
            .into_tensor()
            .close_enough(&expected[0].clone().into_tensor(), Approximation::Approximate)
    }
}
