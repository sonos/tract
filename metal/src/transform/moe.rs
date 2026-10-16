use super::metal_cast_new;
use crate::{kernels, ops};
use tract_core::internal::*;
use tract_core::tract_linalg::block_quant::Q4_0;
use tract_gpu::sync::{DeviceSyncKind, sync_outlet_if_required};
use tract_gpu::utils::as_quant_fact;
use tract_transformers::ops::moe_ffn::{ExpertLayout, MoeFfn, RoutedInputMode};

fn sync_f32_input(
    target: &mut TypedModel,
    name: impl Into<String>,
    outlet: OutletId,
    fact: &TypedFact,
) -> TractResult<OutletId> {
    let name = name.into();
    let device = sync_outlet_if_required(
        target,
        format!("{name}.to-device"),
        outlet,
        DeviceSyncKind::ToDevice,
    )?;
    if fact.datum_type == f32::datum_type() {
        Ok(device)
    } else {
        Ok(target.wire_node(
            format!("{name}.cast-f32"),
            metal_cast_new(f32::datum_type()).context("unsupported Metal cast")?,
            &[device],
        )?[0])
    }
}

fn add_routed_bias(
    target: &mut TypedModel,
    name: impl Into<String>,
    value: OutletId,
    bias: OutletId,
    route_expert_ids: OutletId,
) -> TractResult<OutletId> {
    let name = name.into();
    Ok(target.wire_node(
        format!("{name}.bias-add"),
        ops::MetalRoutedBiasAdd,
        &[value, bias, route_expert_ids],
    )?[0])
}

/// Lower supported linear Q40 MoE blocks to device routing, fused expert
/// projections and weighted combination, preserving CPU fallback otherwise.
pub(super) fn convert_q40_moe_ffn_to_metal(
    source: &TypedModel,
    node: &TypedNode,
    target: &mut TypedModel,
    mapping: &HashMap<OutletId, OutletId>,
    op: &MoeFfn,
) -> TractResult<Option<TVec<OutletId>>> {
    if op.expert_layout != ExpertLayout::Linear
        || !q40_moe_activation_supported(op)
        || (op.act_limit_bits.is_some() && !op.has_w3)
    {
        return Ok(None);
    }
    let facts = source.node_input_facts(node.id)?;
    let indexes = moe_input_indexes(op);
    if !q40_moe_facts_supported(op, &facts, ExpertLayout::Linear) {
        return Ok(None);
    }
    log::trace!("Metal Q40 MoE lowering: {}", node.name);

    // has_w3 blocks fuse w1/w3/(biases)/activation into one op: a single
    // gemv-pair dispatch at decode, one shared expert sort + gather at
    // prefill. Both bias inputs must be present or absent together for the
    // fused kernel; mixed-bias models fall back to the unfused chain.
    let fused_swiglu = op.has_w3 && indexes.w1_bias.is_some() == indexes.w3_bias.is_some();

    let x_device = sync_outlet_if_required(
        target,
        format!("{}.x.to-device", node.name),
        mapping[&node.inputs[0]],
        DeviceSyncKind::ToDevice,
    )?;
    let f16_x_direct = facts[0].datum_type == DatumType::F16 && fused_swiglu;
    let x_device = if facts[0].datum_type != f32::datum_type() && !f16_x_direct {
        target.wire_node(
            format!("{}.x.cast-f32", node.name),
            metal_cast_new(f32::datum_type()).context("unsupported Metal cast")?,
            &[x_device],
        )?[0]
    } else {
        x_device
    };
    let wg_device = sync_outlet_if_required(
        target,
        format!("{}.wg.to-device", node.name),
        mapping[&node.inputs[1]],
        DeviceSyncKind::ToDevice,
    )?;
    let wg_device = if facts[1].datum_type != f32::datum_type() {
        target.wire_node(
            format!("{}.wg.cast-f32", node.name),
            metal_cast_new(f32::datum_type()).context("unsupported Metal cast")?,
            &[wg_device],
        )?[0]
    } else {
        wg_device
    };
    let wg_bias_device = indexes
        .wg_bias
        .map(|ix| {
            sync_f32_input(
                target,
                format!("{}.wg_bias", node.name),
                mapping[&node.inputs[ix]],
                facts[ix],
            )
        })
        .transpose()?;

    let routes = {
        let mut route_inputs = tvec![x_device, wg_device];
        if let Some(wg_bias_device) = wg_bias_device {
            route_inputs.push(wg_bias_device);
        }
        target.wire_node(
            format!("{}.route_topk", node.name),
            ops::MetalRouteTopK { k: op.k, gate: op.gate.clone() },
            &route_inputs,
        )?
    };
    let route_token_ids = sync_outlet_if_required(
        target,
        format!("{}.route_token_ids.to-device", node.name),
        routes[0],
        DeviceSyncKind::ToDevice,
    )?;
    let route_expert_ids = sync_outlet_if_required(
        target,
        format!("{}.route_expert_ids.to-device", node.name),
        routes[1],
        DeviceSyncKind::ToDevice,
    )?;
    let route_weights = sync_outlet_if_required(
        target,
        format!("{}.route_weights.to-device", node.name),
        routes[2],
        DeviceSyncKind::ToDevice,
    )?;

    let x_shape_like = x_device;
    let x_expert_input = if facts[0].rank() == 3 {
        target.wire_node(
            format!("{}.x.rm-batch", node.name),
            tract_gpu::ops::change_axes::GpuAxisOp::new(AxisOp::Rm(0)),
            &[x_device],
        )?[0]
    } else {
        x_device
    };
    let w1_device = sync_outlet_if_required(
        target,
        format!("{}.w1.to-device", node.name),
        mapping[&node.inputs[2]],
        DeviceSyncKind::ToDevice,
    )?;
    let w2_device = sync_outlet_if_required(
        target,
        format!("{}.w2.to-device", node.name),
        mapping[&node.inputs[3]],
        DeviceSyncKind::ToDevice,
    )?;
    let w1_bias_device = indexes
        .w1_bias
        .map(|ix| {
            sync_f32_input(
                target,
                format!("{}.w1_bias", node.name),
                mapping[&node.inputs[ix]],
                facts[ix],
            )
        })
        .transpose()?;
    let w3_bias_device = indexes
        .w3_bias
        .map(|ix| {
            sync_f32_input(
                target,
                format!("{}.w3_bias", node.name),
                mapping[&node.inputs[ix]],
                facts[ix],
            )
        })
        .transpose()?;
    let w2_bias_device = indexes
        .w2_bias
        .map(|ix| {
            sync_f32_input(
                target,
                format!("{}.w2_bias", node.name),
                mapping[&node.inputs[ix]],
                facts[ix],
            )
        })
        .transpose()?;

    // fused_swiglu computed above, next to the f16-x decision it gates
    // (w1_bias/w3_bias devices map 1:1 from the indexes used there).
    let hidden = if fused_swiglu {
        let w3_device = sync_outlet_if_required(
            target,
            format!("{}.w3.to-device", node.name),
            mapping[&node.inputs[indexes.w3.context("missing w3 input index")?]],
            DeviceSyncKind::ToDevice,
        )?;
        let has_bias = w1_bias_device.is_some();
        let mut inputs =
            tvec![x_expert_input, w1_device, w3_device, route_token_ids, route_expert_ids];
        if let (Some(b1), Some(b3)) = (w1_bias_device, w3_bias_device) {
            inputs.push(b1);
            inputs.push(b3);
        }
        target.wire_node(
            format!("{}.w1w3_swiglu", node.name),
            ops::MetalRoutedQ40SwiGlu {
                input_mode: RoutedInputMode::TokenRows,
                activation: match op.act_limit_bits {
                    Some(limit_bits) => crate::ops::routed_q40_swiglu::RoutedActivation::Clamped {
                        alpha_bits: op.act_alpha_bits.unwrap_or(1.0f32.to_bits()),
                        limit_bits,
                    },
                    None => crate::ops::routed_q40_swiglu::RoutedActivation::Plain,
                },
                has_bias,
            },
            &inputs,
        )?[0]
    } else {
        let mut h1 = target.wire_node(
            format!("{}.w1", node.name),
            ops::MetalRoutedQ40MatMul { input_mode: RoutedInputMode::TokenRows },
            &[x_expert_input, w1_device, route_token_ids, route_expert_ids],
        )?[0];
        if let Some(w1_bias_device) = w1_bias_device {
            h1 = add_routed_bias(
                target,
                format!("{}.w1", node.name),
                h1,
                w1_bias_device,
                route_expert_ids,
            )?;
        }

        if op.has_w3 {
            let w3_device = sync_outlet_if_required(
                target,
                format!("{}.w3.to-device", node.name),
                mapping[&node.inputs[indexes.w3.context("missing w3 input index")?]],
                DeviceSyncKind::ToDevice,
            )?;
            let mut up = target.wire_node(
                format!("{}.w3", node.name),
                ops::MetalRoutedQ40MatMul { input_mode: RoutedInputMode::TokenRows },
                &[x_expert_input, w3_device, route_token_ids, route_expert_ids],
            )?[0];
            if let Some(w3_bias_device) = w3_bias_device {
                up = add_routed_bias(
                    target,
                    format!("{}.w3", node.name),
                    up,
                    w3_bias_device,
                    route_expert_ids,
                )?;
            }
            if let Some(limit_bits) = op.act_limit_bits {
                let alpha = op.act_alpha_bits.map(f32::from_bits).unwrap_or(1.0);
                target.wire_node(
                    format!("{}.clamped_swiglu", node.name),
                    ops::MetalClampedSwiGlu::new(alpha, f32::from_bits(limit_bits)),
                    &[h1, up],
                )?[0]
            } else {
                let activated = target.wire_node(
                    format!("{}.activation", node.name),
                    kernels::element_wise::metal_element_wise_op(Box::new(
                        tract_core::ops::nn::Silu {},
                    )),
                    &[h1],
                )?[0];
                target.wire_node(
                    format!("{}.swiglu_mul", node.name),
                    kernels::bin_ops::metal_bin_op(Box::new(tract_core::ops::math::Mul)),
                    &[activated, up],
                )?[0]
            }
        } else {
            target.wire_node(
                format!("{}.activation", node.name),
                kernels::element_wise::metal_element_wise_op(Box::new(
                    tract_core::ops::nn::Silu {},
                )),
                &[h1],
            )?[0]
        }
    };

    let mut route_values = target.wire_node(
        format!("{}.w2", node.name),
        ops::MetalRoutedQ40MatMul { input_mode: RoutedInputMode::RouteRows },
        &[hidden, w2_device, route_token_ids, route_expert_ids],
    )?[0];
    if let Some(w2_bias_device) = w2_bias_device {
        route_values = add_routed_bias(
            target,
            format!("{}.w2", node.name),
            route_values,
            w2_bias_device,
            route_expert_ids,
        )?;
    }
    let output = target.wire_node(
        format!("{}.combine", node.name),
        ops::MetalRoutedCombine { routes_per_token: op.k as u32 },
        &[x_shape_like, route_values, route_token_ids, route_weights],
    )?[0];
    let output = if facts[0].datum_type != f32::datum_type() {
        target.wire_node(
            format!("{}.out.cast", node.name),
            metal_cast_new(facts[0].datum_type).context("unsupported Metal cast")?,
            &[output],
        )?[0]
    } else {
        output
    };

    Ok(Some(tvec![output]))
}

fn q40_moe_activation_supported(op: &MoeFfn) -> bool {
    matches!(op.activation.as_str(), "silu")
        || (op.has_w3 && matches!(op.activation.as_str(), "swiglu"))
}

#[derive(Default)]
struct MoeInputIndexes {
    w3: Option<usize>,
    wg_bias: Option<usize>,
    w1_bias: Option<usize>,
    w3_bias: Option<usize>,
    w2_bias: Option<usize>,
}

fn moe_input_indexes(op: &MoeFfn) -> MoeInputIndexes {
    let mut next = 4;
    let mut take = |present: bool| {
        if present {
            let ix = next;
            next += 1;
            Some(ix)
        } else {
            None
        }
    };
    MoeInputIndexes {
        w3: take(op.has_w3),
        wg_bias: take(op.has_wg_bias),
        w1_bias: take(op.has_w1_bias),
        w3_bias: take(op.has_w3_bias),
        w2_bias: take(op.has_w2_bias),
    }
}

fn fact_is_f16_or_f32(fact: &TypedFact) -> bool {
    matches!(fact.datum_type, DatumType::F16 | DatumType::F32)
}

/// Checks the layout, dtypes and routing limits supported by Metal Q40 MoE lowering.
fn q40_moe_facts_supported(op: &MoeFfn, facts: &[&TypedFact], layout: ExpertLayout) -> bool {
    let indexes = moe_input_indexes(op);
    let x_rank_ok =
        facts[0].rank() == 2 || (facts[0].rank() == 3 && facts[0].shape.dims()[0] == 1.to_dim());
    let x_dt_ok = fact_is_f16_or_f32(facts[0]);
    let wg_dt_ok = fact_is_f16_or_f32(facts[1]);
    let w1_q40 = as_quant_fact(facts[2], &Q4_0).is_some();
    let w2_q40 = as_quant_fact(facts[3], &Q4_0).is_some();
    let w3_q40 =
        !op.has_w3 || indexes.w3.is_some_and(|ix| as_quant_fact(facts[ix], &Q4_0).is_some());
    if facts[2].rank() != 3 {
        return false;
    }
    let num_experts = facts[2].shape[0].clone();
    let (d_hidden, d_model) = match layout {
        ExpertLayout::Linear => (facts[2].shape[1].clone(), facts[2].shape[2].clone()),
        ExpertLayout::Canonical => (facts[2].shape[2].clone(), facts[2].shape[1].clone()),
    };
    let wg_bias_ok = indexes.wg_bias.is_none_or(|ix| {
        fact_is_f16_or_f32(facts[ix]) && facts[ix].rank() == 1 && facts[ix].shape[0] == num_experts
    });
    let w1_bias_ok = indexes.w1_bias.is_none_or(|ix| {
        fact_is_f16_or_f32(facts[ix])
            && facts[ix].rank() == 2
            && facts[ix].shape[0] == num_experts
            && facts[ix].shape[1] == d_hidden
    });
    let w3_bias_ok = indexes.w3_bias.is_none_or(|ix| {
        fact_is_f16_or_f32(facts[ix])
            && facts[ix].rank() == 2
            && facts[ix].shape[0] == num_experts
            && facts[ix].shape[1] == d_hidden
    });
    let w2_bias_ok = indexes.w2_bias.is_none_or(|ix| {
        fact_is_f16_or_f32(facts[ix])
            && facts[ix].rank() == 2
            && facts[ix].shape[0] == num_experts
            && facts[ix].shape[1] == d_model
    });
    // The Metal route-topk kernel scores at most 256 experts per token.
    let experts_ok = num_experts
        .as_i64()
        .is_some_and(|e| (1..=256).contains(&e) && op.k > 0 && op.k <= 16 && op.k <= e as usize);
    x_rank_ok
        && x_dt_ok
        && wg_dt_ok
        && w1_q40
        && w2_q40
        && w3_q40
        && experts_ok
        && wg_bias_ok
        && w1_bias_ok
        && w3_bias_ok
        && w2_bias_ok
}
