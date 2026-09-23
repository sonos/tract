use super::activation::activation_op;
use super::cpu::{
    self, build_expert_plan, build_q40_linear_expert_plan, build_router_plan,
    q40_direct_activation_supported,
};
use super::{
    ExpertLayout, MoeFfn, OptMoeFfn, RouteTopK, RoutedCombine, RoutedInputMode, RoutedMatMul,
    block_quant_group_tensor,
};
use std::sync::Arc;
use tract_nnef::internal::*;
use tract_nnef::tract_core::ops::{konst::Const, math::mul};
use tract_nnef::tract_core::tract_linalg::block_quant::BlockQuantStorage;

impl TypedOp for MoeFfn {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        self.validate_facts(inputs)?;
        let x_fact = inputs[0];
        let output_fact = x_fact.datum_type.fact(x_fact.shape.clone());
        Ok(tvec!(output_fact))
    }

    fn codegen(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        if let Some(patch) = self.lower_constant_experts(model, node)? {
            return Ok(Some(patch));
        }
        self.lower_routed_experts(model, node)
    }

    as_op!();
}

impl MoeFfn {
    fn lower_constant_experts(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let biased_or_clamped = self.has_any_bias() || self.is_clamped_act();
        let wg_const = model.node(node.inputs[1].node).op_as::<Const>();
        let w1_const = model.node(node.inputs[2].node).op_as::<Const>();
        let w2_const = model.node(node.inputs[3].node).op_as::<Const>();
        let w3_const =
            if self.has_w3 { model.node(node.inputs[4].node).op_as::<Const>() } else { None };
        if let (Some(wg_const), Some(w1_const), Some(w2_const)) = (wg_const, w1_const, w2_const)
            && (!self.has_w3 || w3_const.is_some())
        {
            let wg_tensor = wg_const.val().clone();
            let w1_tensor = w1_const.val().clone();
            let w2_tensor = w2_const.val().clone();
            let w3_tensor = w3_const.map(|c| c.val().clone());

            // Biases must be constants too, otherwise the subplans cannot bake
            // them in and the node has to stay on the reference evaluator.
            let idx = self.input_idx();
            let bias_tensor = |slot: Option<usize>| -> Option<Arc<Tensor>> {
                slot.and_then(|i| model.node(node.inputs[i].node).op_as::<Const>())
                    .map(|c| c.val().clone())
            };
            let wg_bias_tensor = bias_tensor(idx.wg_bias);
            let w1_bias_tensor = bias_tensor(idx.w1_bias);
            let w3_bias_tensor = bias_tensor(idx.w3_bias);
            let w2_bias_tensor = bias_tensor(idx.w2_bias);
            if (self.has_wg_bias && wg_bias_tensor.is_none())
                || (self.has_w1_bias && w1_bias_tensor.is_none())
                || (self.has_w3_bias && w3_bias_tensor.is_none())
                || (self.has_w2_bias && w2_bias_tensor.is_none())
            {
                return Ok(None);
            }

            let expert_tensors_are_plain = w1_tensor.is_plain()
                && w2_tensor.is_plain()
                && w3_tensor.as_ref().is_none_or(|t| t.is_plain());
            // Block-quantness is tracked per projection. Exports can mix
            // precisions across the three (gpt-oss ships Q40 gate/up with a
            // float down projection), and such a model must still reach the
            // subplan path: falling back to the reference evaluator costs
            // about two orders of magnitude in decode throughput.
            let w1_bq = w1_tensor.storage_as::<BlockQuantStorage>().is_some();
            let w2_bq = w2_tensor.storage_as::<BlockQuantStorage>().is_some();
            let w3_bq =
                w3_tensor.as_ref().is_none_or(|t| t.storage_as::<BlockQuantStorage>().is_some());
            let is_linear = self.expert_layout == ExpertLayout::Linear;
            // The direct routed Q40 plan packs all three projections, so it
            // needs all three block-quantized.
            let expert_tensors_are_linear_block_quant = is_linear && w1_bq && w2_bq && w3_bq;
            let expert_tensors_are_linear_mixed_quant = is_linear && (w1_bq || w2_bq || w3_bq);
            if !expert_tensors_are_plain && !expert_tensors_are_linear_mixed_quant {
                return Ok(None);
            }

            let num_experts = w1_tensor.shape()[0];
            let d_model = match self.expert_layout {
                ExpertLayout::Canonical => w1_tensor.shape()[1],
                ExpertLayout::Linear => w1_tensor.shape()[2],
            };
            let d_hidden = match self.expert_layout {
                ExpertLayout::Canonical => w1_tensor.shape()[2],
                ExpertLayout::Linear => w1_tensor.shape()[1],
            };

            let router_plan =
                build_router_plan(&wg_tensor, &model.symbols).context("Building router plan")?;

            let q40_linear_plan = if expert_tensors_are_linear_block_quant
                && !biased_or_clamped
                && q40_direct_activation_supported(&self.activation, self.has_w3)
            {
                Some(
                    build_q40_linear_expert_plan(
                        &w1_tensor,
                        &w2_tensor,
                        w3_tensor.as_deref(),
                        &self.activation,
                        num_experts,
                    )
                    .context("Building direct Q40 linear expert plan")?,
                )
            } else {
                None
            };

            // Fallback path only: each plan slices its own expert out of the
            // shared weight tensors and optimizes an independent sub-model,
            // which includes prepacking that expert's weights. Those builds do
            // not depend on each other, but run serially they serialize
            // layers x experts optimizations into model load (768 of them for
            // a 24-layer 32-expert model, ~62s). Spread them over rayon's
            // global pool, as FlashSdpa already does for its heads.
            let build_one = |eid: usize| -> TractResult<Arc<TypedSimplePlan>> {
                let w1_e = if w1_bq {
                    block_quant_group_tensor(&w1_tensor, eid)?
                } else {
                    match self.expert_layout {
                        ExpertLayout::Canonical => {
                            w1_tensor.slice(0, eid, eid + 1)?.into_shape(&[d_model, d_hidden])?
                        }
                        ExpertLayout::Linear => {
                            w1_tensor.slice(0, eid, eid + 1)?.into_shape(&[d_hidden, d_model])?
                        }
                    }
                };
                let w2_e = if w2_bq {
                    block_quant_group_tensor(&w2_tensor, eid)?
                } else {
                    match self.expert_layout {
                        ExpertLayout::Canonical => {
                            w2_tensor.slice(0, eid, eid + 1)?.into_shape(&[d_hidden, d_model])?
                        }
                        ExpertLayout::Linear => {
                            w2_tensor.slice(0, eid, eid + 1)?.into_shape(&[d_model, d_hidden])?
                        }
                    }
                };
                let w3_e = if let Some(ref w3) = w3_tensor {
                    Some(if w3_bq {
                        block_quant_group_tensor(w3, eid)?
                    } else {
                        match self.expert_layout {
                            ExpertLayout::Canonical => {
                                w3.slice(0, eid, eid + 1)?.into_shape(&[d_model, d_hidden])?
                            }
                            ExpertLayout::Linear => {
                                w3.slice(0, eid, eid + 1)?.into_shape(&[d_hidden, d_model])?
                            }
                        }
                    })
                } else {
                    None
                };

                let bias_e = |t: &Option<Arc<Tensor>>, dim: usize| {
                    t.as_ref().map(|b| b.slice(0, eid, eid + 1)?.into_shape(&[dim])).transpose()
                };
                let w1_bias_e = bias_e(&w1_bias_tensor, d_hidden)?;
                let w3_bias_e = bias_e(&w3_bias_tensor, d_hidden)?;
                let w2_bias_e = bias_e(&w2_bias_tensor, d_model)?;

                build_expert_plan(
                    &w1_e,
                    &w2_e,
                    w3_e.as_ref(),
                    w1_bias_e.as_ref(),
                    w3_bias_e.as_ref(),
                    w2_bias_e.as_ref(),
                    &self.activation,
                    self.act_alpha_bits.map(f32::from_bits),
                    self.act_limit_bits.map(f32::from_bits),
                    self.expert_layout,
                    expert_tensors_are_linear_mixed_quant,
                    &model.symbols,
                )
                .with_context(|| format!("Building expert plan for expert {eid}"))
            };

            let experts = if let Some(plan) = q40_linear_plan {
                cpu::CpuExpertPlan::PackedQ40(plan)
            } else {
                let plans = {
                    #[cfg(not(target_family = "wasm"))]
                    {
                        use rayon::prelude::*;
                        (0..num_experts)
                            .into_par_iter()
                            .map(build_one)
                            .collect::<TractResult<Vec<_>>>()?
                    }
                    #[cfg(target_family = "wasm")]
                    {
                        (0..num_experts).map(build_one).collect::<TractResult<Vec<_>>>()?
                    }
                };
                cpu::CpuExpertPlan::PerExpert(plans)
            };

            let opt_op = OptMoeFfn {
                k: self.k,
                gate: self.gate.clone(),
                num_experts,
                d_model,
                d_hidden,
                router_plan,
                wg_bias: wg_bias_tensor
                    .map(|b| b.cast_to::<f32>().map(|b| b.into_owned()))
                    .transpose()?,
                experts,
            };

            let mut patch = TypedModelPatch::default();
            let x_tap = patch.tap_model(model, node.inputs[0])?;
            let wires = patch.wire_node(&node.name, opt_op, &[x_tap])?;
            patch.shunt_outside(model, OutletId::new(node.id, 0), wires[0])?;
            return Ok(Some(patch));
        }

        Ok(None)
    }

    fn lower_routed_experts(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let biased_or_clamped = self.has_any_bias() || self.is_clamped_act();
        let act_op = activation_op(&self.activation, self.has_w3);
        if self.expert_layout == ExpertLayout::Linear {
            return Ok(None);
        }

        // The routed primitive lowering below models neither per-expert
        // biases nor the clamped activation: ops carrying either must stay
        // on the reference evaluator when their weights are not all consts.
        if biased_or_clamped {
            return Ok(None);
        }
        let Some(act_op) = act_op else {
            return Ok(None);
        };

        // Routed ops validate local bounds, not cross-projection constraints
        // such as equal expert counts or non-broadcasting gate/up widths.
        // Retain the reference runtime checks when weight shapes are unresolved.
        for &input in &node.inputs[1..] {
            if model.outlet_fact(input)?.shape.as_concrete().is_none() {
                return Ok(None);
            }
        }

        let expert_inputs: &[usize] = if self.has_w3 { &[2, 3, 4] } else { &[2, 3] };
        for &input_ix in expert_inputs {
            if let Some(konst) = model.node(node.inputs[input_ix].node).op_as::<Const>()
                && !konst.val().is_plain()
            {
                return Ok(None);
            }
        }
        let cache_weights = |input_ix: usize| {
            model
                .node(node.inputs[input_ix].node)
                .op_as::<Const>()
                .is_some_and(|konst| konst.val().is_plain())
        };
        let cache_w1 = cache_weights(2);
        let cache_w2 = cache_weights(3);
        let cache_w3 = self.has_w3 && cache_weights(4);

        let mut patch = TypedModelPatch::default();
        let x = patch.tap_model(model, node.inputs[0])?;
        let wg = patch.tap_model(model, node.inputs[1])?;
        let w1 = patch.tap_model(model, node.inputs[2])?;
        let w2 = patch.tap_model(model, node.inputs[3])?;

        let routes = patch.wire_node(
            format!("{}.route_topk", node.name),
            RouteTopK { k: self.k, gate: self.gate.clone() },
            &[x, wg],
        )?;
        let route_token_ids = routes[0];
        let route_expert_ids = routes[1];
        let route_weights = routes[2];

        let h1 = patch.wire_node(
            format!("{}.w1", node.name),
            RoutedMatMul { input_mode: RoutedInputMode::TokenRows, cache_weights: cache_w1 },
            &[x, w1, route_token_ids, route_expert_ids],
        )?[0];
        let activated = patch.wire_node(format!("{}.activation", node.name), act_op, &[h1])?[0];

        let hidden = if self.has_w3 {
            let w3 = patch.tap_model(model, node.inputs[4])?;
            let gate = patch.wire_node(
                format!("{}.w3", node.name),
                RoutedMatMul { input_mode: RoutedInputMode::TokenRows, cache_weights: cache_w3 },
                &[x, w3, route_token_ids, route_expert_ids],
            )?[0];
            patch.wire_node(format!("{}.swiglu_mul", node.name), mul(), &[activated, gate])?[0]
        } else {
            activated
        };

        let route_values = patch.wire_node(
            format!("{}.w2", node.name),
            RoutedMatMul { input_mode: RoutedInputMode::RouteRows, cache_weights: cache_w2 },
            &[hidden, w2, route_token_ids, route_expert_ids],
        )?[0];
        let output = patch.wire_node(
            format!("{}.combine", node.name),
            RoutedCombine,
            &[x, route_values, route_token_ids, route_weights],
        )?[0];
        patch.shunt_outside(model, OutletId::new(node.id, 0), output)?;
        Ok(Some(patch))
    }
}
