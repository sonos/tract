use std::str::FromStr;

use tract_core::ops::array::{MultiBroadcastTo, TypedConcat};
use tract_core::ops::cast::Cast;
use tract_core::ops::einsum::EinSum;
use tract_core::ops::source::TypedSource;
use tract_core::ops::{change_axes, math};
use tract_nnef::internal::*;
use tract_nnef::ser::datum_type;
use tract_nnef::tract_core::ops::nn::{Softmax, SoftmaxExp, SoftmaxKind};
use tract_nnef::tract_ndarray::Array5;

use crate::ops::dyn_kv_cache::DynKeyValueCache;
use crate::ops::flash_sdpa::FlashSdpaOp;

use super::previous_node;
use super::scaled_masked_softmax::ScaledMaskedSoftmax;

pub fn register(registry: &mut Registry) {
    registry.register_dumper(ser_sdpa);
    registry.register_primitive(
        "tract_transformers_sdpa",
        &[
            TypeName::Scalar.tensor().named("q"),
            TypeName::Scalar.tensor().named("k"),
            TypeName::Scalar.tensor().named("v"),
            TypeName::Scalar.tensor().named("mask"),
            TypeName::Scalar.named("scale"),
            TypeName::String.named("datum_type"),
            TypeName::String.named("acc_datum_type"),
            TypeName::Logical.named("is_causal"),
        ],
        &[("output", TypeName::Scalar.tensor())],
        deser_spda,
    );
}

fn ser_sdpa(ast: &mut IntoAst, node: &TypedNode, op: &Sdpa) -> TractResult<Option<Arc<RValue>>> {
    // Inputs settings
    let q = ast.mapping[&node.inputs[0]].clone();
    let k = ast.mapping[&node.inputs[1]].clone();
    let v = ast.mapping[&node.inputs[2]].clone();
    let mut inputs = vec![q, k, v];
    if let Some(mask) = node.inputs.get(3).as_ref().map(|it| ast.mapping[it].clone()) {
        inputs.push(mask);
    }

    // Attributes settings
    let mut attrs = vec![
        ("is_causal", logical(op.is_causal)),
        ("datum_type", datum_type(op.datum_type)),
        ("acc_datum_type", datum_type(op.acc_datum_type)),
    ];
    if let Some(scale) = op.scale.as_ref() {
        attrs.push(("scale", numeric(scale.cast_to_scalar::<f32>()?)));
    }

    Ok(Some(invocation("tract_transformers_sdpa", &inputs, &attrs)))
}

fn deser_spda(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let q = invocation.named_arg_as(builder, "q")?;
    let k = invocation.named_arg_as(builder, "k")?;
    let v = invocation.named_arg_as(builder, "v")?;
    let mut inputs = vec![q, k, v];
    let q_rank = builder.model.outlet_fact(q)?.rank();
    if let Some(mut mask) = invocation.get_named_arg_as(builder, "mask")? {
        let mask_fact = builder.model.outlet_fact(mask)?;
        ensure!(mask_fact.rank() <= q_rank);
        for _ in mask_fact.rank()..q_rank {
            mask = builder.wire_as_outlets(AxisOp::Add(0), &[mask])?[0];
        }
        inputs.push(mask);
    };
    let scale: Option<f32> = invocation.get_named_arg_as(builder, "scale")?;
    let datum_type =
        DatumType::from_str(&invocation.named_arg_as::<String>(builder, "datum_type")?)?;
    let acc_datum_type =
        DatumType::from_str(&invocation.named_arg_as::<String>(builder, "acc_datum_type")?)?;
    let is_causal = invocation.named_arg_as(builder, "is_causal")?;
    builder.wire(Sdpa { scale: scale.map(tensor0), datum_type, acc_datum_type, is_causal }, &inputs)
}

#[derive(Debug, Clone)]
pub struct Sdpa {
    pub scale: Option<Tensor>,
    pub datum_type: DatumType,
    pub acc_datum_type: DatumType,
    pub is_causal: bool,
}

impl Sdpa {
    fn wire_softmax(
        &self,
        graph: &mut TypedModel,
        scores: OutletId,
        mask: Option<OutletId>,
        scale: f32,
    ) -> TractResult<OutletId> {
        let scores_fact = graph.outlet_fact(scores)?.clone();
        let rank = scores_fact.rank();
        ensure!(rank == 5);

        let scale = tensor0(scale).cast_to_dt(self.acc_datum_type)?.into_owned();
        if let Some(mask) = mask {
            ensure!(graph.outlet_fact(mask)?.rank() == 5);
            graph
                .wire_node(
                    "att_scaled_masked_softmax",
                    ScaledMaskedSoftmax { scale: scale.into() },
                    &[scores, mask],
                )
                .map(|o| o[0])
        } else {
            let scale_const = graph.add_const("scale", scale)?;
            let scaled_scores = wire_with_rank_broadcast(
                "scale_scores",
                graph,
                math::mul(),
                &[scores, scale_const],
            )?[0];
            graph
                .wire_node(
                    "att_softmax",
                    Softmax::new(tvec![rank - 1], None, SoftmaxKind::Softmax(SoftmaxExp::Libc)),
                    &[scaled_scores],
                )
                .map(|o| o[0])
        }
    }

    fn build_sdpa_graph(&self, input_facts: TVec<&TypedFact>) -> TractResult<TypedModel> {
        use change_axes::AxisOp::*;
        let mut graph = TypedModel::default();
        let mut q = graph.add_source("q", input_facts[0].clone())?;
        let mut k = graph.add_source("k", input_facts[1].clone())?;
        let mut v = graph.add_source("v", input_facts[2].clone())?;
        let mut mask =
            input_facts.get(3).map(|m| graph.add_source("mask", (*m).clone())).transpose()?;

        if input_facts[0].rank() == 3 {
            q = graph.wire_node("reshape_q_heads", Add(1), &[q])?[0];
            k = graph.wire_node("reshape_k_heads", Add(1), &[k])?[0];
            v = graph.wire_node("reshape_v_heads", Add(1), &[v])?[0];
            if let Some(m) = &mut mask {
                *m = graph.wire_node("reshape_m_heads", Add(1), &[*m])?[0];
            }
        }

        let [_, qh, att_rows, _qd] = &*graph.outlet_fact(q)?.shape.clone() else { unreachable!() };
        let [_b, kh, att_cols, kd] = &*graph.outlet_fact(k)?.shape.clone() else { unreachable!() };

        let num_qh = qh.to_usize()?;
        let num_kh = kh.to_usize()?;
        let num_kd = kd.to_usize()?;
        let num_att_rows = att_rows.to_usize()?;
        let num_att_cols = att_cols.to_usize()?;

        let g = num_qh / num_kh;

        q = graph.wire_node(
            "reshape_q_gha",
            Reshape(1, tvec!(qh.clone()), tvec!(kh.clone(), g.to_dim())),
            &[q],
        )?[0];
        k = graph.wire_node("reshape_k_gha", change_axes::AxisOp::Add(2), &[k])?[0];
        v = graph.wire_node("reshape_v_gha", change_axes::AxisOp::Add(2), &[v])?[0];
        if let Some(m) = &mut mask {
            if graph.outlet_fact(*m)?.shape[1].is_one() {
                *m = graph.wire_node("reshape_m_heads_groups", Add(2), &[*m])?[0];
            } else {
                *m = graph.wire_node(
                    "reshape_m_head_groups",
                    Reshape(1, tvec!(qh.clone()), tvec!(kh.clone(), g.to_dim())),
                    &[*m],
                )?[0];
            }
        }

        let scale = self
            .scale
            .as_ref()
            .map(|t| *t.to_scalar::<f32>().unwrap())
            .unwrap_or_else(|| (num_kd as f32).sqrt().recip());

        if self.is_causal {
            let m_array =
                Array5::from_shape_fn([1, 1, 1, num_att_rows, num_att_cols], |(_, _, _, r, c)| {
                    if c > (num_att_cols - num_att_rows) + r { f32::NEG_INFINITY } else { 0.0f32 }
                });
            mask = Some(graph.add_const(
                "causal_mask",
                m_array.into_tensor().cast_to_dt(self.acc_datum_type)?.into_owned(),
            )?);
        };

        let scores_einsum = EinSum::new("bhgmk,bhgnk->bhgmn".parse().unwrap(), self.acc_datum_type);
        let scores = graph.wire_node("scores", scores_einsum, &[q, k])?[0];
        if let Some(m) = &mut mask {
            if graph.outlet_fact(*m)?.datum_type != self.acc_datum_type {
                *m = graph.wire_node("cast_mask", Cast::new(self.acc_datum_type), &[*m])?[0];
            }
        }

        let attention_weights =
            self.wire_softmax(&mut graph, scores, mask, scale).context("In wire_softmax")?;
        let mut output = graph.wire_node(
            "att_out",
            EinSum::new("bhgmn,bhgnv->bhgmv".parse().unwrap(), self.acc_datum_type),
            &[attention_weights, v],
        )?[0];
        output = graph.wire_node(
            "reshape_out_gha",
            Reshape(1, tvec!(kh.clone(), g.to_dim()), tvec!(qh.clone())),
            &[output],
        )?[0];
        if input_facts[0].rank() == 3 {
            output = graph.wire_node("reshape_out_heads", Rm(1), &[output])?[0];
        }
        if graph.outlet_fact(output)?.datum_type != input_facts[0].datum_type {
            output =
                graph.wire_node("cast_output", Cast::new(input_facts[0].datum_type), &[output])?[0];
        }
        graph.set_output_outlets(&[output])?;
        Ok(graph)
    }

    pub fn patch_sdpa(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let input_facts = model.node_input_facts(node.id)?;
        let subgraph = self.build_sdpa_graph(input_facts)?;

        let mut patch = TypedModelPatch::new(format!("Explode SDPA node {}", node.name));
        patch.model = subgraph.into_decluttered()?;

        let body_inputs = patch.model.input_outlets()?;
        for (i, body_input_outlet) in body_inputs.iter().enumerate() {
            patch.taps.insert(*body_input_outlet, node.inputs[i]);
        }

        let body_outputs = patch.model.output_outlets()?;
        patch.shunt_outside(model, node.id.into(), body_outputs[0])?;
        //println!("{}",&patch.model);
        Ok(Some(patch))
    }
}

impl Op for Sdpa {
    fn name(&self) -> StaticName {
        "SDPA".into()
    }
    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("scale: {:?}", self.scale)])
    }
    op_as_typed_op!();
}

impl EvalOp for Sdpa {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input_facts: TVec<TypedFact> =
            inputs.iter().map(|tv| TypedFact::from(tv.clone().into_arc_tensor())).collect();
        let input_fact_refs: TVec<&TypedFact> = input_facts.iter().collect();
        let body =
            self.build_sdpa_graph(input_fact_refs).context("Wiring adhoc fallback graph ")?;
        let plan = TypedSimplePlan::new(body)?;
        plan.run(inputs)
    }
}

impl TypedOp for Sdpa {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        if self.is_causal {
            ensure!(inputs.len() == 3, "Mask cannot be provided if is_causal=true")
        };
        let rank = inputs[0].rank();
        ensure!(rank == 3 || rank == 4, "Input tensors must be 3D or 4D");
        ensure!(
            inputs[..3].iter().map(|it| it.rank()).all(|r| r == rank),
            "Q, K and V should have the same rank {}",
            rank
        );
        let mask = inputs.get(3);
        ensure!(mask.is_none_or(|m| m.rank() == rank));

        let q_shape = &inputs[0].shape.dims();
        let k_shape = &inputs[1].shape.dims();
        let v_shape = &inputs[2].shape.dims();

        ensure!(
            q_shape[0] == k_shape[0]
                && q_shape[0] == v_shape[0]
                && mask.as_ref().is_none_or(|m| m.shape[0].is_one() || m.shape[0] == q_shape[0])
        );

        if rank == 4 {
            let q_heads = q_shape[1].to_i64()?;
            let k_heads = k_shape[1].to_i64()?;
            let v_heads = v_shape[1].to_i64()?;
            ensure!(k_heads == v_heads, "K and V must have the same number of heads.");
            ensure!(
                q_heads % k_heads == 0,
                "Q heads ({}) must be a multiple of K/V heads ({})",
                q_heads,
                k_heads
            );
            ensure!(
                mask.as_ref().is_none_or(|m| m.shape[1].is_one() || m.shape[1] == q_heads.into())
            );
        }

        let output_shape = match rank {
            3 => {
                if let (&[b, seq_len, _], &[_, _, out_dim]) = (q_shape, v_shape) {
                    tvec!(b.clone(), seq_len.clone(), out_dim.clone())
                } else {
                    unreachable!()
                }
            }
            4 => {
                if let (&[b, n_heads, seq_len, _], &[_, _, _, out_dim]) = (q_shape, v_shape) {
                    tvec!(b.clone(), n_heads.clone(), seq_len.clone(), out_dim.clone())
                } else {
                    unreachable!()
                }
            }
            _ => unreachable!(),
        };

        let out_fact = inputs[0].datum_type().unwrap().fact(output_shape);
        Ok(tvec!(out_fact))
    }

    fn codegen(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        if self.acc_datum_type.is::<f32>() {
            let scale = self.scale.as_ref().map(|t| t.cast_to_scalar()).transpose()?;
            let op = FlashSdpaOp { causal: self.is_causal, scale };
            TypedModelPatch::replace_single_op(model, node, &node.inputs, op).map(Some)
        } else {
            todo!();
            // self.patch_sdpa(model, node).context("Wiring fallback SDPA")?
        }
    }
    as_op!();
}

// KV cache broadcast is: input -> concat -> AddAxis -> Broadcast -> Reshape
pub fn match_broadcast_kv_cache_pattern(
    model: &TypedModel,
    start_outlet: OutletId,
) -> TractResult<Option<OutletId>> {
    // Find Reshape node
    let reshape_node = model.node(start_outlet.node);
    rule_if!(
        reshape_node.op_is::<change_axes::AxisOp>()
            && matches!(
                reshape_node.op_as::<change_axes::AxisOp>().unwrap(),
                change_axes::AxisOp::Reshape(1, _, _)
            )
    );

    // Find broadcast node
    rule_if_some!(broadcast_node = previous_node(model, reshape_node));
    rule_if!(broadcast_node.op_is::<MultiBroadcastTo>());

    // Find add axis node
    rule_if_some!(unsqueeze_node = previous_node(model, broadcast_node));
    rule_if!(
        unsqueeze_node.op_is::<change_axes::AxisOp>()
            && matches!(
                unsqueeze_node.op_as::<change_axes::AxisOp>().unwrap(),
                change_axes::AxisOp::Add(2)
            )
    );

    fn is_concat(model: &TypedModel, n: &Node<TypedFact, Box<dyn TypedOp>>) -> bool {
        n.op_is::<TypedConcat>()
            && n.inputs.len() == 2
            && n.outputs.len() == 1
            && model.outputs.contains(&n.id.into())
    }

    fn is_dynkv(n: &Node<TypedFact, Box<dyn TypedOp>>) -> bool {
        n.op_is::<DynKeyValueCache>() && n.inputs.len() == 1 && n.outputs.len() == 1
    }

    // Find concat or dyn kvcache node
    rule_if_some!(node = previous_node(model, unsqueeze_node));
    rule_if!(is_concat(model, node) || is_dynkv(node));

    let kv_outlet = unsqueeze_node.inputs[0];
    if is_dynkv(node) {
        return Ok(Some(kv_outlet));
    }

    // node is concat, we need to check one input is a source
    let input0_node = model.node(node.inputs[0].node);
    let input1_node = model.node(node.inputs[1].node);
    if input0_node.op_is::<TypedSource>() || input1_node.op_is::<TypedSource>() {
        return Ok(Some(kv_outlet));
    }

    Ok(None)
}

pub fn fuse_kv_cache_broadcast_rule(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    node_name: &str,
    op: &Sdpa,
) -> TractResult<Option<TypedModelPatch>> {
    rule_if_some!(new_k_outlet = match_broadcast_kv_cache_pattern(model, node.inputs[1])?);
    rule_if_some!(new_v_outlet = match_broadcast_kv_cache_pattern(model, node.inputs[2])?);

    let mut patch = TypedModelPatch::default();
    let mut new_sdpa_inputs = node.inputs.clone();
    new_sdpa_inputs[1] = new_k_outlet;
    new_sdpa_inputs[2] = new_v_outlet;

    let tapped_inputs = patch.taps(model, &new_sdpa_inputs)?;

    let new_sdpa_node = patch.wire_node(
        format!("{}.sdpa_fused_kv_broadcast", node_name),
        op.clone(),
        &tapped_inputs,
    )?;

    patch.shunt_outside(model, node.id.into(), new_sdpa_node[0])?;

    Ok(Some(patch))
}
