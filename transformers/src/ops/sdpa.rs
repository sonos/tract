use std::str::FromStr;

use tract_core::ops::array::{MultiBroadcastTo, TypedConcat};
use tract_core::ops::cast::Cast;
use tract_core::ops::einsum::EinSum;
use tract_core::ops::source::TypedSource;
use tract_core::ops::{change_axes, math};
use tract_ndarray::Array2;
use tract_nnef::internal::*;
use tract_nnef::ser::datum_type;
use tract_nnef::tract_core::ops::nn::{Softmax, SoftmaxExp, SoftmaxKind};

use crate::ops::dyn_kv_cache::DynKeyValueCache;
use crate::ops::flash_sdpa::FlashSdpaOp;
use crate::rule_ensure;

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
    if let Some(mut mask) = invocation.get_named_arg_as(builder, "mask")? {
        for _ in 0..(builder.model.outlet_fact(mask)?.rank() - 2) {
            mask = builder.wire_as_outlets(AxisOp::Rm(0), &[mask])?[0];
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
    fn wire_const_causal_mask(
        &self,
        graph: &mut TypedModel,
        past_seq_len: usize,
        seq_len: usize,
    ) -> TractResult<Option<OutletId>> {
        let m_array = Array2::from_shape_fn([seq_len, past_seq_len + seq_len], |(r, c)| {
            if c > past_seq_len + r {
                f32::NEG_INFINITY
            } else {
                0.0f32
            }
        });
        let causal_mask = graph.add_const(
            "causal_mask",
            m_array.into_tensor().cast_to_dt(self.acc_datum_type)?.into_owned(),
        )?;
        Ok(Some(causal_mask))
    }

    fn normalize_mask(
        &self,
        graph: &mut TypedModel,
        mut mask: OutletId,
        scores_fact: &TypedFact,
    ) -> TractResult<OutletId> {
        let mask_fact = graph.outlet_fact(mask)?.clone();
        if mask_fact.datum_type != self.acc_datum_type {
            mask = graph.wire_node("cast_mask", Cast::new(self.acc_datum_type), &[mask])?[0];
        }

        let mut mask_shape = graph.outlet_fact(mask)?.shape.to_tvec();
        while mask_shape.len() < scores_fact.rank() {
            mask = graph.wire_node(
                format!("add_mask_axis_{}", mask_shape.len()),
                change_axes::AxisOp::Add(0),
                &[mask],
            )?[0];
            mask_shape = graph.outlet_fact(mask)?.shape.to_tvec();
        }
        Ok(mask)
    }

    fn wire_softmax(
        &self,
        graph: &mut TypedModel,
        scores: OutletId,
        mask: Option<OutletId>,
        scale: f32,
    ) -> TractResult<OutletId> {
        let scores_fact = graph.outlet_fact(scores)?.clone();
        let rank = scores_fact.rank();
        let scale = tensor0(scale).cast_to_dt(self.acc_datum_type)?.into_owned();

        let att_weights = if let Some(m) = mask {
            let scores_shape = scores_fact.shape.to_tvec();
            let (outer, [qs, ks]) = scores_shape.split_at(rank - 2) else { unreachable!() };
            let flat_dim = outer.iter().product::<TDim>();
            let scores_shape_3d = tvec![flat_dim.clone(), qs.clone(), ks.clone()];
            let reshaped_scores = graph.wire_node(
                "reshape_scores_for_softmax",
                change_axes::AxisOp::Reshape(0, scores_shape.clone(), scores_shape_3d.clone()),
                &[scores],
            )?[0];

            let mask_shape = graph.outlet_fact(m)?.shape.to_tvec();
            let (m_outer, [qs, ks]) = mask_shape.split_at(rank - 2) else { unreachable!() };
            let m_flat_dim = m_outer.iter().product::<TDim>();
            let mask_shape_3d = tvec![m_flat_dim, qs.clone(), ks.clone()];
            let reshaped_mask = graph.wire_node(
                "reshape_mask_for_softmax",
                change_axes::AxisOp::Reshape(0, mask_shape.clone(), mask_shape_3d),
                &[m],
            )?[0];

            let weights_3d = graph.wire_node(
                "att_scaled_masked_softmax",
                ScaledMaskedSoftmax { scale: scale.into() },
                &[reshaped_scores, reshaped_mask],
            )?[0];

            graph.wire_node(
                "reshape_post_scaled_masked_softmax",
                change_axes::AxisOp::Reshape(0, scores_shape_3d, scores_shape),
                &[weights_3d],
            )?[0]
        } else {
            let scale_const = graph.add_const("scale", scale)?;
            let scaled_scores = wire_with_rank_broadcast(
                "scale_scores",
                graph,
                math::mul(),
                &[scores, scale_const],
            )?[0];
            graph.wire_node(
                "att_softmax",
                Softmax::new(tvec![rank - 1], None, SoftmaxKind::Softmax(SoftmaxExp::Libc)),
                &[scaled_scores],
            )?[0]
        };

        Ok(att_weights)
    }

    fn build_sdpa_graph(&self, mut input_facts: TVec<&TypedFact>) -> TractResult<TypedModel> {
        let mut graph = TypedModel::default();
        let mut q_fact = input_facts.remove(0).clone();
        let mut k_fact = input_facts.remove(0).clone();
        let v_fact = input_facts.remove(0).clone();
        let m_fact = input_facts.pop().cloned();
        let mut q = graph.add_source("q", q_fact.clone())?;
        let mut k = graph.add_source("k", k_fact.clone())?;
        let mut v = graph.add_source("v", v_fact.clone())?;
        let mut m = if let Some(m) = m_fact.as_ref() {
            Some(graph.add_source("mask", m.clone())?)
        } else {
            None
        };

        let mut rank = q_fact.rank();
        let mut final_reshape = None;
        if rank == 4 {
            let qshape = q_fact.shape.to_tvec();
            let kshape = k_fact.shape.to_tvec();
            let [_, qh, sq, qd] = qshape.as_slice() else { unreachable!() };
            let [b, kh, sk, kd] = kshape.as_slice() else { unreachable!() };

            let num_q_heads = qh.to_usize()?;
            let num_k_heads = kh.to_usize()?;
            if num_q_heads > num_k_heads {
                let g = num_q_heads / num_k_heads;

                let new_qshape = tvec![b.clone(), kh.clone(), g.into(), sq.clone(), qd.clone()];
                let new_kshape = tvec![b.clone(), kh.clone(), 1.into(), sk.clone(), kd.clone()];
                q = graph.wire_node(
                    "reshape_q",
                    change_axes::AxisOp::Reshape(0, qshape.clone(), new_qshape.clone()),
                    &[q],
                )?[0];
                k = graph.wire_node(
                    "reshape_k",
                    change_axes::AxisOp::Reshape(0, kshape.clone(), new_kshape.clone()),
                    &[k],
                )?[0];
                v = graph.wire_node(
                    "reshape_v",
                    change_axes::AxisOp::Reshape(0, kshape.clone(), new_kshape.clone()),
                    &[v],
                )?[0];
                let dt = q_fact.datum_type;
                q_fact = TypedFact::dt_shape(dt, new_qshape);
                k_fact = TypedFact::dt_shape(dt, new_kshape.clone());

                final_reshape = Some((
                    tvec![b.clone(), kh.clone(), g.into(), sq.clone(), kd.clone()],
                    tvec![b.clone(), qh.clone(), sq.clone(), kd.clone()],
                ));
                rank += 1;
            }
        }

        let custom_scale = self.scale.as_ref().map(|t| *t.to_scalar::<f32>().unwrap());
        let d_k = k_fact.shape[rank - 1].to_i64()? as f32;
        let default_scale = 1.0 / d_k.sqrt();
        let scale = custom_scale.unwrap_or(default_scale);

        if self.is_causal {
            let q_seq_len = q_fact.shape[rank - 2].to_usize()?;
            let k_seq_len = k_fact.shape[rank - 2].to_usize()?;
            m = self.wire_const_causal_mask(&mut graph, k_seq_len - q_seq_len, q_seq_len)?;
        };

        let axes = match rank {
            3 => "amk,ank->amn".parse().unwrap(),
            4 => "bhmk,bhnk->bhmn".parse().unwrap(),
            5 => "bhgmk,bhgnk->bhgmn".parse().unwrap(),
            _ => unreachable!(),
        };

        let scores_einsum = EinSum::new(axes, self.acc_datum_type);
        let scores = graph.wire_node("scores", scores_einsum, &[q, k])?[0];
        if m.is_some() {
            let scores_fact = graph.outlet_fact(scores)?.clone();
            m = Some(self.normalize_mask(&mut graph, m.unwrap(), &scores_fact)?);
        }
        let attention_weights = self.wire_softmax(&mut graph, scores, m, scale)?;
        let axes = match rank {
            3 => "amk,akn->amn".parse().unwrap(),
            4 => "bhmn,bhnv->bhmv".parse().unwrap(),
            5 => "bhgmn,bhgnv->bhgmv".parse().unwrap(),
            _ => unreachable!(),
        };
        let mut output = graph.wire_node(
            "att_out",
            EinSum::new(axes, self.acc_datum_type),
            &[attention_weights, v],
        )?[0];
        if let Some((from, to)) = final_reshape {
            output = graph.wire_node(
                "final_reshape",
                change_axes::AxisOp::Reshape(0, from, to),
                &[output],
            )?[0];
        }
        let current_output_fact = graph.outlet_fact(output)?;
        if current_output_fact.datum_type != q_fact.datum_type().unwrap() {
            output = graph.wire_node(
                "cast_output",
                Cast::new(q_fact.datum_type().unwrap()),
                &[output],
            )?[0];
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
        let body = self.build_sdpa_graph(input_fact_refs)?;
        let plan = TypedSimplePlan::new(&body)?;
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
        if let Some(mask) = inputs.get(3) {
            ensure!(mask.rank() == 2, "Mask must be of rank 2.");
        }

        let q_shape = &inputs[0].shape.dims();
        let k_shape = &inputs[1].shape.dims();
        let v_shape = &inputs[2].shape.dims();

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
        let input_facts = model.node_input_facts(node.id)?;
        // optimized sdpa assume mask first dimensions (batch and optional head) are 1
        if input_facts.len() == 3
            || input_facts[3].shape.iter().rev().skip(2).all(|d| d.is_one())
                && self.acc_datum_type.is::<f32>()
        {
            let scale = self.scale.as_ref().map(|t| t.cast_to_scalar()).transpose()?;
            let op = FlashSdpaOp { causal: self.is_causal, scale };
            TypedModelPatch::replace_single_op(model, node, &node.inputs, op).map(Some)
        } else {
            self.patch_sdpa(model, node)
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
    rule_ensure!(
        reshape_node.op_is::<change_axes::AxisOp>()
            && matches!(
                reshape_node.op_as::<change_axes::AxisOp>().unwrap(),
                change_axes::AxisOp::Reshape(1, _, _)
            )
    );

    // Find broadcast node
    let Some(broadcast_node) = previous_node(model, reshape_node) else { return Ok(None) };
    rule_ensure!(broadcast_node.op_is::<MultiBroadcastTo>());

    // Find add axis node
    let Some(unsqueeze_node) = previous_node(model, broadcast_node) else { return Ok(None) };
    rule_ensure!(
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
    let Some(node) = previous_node(model, unsqueeze_node) else { return Ok(None) };
    rule_ensure!(is_concat(model, node) || is_dynkv(node));

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
    let matched_src_k_outlet = match_broadcast_kv_cache_pattern(model, node.inputs[1])?;
    let matched_src_v_outlet = match_broadcast_kv_cache_pattern(model, node.inputs[2])?;

    let (new_k_outlet, new_v_outlet) = match (matched_src_k_outlet, matched_src_v_outlet) {
        (Some(k_outlet), Some(v_outlet)) => (k_outlet, v_outlet),
        _ => return Ok(None),
    };
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
