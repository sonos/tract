use std::str::FromStr;

use tract_core::ops::array::{MultiBroadcastTo, TypedConcat};
use tract_core::ops::{change_axes, math, OpStateFreeze};
use tract_core::ops::einsum::EinSum;
use tract_core::ops::source::TypedSource;
use tract_ndarray::Array2;
use tract_nnef::internal::*;
use tract_nnef::ser::datum_type;
use tract_nnef::tract_core::ops::nn::{Softmax, SoftmaxExp, SoftmaxKind};

use crate::ops::dyn_kv_cache::DynKeyValueCache;
use crate::rule_ensure;

use super::previous_node;

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

fn deser_spda(model: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let q = invocation.named_arg_as(model, "q")?;
    let k = invocation.named_arg_as(model, "k")?;
    let v = invocation.named_arg_as(model, "v")?;
    let mut inputs = vec![q, k, v];
    if let Some(mask) = invocation.get_named_arg_as(model, "mask")? {
        inputs.push(mask);
    };
    let scale: Option<f32> = invocation.get_named_arg_as(model, "scale")?;
    let datum_type =
        DatumType::from_str(&invocation.named_arg_as::<String>(model, "datum_type")?)?;
    let acc_datum_type =
        DatumType::from_str(&invocation.named_arg_as::<String>(model, "acc_datum_type")?)?;
    let is_causal = invocation.named_arg_as(model, "is_causal")?;
    model.wire(
        Sdpa { scale: scale.map(tensor0), datum_type, acc_datum_type, is_causal, subgraph: None, optimized: false },
        &inputs,
    )
}

#[derive(Debug, Clone)]
pub struct Sdpa {
    pub scale: Option<Tensor>,
    pub datum_type: DatumType,
    pub acc_datum_type: DatumType,
    pub is_causal: bool,
    pub subgraph: Option<TypedModel>,
    pub optimized: bool,
}

impl Sdpa {
    fn build_sdpa_graph(&self, mut input_facts: TVec<&TypedFact>) -> TractResult<TypedModel> {
        let mut new_model = TypedModel::default();
        let mut q_fact = input_facts.remove(0).clone();
        let mut k_fact = input_facts.remove(0).clone();
        let v_fact = input_facts.remove(0).clone();
        let m_fact = input_facts.pop().cloned();
        
        let mut q = new_model.add_source("q", q_fact.clone())?;
        let mut k = new_model.add_source("k", k_fact.clone())?;
        let mut v = new_model.add_source("v", v_fact.clone())?;
        let mut m = if let Some(m) = m_fact.as_ref() {
            Some(new_model.add_source("mask", m.clone())?)
        } else {
            None
        };

        let mut rank = q_fact.rank();
        let dt = q_fact.datum_type;
        let mut final_reshape = None;
        if rank == 4 {
            let q_shape = q_fact.shape.to_tvec();
            let k_shape = k_fact.shape.to_tvec();
            let [_, q_heads, _, q_dims] = q_shape.as_slice() else { unreachable!() };
            let [b, k_heads, seq_len, k_dims] = k_shape.as_slice() else { unreachable!() };

            let num_q_heads = q_heads.to_usize()?;
            let num_k_heads = k_heads.to_usize()?;
            if num_q_heads > num_k_heads {
                let g = num_q_heads / num_k_heads;
                
                let new_q_shape = tvec![b.clone(), k_heads.clone(), g.into(), seq_len.clone(), q_dims.clone()];
                let new_kv_shape = tvec![b.clone(), k_heads.clone(), 1.into(), seq_len.clone(), k_dims.clone()];
                
                q = new_model.wire_node("reshape_q", change_axes::AxisOp::Reshape(0, q_shape.clone(), new_q_shape.clone()), &[q])?[0];
                k = new_model.wire_node("reshape_k", change_axes::AxisOp::Reshape(0, k_shape.clone(), new_kv_shape.clone()), &[k])?[0];
                v = new_model.wire_node("reshape_v", change_axes::AxisOp::Reshape(0, k_shape.clone(), new_kv_shape.clone()), &[v])?[0];
                q_fact = TypedFact::dt_shape(dt, new_q_shape);
                k_fact = TypedFact::dt_shape(dt, new_kv_shape.clone());
                if let Some(m) = m.as_mut() {
                    let m_shape = m_fact.unwrap().shape.to_tvec();
                    let new_m_shape = tvec![b.clone(), k_heads.clone(), g.into(), seq_len.clone(), seq_len.clone()];
                    *m = new_model.wire_node("reshape_m", change_axes::AxisOp::Reshape(0, m_shape, new_m_shape.clone()), &[*m])?[0];
                }
                final_reshape = Some((tvec![b.clone(), k_heads.clone(), g.into(), seq_len.clone(), k_dims.clone()], tvec![b.clone(), q_heads.clone(), seq_len.clone(), k_dims.clone()]));
                rank +=1;
            }
        }

        let d_k = k_fact.shape[rank - 1].to_i64()? as f32;
        let scale_value = self.scale.as_ref().map(|t| *t.to_scalar::<f32>().unwrap()).unwrap_or(1.0 / d_k.sqrt());
        let scale_const = new_model.add_const("scale", tensor0(scale_value).cast_to_dt(dt)?.into_owned())?;
        if self.is_causal {
            let q_seq_len = q_fact.shape[rank - 2].to_usize()?;
            let k_seq_len = k_fact.shape[rank - 2].to_usize()?;
            let m_array = Array2::from_shape_fn([q_seq_len, k_seq_len], |(r, c)| {
                if c > r {
                    f32::NEG_INFINITY
                } else {
                    0.0f32
                }
            });
            let causal_mask = new_model.add_const("causal_mask", m_array.into_tensor().cast_to_dt(dt)?.into_owned())?;
            m = Some(causal_mask);
        };

        let axes = match rank {
            3 => "amk,ank->amn".parse().unwrap(),
            4 => "bhmk,bhnk->bhmn".parse().unwrap(),
            5 => "bhgmk,bhgnk->bhgmn".parse().unwrap(),
            _ => unreachable!(),
        };

        let q_dot_kt = new_model.wire_node("scores",EinSum::new(axes, self.acc_datum_type), &[q, k])?[0];
        let scaled_scores = wire_with_rank_broadcast("scale_scores",&mut new_model, math::mul(), &[q_dot_kt, scale_const])?[0];
        let scaled_masked_scores = if let Some(m) = m {
            wire_with_rank_broadcast("apply_mask", &mut new_model, math::add(), &[scaled_scores, m])?[0]
        } else {
            scaled_scores
        };

        let attention_weights = new_model.wire_node(
            "att_softmax",
            Softmax::new(tvec![rank-1], None, SoftmaxKind::Softmax(SoftmaxExp::Libc)),
            &[scaled_masked_scores],
        )?[0];
        let axes = match rank {
            3 => "amk,akn->amn".parse().unwrap(),
            4 => "bhmn,bhnv->bhmv".parse().unwrap(),
            5 => "bhgmn,bhgnv->bhgmv".parse().unwrap(),
            _ => unreachable!(),
        };
        let mut output = new_model.wire_node("att_out", EinSum::new(axes, self.acc_datum_type), &[attention_weights, v])?[0];
        if let Some((from, to)) = final_reshape {
            output = new_model.wire_node("final_reshape", change_axes::AxisOp::Reshape(0, from, to), &[output])?[0];
        }
        new_model.set_output_outlets(&[output])?;
        Ok(new_model)
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
        self.subgraph.is_none()
    }

    fn state(
        &self,
        _session: &mut SessionState,
        _node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        if let Some(model) = &self.subgraph {
            let plan = TypedSimplePlan::new(model.clone())?;
            Ok(Some(Box::new(SdpaState { plan })))
        } else {
            Ok(None)
        }
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
            inputs.iter().map(|it| it.rank()).all(|r| r == rank),
            "All inputs should have the same rank {}",
            rank
        );

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

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        if self.subgraph.is_some() {
            return Ok(None);
        }

        let input_facts = model.node_input_facts(node.id)?;
        let subgraph = self.build_sdpa_graph(input_facts)?;
        let decluttered_subgraph = subgraph.into_decluttered()?;
        let new_op = Sdpa { subgraph: Some(decluttered_subgraph), optimized: false, ..self.clone() };
        TypedModelPatch::replace_single_op(&model, node, &node.inputs, new_op).map(Some)
    }

    fn codegen(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        if self.optimized {
            return Ok(None)
        }
        if let Some(graph) = &self.subgraph {
            let optimized_body = graph.clone().into_optimized()?; 
            let new_op = Sdpa { subgraph: Some(optimized_body), optimized: true, ..self.clone() };
            return TypedModelPatch::replace_single_op(model, node, &node.inputs, new_op).map(Some);
        }
        Ok(None)
    }

    as_op!();
}

#[derive(Debug, Clone)]
struct SdpaState {
    plan: TypedSimplePlan<TypedModel>,
}

impl OpState for SdpaState {
    fn eval(
        &mut self,
        _session: &mut SessionState,
        _op: &dyn Op,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        self.plan.run(inputs)
    }
}

#[derive(Debug, Clone)]
struct FrozenSdpaState {
    plan: TypedSimplePlan<TypedModel>,
}

impl OpStateFreeze for SdpaState {
    fn freeze(&self) -> Box<dyn FrozenOpState> {
        unimplemented!()
    }
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
