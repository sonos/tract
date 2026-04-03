use std::hash::{Hash, Hasher};
use std::sync::Arc;

use tract_ndarray::{s, Array2, ArrayView2, Axis};
use tract_nnef::internal::*;
use tract_nnef::tract_core::ops::einsum::EinSum;
use tract_nnef::tract_core::ops::konst::Const;
use tract_nnef::tract_core::ops::math::mul;
use tract_nnef::tract_core::ops::{FrozenOpState, OpStateFreeze};

use super::gelu_approximate::GeluApproximate;
use super::silu::Silu;

pub fn register(registry: &mut Registry) {
    registry.register_primitive(
        "tract_moe_ffn",
        &[
            TypeName::Scalar.tensor().named("x"),
            TypeName::Scalar.tensor().named("wg"),
            TypeName::Scalar.tensor().named("w1"),
            TypeName::Scalar.tensor().named("w2"),
            TypeName::Scalar.tensor().named("w3"),
            TypeName::Integer.named("k"),
            TypeName::String.named("activation"),
            TypeName::Logical.named("normalize_gates"),
        ],
        &[
            ("output", TypeName::Scalar.tensor()),
            ("router_logits", TypeName::Scalar.tensor()),
        ],
        deser_moe_ffn,
    );
}

fn deser_moe_ffn(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<Value> {
    let x = invocation.named_arg_as(builder, "x")?;
    let wg = invocation.named_arg_as(builder, "wg")?;
    let w1 = invocation.named_arg_as(builder, "w1")?;
    let w2 = invocation.named_arg_as(builder, "w2")?;
    let w3: Option<OutletId> = invocation.get_named_arg_as(builder, "w3")?;
    let k: i64 = invocation.named_arg_as(builder, "k")?;
    let activation: String = invocation.named_arg_as(builder, "activation")?;
    let normalize_gates: bool = invocation.named_arg_as(builder, "normalize_gates")?;

    let mut inputs = vec![x, wg, w1, w2];
    let has_w3 = w3.is_some();
    if let Some(w3) = w3 {
        inputs.push(w3);
    }

    builder.wire(
        MoeFfn {
            k: k as usize,
            activation,
            normalize_gates,
            has_w3,
        },
        &inputs,
    )
}

#[derive(Clone, Debug, Hash)]
pub struct MoeFfn {
    pub k: usize,
    pub activation: String,
    pub normalize_gates: bool,
    pub has_w3: bool,
}

impl Op for MoeFfn {
    fn name(&self) -> StaticName {
        "MoeFfn".to_string().into()
    }
    op_as_typed_op!();
}

impl EvalOp for MoeFfn {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        // inputs: x [T,D], wg [E,D] or [1,E,D], w1 [E,D,H], w2 [E,H,D], [w3 [E,D,H]]
        let x = inputs[0].to_array_view::<f32>()?;
        let wg_raw = inputs[1].to_array_view::<f32>()?;
        let w1 = inputs[2].to_array_view::<f32>()?;
        let w2 = inputs[3].to_array_view::<f32>()?;
        let w3 = if self.has_w3 {
            Some(inputs[4].to_array_view::<f32>()?)
        } else {
            None
        };

        // Normalize wg to 2D [E, D] (may be [1, E, D] from unsqueeze)
        let wg: ArrayView2<f32> = if wg_raw.ndim() == 3 {
            wg_raw.index_axis(Axis(0), 0).into_dimensionality()?
        } else {
            wg_raw.into_dimensionality()?
        };

        // Normalize x to 2D [T, D] (may be [B, S, D] with B=1)
        let x_ndim = x.ndim();
        let x_orig_shape: Vec<usize> = x.shape().to_vec();
        let x: ArrayView2<f32> = if x_ndim == 3 {
            x.into_shape_with_order((x_orig_shape[0] * x_orig_shape[1], x_orig_shape[2]))?.into_dimensionality()?
        } else {
            x.into_dimensionality()?
        };

        let t_tokens = x.shape()[0];
        let d_model = x.shape()[1];
        let num_experts = wg.shape()[0];
        let _d_hidden = w1.shape()[2];

        // ---- Step 1: Router ----
        // logits = x @ wg.T  [T, D] @ [D, E] -> [T, E]
        let router_logits: Array2<f32> = x.dot(&wg.t());

        // ---- Step 2: Top-k selection + gate weights per token ----
        // assignments[token] = Vec<(expert_id, gate_weight)>
        let mut assignments: Vec<Vec<(usize, f32)>> = Vec::with_capacity(t_tokens);
        for t in 0..t_tokens {
            let row = router_logits.row(t);
            let mut scores: Vec<(usize, f32)> =
                row.iter().enumerate().map(|(e, &s)| (e, s)).collect();
            scores.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
            scores.truncate(self.k);

            let gate_weights: Vec<f32> = if self.normalize_gates {
                let max_s = scores.iter().map(|(_, s)| *s).fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = scores.iter().map(|(_, s)| (s - max_s).exp()).collect();
                let sum: f32 = exps.iter().sum();
                exps.iter().map(|e| e / sum).collect()
            } else {
                scores.iter().map(|(_, s)| *s).collect()
            };

            assignments.push(
                scores
                    .iter()
                    .zip(gate_weights)
                    .map(|((eid, _), gw)| (*eid, gw))
                    .collect(),
            );
        }

        // ---- Step 3: Group tokens per expert ----
        // expert_tokens[eid] = Vec<(token_idx, gate_weight)>
        let mut expert_tokens: Vec<Vec<(usize, f32)>> = vec![Vec::new(); num_experts];
        for (t, token_experts) in assignments.iter().enumerate() {
            for &(eid, gw) in token_experts {
                expert_tokens[eid].push((t, gw));
            }
        }

        // ---- Step 4: Batched expert computation (conditional!) ----
        let mut output = Array2::<f32>::zeros((t_tokens, d_model));

        for eid in 0..num_experts {
            let tokens = &expert_tokens[eid];
            if tokens.is_empty() {
                continue; // Skip unused experts entirely
            }
            let n = tokens.len();

            // Gather: build x_batch [n, D] from selected tokens
            let mut x_batch = Array2::<f32>::zeros((n, d_model));
            for (i, &(t, _)) in tokens.iter().enumerate() {
                x_batch.row_mut(i).assign(&x.row(t));
            }

            // Expert weight slices for this expert
            let w1_e = w1.slice(s![eid, .., ..]); // [D, H]
            let w2_e = w2.slice(s![eid, .., ..]); // [H, D]

            // h = x_batch @ w1_e  -> [n, H]  (BLAS-backed GEMM)
            let mut h: Array2<f32> = x_batch.dot(&w1_e);

            if let Some(ref w3) = w3 {
                // SwiGLU: h = silu(h) * (x_batch @ w3_e)
                let w3_e = w3.slice(s![eid, .., ..]); // [D, H]
                let gate: Array2<f32> = x_batch.dot(&w3_e); // [n, H]

                h.iter_mut().zip(gate.iter()).for_each(|(h_val, &g_val)| {
                    let silu = *h_val / (1.0 + (-*h_val).exp());
                    *h_val = silu * g_val;
                });
            } else {
                // Simple silu activation
                h.iter_mut().for_each(|h_val| {
                    *h_val = *h_val / (1.0 + (-*h_val).exp());
                });
            }

            // y_expert = h @ w2_e  -> [n, D]  (BLAS-backed GEMM)
            let y_expert: Array2<f32> = h.dot(&w2_e);

            // ---- Step 5: Scatter-add weighted results back ----
            for (i, &(t, gw)) in tokens.iter().enumerate() {
                let y_row = y_expert.row(i);
                let mut out_row = output.row_mut(t);
                out_row.scaled_add(gw, &y_row);
            }
        }

        // Restore original rank if input was 3D
        let output_tensor = if x_ndim == 3 {
            output.into_shape_with_order((x_orig_shape[0], x_orig_shape[1], d_model))?.into_tensor()
        } else {
            output.into_tensor()
        };
        let router_tensor = if x_ndim == 3 {
            router_logits.into_shape_with_order((x_orig_shape[0], x_orig_shape[1], num_experts))?.into_tensor()
        } else {
            router_logits.into_tensor()
        };
        Ok(tvec![output_tensor.into_tvalue(), router_tensor.into_tvalue()])
    }
}

impl TypedOp for MoeFfn {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        // Output 0: same shape as input x
        let x_fact = inputs[0];
        let output_fact = x_fact.datum_type.fact(x_fact.shape.clone());

        // Output 1: router_logits — same leading dims as x, last dim = E
        let wg_fact = inputs[1];
        let e_dim = if wg_fact.rank() == 3 {
            wg_fact.shape[1].clone()
        } else {
            wg_fact.shape[0].clone()
        };
        let mut router_shape: TVec<TDim> = x_fact.shape.iter().cloned().collect();
        // Replace last dim (D) with E
        if let Some(last) = router_shape.last_mut() {
            *last = e_dim;
        }
        let router_fact = x_fact.datum_type.fact(router_shape);

        Ok(tvec!(output_fact, router_fact))
    }

    fn codegen(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        // Only optimize if all weights are constants
        let wg_const = model.node(node.inputs[1].node).op_as::<Const>();
        let w1_const = model.node(node.inputs[2].node).op_as::<Const>();
        let w2_const = model.node(node.inputs[3].node).op_as::<Const>();
        let w3_const = if self.has_w3 {
            let c = model.node(node.inputs[4].node).op_as::<Const>();
            if c.is_none() {
                return Ok(None);
            }
            c
        } else {
            None
        };

        if wg_const.is_none() || w1_const.is_none() || w2_const.is_none() {
            return Ok(None);
        }

        let wg_tensor = wg_const.unwrap().val().clone();
        let w1_tensor = w1_const.unwrap().val().clone();
        let w2_tensor = w2_const.unwrap().val().clone();
        let w3_tensor = w3_const.map(|c| c.val().clone());

        let dt = model.outlet_fact(node.inputs[0])?.datum_type;

        // Bail if the activation is not supported by the optimized path
        if activation_op(&self.activation, self.has_w3).is_none() {
            return Ok(None);
        }

        let num_experts = w1_tensor.shape()[0];
        let d_model = w1_tensor.shape()[1];
        let d_hidden = w1_tensor.shape()[2];

        // Build router plan: x [T, D] @ wg.T -> [T, E]
        let router_plan =
            build_router_plan(&wg_tensor, dt, &model.symbols).context("Building router plan")?;

        // Build per-expert plans
        let mut expert_plans = Vec::with_capacity(num_experts);
        for eid in 0..num_experts {
            let w1_e = w1_tensor.slice(0, eid, eid + 1)?.into_shape(&[d_model, d_hidden])?;
            let w2_e = w2_tensor.slice(0, eid, eid + 1)?.into_shape(&[d_hidden, d_model])?;
            let w3_e = if let Some(ref w3) = w3_tensor {
                Some(w3.slice(0, eid, eid + 1)?.into_shape(&[d_model, d_hidden])?)
            } else {
                None
            };

            let plan = build_expert_plan(
                &w1_e,
                &w2_e,
                w3_e.as_ref(),
                &self.activation,
                dt,
                &model.symbols,
            )
            .with_context(|| format!("Building expert plan for expert {eid}"))?;
            expert_plans.push(plan);
        }

        let opt_op = OptMoeFfn {
            k: self.k,
            normalize_gates: self.normalize_gates,
            num_experts,
            d_model,
            d_hidden,
            router_plan,
            expert_plans,
        };

        let mut patch = TypedModelPatch::default();
        let x_tap = patch.tap_model(model, node.inputs[0])?;
        let wires = patch.wire_node(&node.name, opt_op, &[x_tap])?;
        patch.shunt_outside(model, OutletId::new(node.id, 0), wires[0])?;
        patch.shunt_outside(model, OutletId::new(node.id, 1), wires[1])?;
        Ok(Some(patch))
    }

    as_op!();
}

// ---------------------------------------------------------------------------
// Activation helper
// ---------------------------------------------------------------------------

fn activation_op(name: &str, has_w3: bool) -> Option<Box<dyn TypedOp>> {
    match name {
        "silu" => Some(Box::new(Silu)),
        // SwiGLU: the inner activation is silu, w3 provides the gate branch
        "swiglu" if has_w3 => Some(Box::new(Silu)),
        "gelu" => Some(Box::new(GeluApproximate { fast_impl: false })),
        "relu" => Some(Box::new(tract_nnef::tract_core::ops::nn::leaky_relu(0.0))),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Sub-model builders
// ---------------------------------------------------------------------------

fn build_router_plan(
    wg: &Arc<Tensor>,
    dt: DatumType,
    symbols: &SymbolScope,
) -> TractResult<Arc<TypedSimplePlan>> {
    let mut model = TypedModel::default();
    model.symbols = symbols.clone();
    let n_sym = symbols.sym("moe_t");

    // wg is [E, D] or [1, E, D] — normalize to [E, D]
    let wg_2d = if wg.rank() == 3 {
        wg.slice(0, 0, 1)?.into_shape(&[wg.shape()[1], wg.shape()[2]])?
    } else {
        (**wg).clone()
    };

    let d_model = wg_2d.shape()[1];
    let _num_experts = wg_2d.shape()[0];

    // x: [T, D]
    let x = model.add_source("x", dt.fact(&[n_sym.to_dim(), d_model.to_dim()]))?;
    // wg: [E, D] as constant
    let wg_const = model.add_const("wg", wg_2d)?;

    // router_logits = x @ wg.T  -> [T, E]
    // EinSum: "ij,kj->ik" means i=T, j=D (contracted), k=E
    let axes: AxesMapping = "ij,kj->ik".parse()?;
    let logits = model.wire_node("router_logits", EinSum::new(axes, dt), &[x, wg_const])?[0];

    model.set_output_outlets(&[logits])?;
    SimplePlan::new(model.into_optimized()?)
}

fn build_expert_plan(
    w1: &Tensor,
    w2: &Tensor,
    w3: Option<&Tensor>,
    activation: &str,
    dt: DatumType,
    symbols: &SymbolScope,
) -> TractResult<Arc<TypedSimplePlan>> {
    let mut model = TypedModel::default();
    model.symbols = symbols.clone();
    let n_sym = symbols.sym("moe_n");

    let d_model = w1.shape()[0]; // w1: [D, H]
    let _d_hidden = w1.shape()[1];

    // Input: x_batch [n, D]
    let x = model.add_source("x", dt.fact(&[n_sym.to_dim(), d_model.to_dim()]))?;

    // w1 matmul: x_batch [n,D] @ w1 [D,H] -> [n,H]
    let w1_const = model.add_const("w1", w1.clone())?;
    let axes_mm: AxesMapping = "ij,jk->ik".parse()?;
    let h = model.wire_node("w1_matmul", EinSum::new(axes_mm.clone(), dt), &[x, w1_const])?[0];

    // Activation (caller guarantees activation_op returns Some via codegen check)
    let act_op = activation_op(activation, w3.is_some())
        .ok_or_else(|| format_err!("Unsupported activation: {activation}"))?;
    let h = model.wire_node("activation", act_op, &[h])?[0];

    // Optional SwiGLU: gate = x @ w3, h = h * gate
    let h = if let Some(w3) = w3 {
        let w3_const = model.add_const("w3", w3.clone())?;
        let gate =
            model.wire_node("w3_matmul", EinSum::new(axes_mm.clone(), dt), &[x, w3_const])?[0];
        model.wire_node("swiglu_mul", mul(), &[h, gate])?[0]
    } else {
        h
    };

    // w2 matmul: h [n,H] @ w2 [H,D] -> [n,D]
    let w2_const = model.add_const("w2", w2.clone())?;
    let y = model.wire_node("w2_matmul", EinSum::new(axes_mm, dt), &[h, w2_const])?[0];

    model.set_output_outlets(&[y])?;
    SimplePlan::new(model.into_optimized()?)
}

// ---------------------------------------------------------------------------
// OptMoeFfn — optimized MoE FFN with pre-compiled expert sub-plans
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct OptMoeFfn {
    pub k: usize,
    pub normalize_gates: bool,
    pub num_experts: usize,
    pub d_model: usize,
    pub d_hidden: usize,
    pub router_plan: Arc<TypedSimplePlan>,
    pub expert_plans: Vec<Arc<TypedSimplePlan>>,
}

impl Hash for OptMoeFfn {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.k.hash(state);
        self.normalize_gates.hash(state);
        self.num_experts.hash(state);
        self.d_model.hash(state);
        self.d_hidden.hash(state);
    }
}

impl Op for OptMoeFfn {
    fn name(&self) -> StaticName {
        "OptMoeFfn".to_string().into()
    }
    op_as_typed_op!();
}

impl EvalOp for OptMoeFfn {
    fn is_stateless(&self) -> bool {
        false
    }

    fn state(
        &self,
        _session: &TurnState,
        _node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        let router_state = self.router_plan.spawn()?;
        let expert_states = self
            .expert_plans
            .iter()
            .map(|p| p.spawn())
            .collect::<TractResult<Vec<_>>>()?;
        Ok(Some(Box::new(OptMoeFfnState {
            op: self.clone(),
            router_state,
            expert_states,
        })))
    }
}

// ---------------------------------------------------------------------------
// OptMoeFfnState — pre-spawned plan states, reused across eval calls
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct OptMoeFfnState {
    op: OptMoeFfn,
    router_state: TypedSimpleState,
    expert_states: Vec<TypedSimpleState>,
}

#[derive(Clone, Debug)]
struct FrozenOptMoeFfnState {
    op: OptMoeFfn,
    router_state: TypedFrozenSimpleState,
    expert_states: Vec<TypedFrozenSimpleState>,
}

impl OpStateFreeze for OptMoeFfnState {
    fn freeze(&self) -> Box<dyn FrozenOpState> {
        Box::new(FrozenOptMoeFfnState {
            op: self.op.clone(),
            router_state: self.router_state.freeze(),
            expert_states: self.expert_states.iter().map(|s| s.freeze()).collect(),
        })
    }
}

impl FrozenOpState for FrozenOptMoeFfnState {
    fn unfreeze(&self) -> Box<dyn OpState> {
        Box::new(OptMoeFfnState {
            op: self.op.clone(),
            router_state: self.router_state.unfreeze(),
            expert_states: self.expert_states.iter().map(|s| s.unfreeze()).collect(),
        })
    }
}

impl OpState for OptMoeFfnState {
    fn eval(
        &mut self,
        _session: &mut TurnState,
        _op: &dyn Op,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let op = &self.op;
        let x_input = &inputs[0];
        let x_view = x_input.to_array_view::<f32>()?;
        let x_ndim = x_view.ndim();
        let x_orig_shape: Vec<usize> = x_view.shape().to_vec();

        // Normalize x to 2D [T, D]
        let x: ArrayView2<f32> = if x_ndim == 3 {
            x_view
                .into_shape_with_order((x_orig_shape[0] * x_orig_shape[1], x_orig_shape[2]))?
                .into_dimensionality()?
        } else {
            x_view.into_dimensionality()?
        };

        let t_tokens = x.shape()[0];
        let d_model = x.shape()[1];
        let dt = x_input.datum_type();

        // ---- Step 1: Router via pre-spawned state ----
        let x_2d_tensor = if x_ndim == 3 {
            let mut t = Tensor::zero_dt(dt, &[t_tokens, d_model])?;
            t.as_slice_mut::<f32>()?.copy_from_slice(
                x.as_slice().context("x not contiguous for router")?,
            );
            t
        } else {
            (*x_input).clone().into_tensor()
        };

        let router_result = self.router_state.run(tvec![x_2d_tensor.into_tvalue()])?;
        let router_logits_tv = &router_result[0];
        let router_logits = router_logits_tv.to_array_view::<f32>()?;
        let router_logits: ArrayView2<f32> = router_logits.into_dimensionality()?;

        // ---- Step 2: Top-k selection + gate weights ----
        let mut assignments: Vec<Vec<(usize, f32)>> = Vec::with_capacity(t_tokens);
        for t in 0..t_tokens {
            let row = router_logits.row(t);
            let mut scores: Vec<(usize, f32)> =
                row.iter().enumerate().map(|(e, &s)| (e, s)).collect();
            scores.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
            scores.truncate(op.k);

            let gate_weights: Vec<f32> = if op.normalize_gates {
                let max_s =
                    scores.iter().map(|(_, s)| *s).fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = scores.iter().map(|(_, s)| (s - max_s).exp()).collect();
                let sum: f32 = exps.iter().sum();
                exps.iter().map(|e| e / sum).collect()
            } else {
                scores.iter().map(|(_, s)| *s).collect()
            };

            assignments.push(
                scores
                    .iter()
                    .zip(gate_weights)
                    .map(|((eid, _), gw)| (*eid, gw))
                    .collect(),
            );
        }

        // ---- Step 3: Group tokens per expert ----
        let mut expert_tokens: Vec<Vec<(usize, f32)>> = vec![Vec::new(); op.num_experts];
        for (t, token_experts) in assignments.iter().enumerate() {
            for &(eid, gw) in token_experts {
                expert_tokens[eid].push((t, gw));
            }
        }

        // ---- Step 4: Per-expert computation via pre-spawned states ----
        let mut output = Array2::<f32>::zeros((t_tokens, d_model));

        for eid in 0..op.num_experts {
            let tokens = &expert_tokens[eid];
            if tokens.is_empty() {
                continue;
            }
            let n = tokens.len();

            // Gather: build x_batch [n, D]
            let mut x_batch = Tensor::zero_dt(dt, &[n, d_model])?;
            {
                let x_batch_slice = x_batch.as_slice_mut::<f32>()?;
                for (i, &(t, _)) in tokens.iter().enumerate() {
                    let src = x.row(t);
                    x_batch_slice[i * d_model..(i + 1) * d_model]
                        .copy_from_slice(src.as_slice().unwrap());
                }
            }

            // Run expert plan (reusing pre-spawned state)
            let y_expert = self.expert_states[eid].run(tvec![x_batch.into_tvalue()])?;

            // Scatter-add weighted results
            let y_view = y_expert[0].to_array_view::<f32>()?;
            let y_view: ArrayView2<f32> = y_view.into_dimensionality()?;
            for (i, &(t, gw)) in tokens.iter().enumerate() {
                let y_row = y_view.row(i);
                let mut out_row = output.row_mut(t);
                out_row.scaled_add(gw, &y_row);
            }
        }

        // ---- Restore shapes ----
        let output_tensor = if x_ndim == 3 {
            output
                .into_shape_with_order((x_orig_shape[0], x_orig_shape[1], d_model))?
                .into_tensor()
        } else {
            output.into_tensor()
        };
        let router_tensor = if x_ndim == 3 {
            let rl = router_logits_tv.clone().into_tensor();
            rl.into_shape(&[x_orig_shape[0], x_orig_shape[1], op.num_experts])?
        } else {
            router_logits_tv.clone().into_tensor()
        };

        Ok(tvec![output_tensor.into_tvalue(), router_tensor.into_tvalue()])
    }
}

impl TypedOp for OptMoeFfn {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let x_fact = inputs[0];
        let output_fact = x_fact.datum_type.fact(x_fact.shape.clone());

        let mut router_shape: TVec<TDim> = x_fact.shape.iter().cloned().collect();
        if let Some(last) = router_shape.last_mut() {
            *last = self.num_experts.to_dim();
        }
        let router_fact = x_fact.datum_type.fact(router_shape);

        Ok(tvec!(output_fact, router_fact))
    }

    as_op!();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_moe_model(
        t_tokens: usize,
        d_model: usize,
        d_hidden: usize,
        num_experts: usize,
        k: usize,
        has_w3: bool,
    ) -> TractResult<(TypedModel, Tensor)> {
        let mut model = TypedModel::default();

        let x = model.add_source("x", f32::datum_type().fact(&[t_tokens, d_model]))?;

        // Deterministic pseudo-random weights
        let mut rng_state: u64 = 42;
        let mut next_f32 = || -> f32 {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng_state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        };

        let make_tensor = |shape: &[usize], rng: &mut dyn FnMut() -> f32| -> Tensor {
            let n: usize = shape.iter().product();
            let data: Vec<f32> = (0..n).map(|_| rng()).collect();
            tract_ndarray::ArrayD::from_shape_vec(shape, data).unwrap().into_tensor()
        };

        let wg_data = make_tensor(&[num_experts, d_model], &mut next_f32);
        let w1_data = make_tensor(&[num_experts, d_model, d_hidden], &mut next_f32);
        let w2_data = make_tensor(&[num_experts, d_hidden, d_model], &mut next_f32);

        let wg = model.add_const("wg", wg_data)?;
        let w1 = model.add_const("w1", w1_data)?;
        let w2 = model.add_const("w2", w2_data)?;

        let mut inputs = vec![x, wg, w1, w2];

        if has_w3 {
            let w3_data = make_tensor(&[num_experts, d_model, d_hidden], &mut next_f32);
            let w3 = model.add_const("w3", w3_data)?;
            inputs.push(w3);
        }

        let op = MoeFfn {
            k,
            activation: "silu".to_string(),
            normalize_gates: true,
            has_w3,
        };
        let outputs = model.wire_node("moe", op, &inputs)?;
        model.set_output_outlets(&outputs)?;

        // Create input tensor
        let x_data = make_tensor(&[t_tokens, d_model], &mut next_f32);

        Ok((model, x_data))
    }

    #[test]
    fn test_opt_moe_ffn_matches_reference() -> TractResult<()> {
        // Test with SwiGLU (has_w3=true)
        let (model, x_data) = make_moe_model(8, 16, 32, 4, 2, true)?;

        // Run reference (unoptimized)
        let ref_plan = SimplePlan::new(model.clone())?;
        let ref_result = ref_plan.spawn()?.run(tvec![x_data.clone().into_tvalue()])?;

        // Run optimized
        let opt_model = model.into_optimized()?;

        // Verify MoeFfn was replaced with OptMoeFfn
        let has_opt = opt_model.nodes().iter().any(|n| n.op_is::<OptMoeFfn>());
        assert!(has_opt, "Expected OptMoeFfn in optimized model");

        let opt_plan = SimplePlan::new(opt_model)?;
        let opt_result = opt_plan.spawn()?.run(tvec![x_data.into_tvalue()])?;

        // Compare outputs
        ref_result[0].close_enough(&opt_result[0], Approximation::Approximate)?;
        ref_result[1].close_enough(&opt_result[1], Approximation::Approximate)?;

        Ok(())
    }

    #[test]
    fn test_opt_moe_ffn_no_w3() -> TractResult<()> {
        // Test without SwiGLU (has_w3=false)
        let (model, x_data) = make_moe_model(8, 16, 32, 4, 2, false)?;

        let ref_plan = SimplePlan::new(model.clone())?;
        let ref_result = ref_plan.spawn()?.run(tvec![x_data.clone().into_tvalue()])?;

        let opt_model = model.into_optimized()?;
        let opt_plan = SimplePlan::new(opt_model)?;
        let opt_result = opt_plan.spawn()?.run(tvec![x_data.into_tvalue()])?;

        ref_result[0].close_enough(&opt_result[0], Approximation::Approximate)?;
        ref_result[1].close_enough(&opt_result[1], Approximation::Approximate)?;

        Ok(())
    }

    #[test]
    fn test_opt_moe_ffn_top1() -> TractResult<()> {
        let (model, x_data) = make_moe_model(16, 8, 16, 8, 1, true)?;

        let ref_plan = SimplePlan::new(model.clone())?;
        let ref_result = ref_plan.spawn()?.run(tvec![x_data.clone().into_tvalue()])?;

        let opt_model = model.into_optimized()?;
        let opt_plan = SimplePlan::new(opt_model)?;
        let opt_result = opt_plan.spawn()?.run(tvec![x_data.into_tvalue()])?;

        ref_result[0].close_enough(&opt_result[0], Approximation::Approximate)?;
        ref_result[1].close_enough(&opt_result[1], Approximation::Approximate)?;

        Ok(())
    }

    #[test]
    fn test_codegen_fallback_on_non_const_weights() -> TractResult<()> {
        // When weights are inputs (not constants), codegen should not fire
        let mut model = TypedModel::default();
        let x = model.add_source("x", f32::datum_type().fact(&[4, 8]))?;
        let wg = model.add_source("wg", f32::datum_type().fact(&[2, 8]))?;
        let w1 = model.add_source("w1", f32::datum_type().fact(&[2, 8, 16]))?;
        let w2 = model.add_source("w2", f32::datum_type().fact(&[2, 16, 8]))?;

        let op = MoeFfn {
            k: 1,
            activation: "silu".to_string(),
            normalize_gates: true,
            has_w3: false,
        };
        let outputs = model.wire_node("moe", op, &[x, wg, w1, w2])?;
        model.set_output_outlets(&outputs)?;

        let opt_model = model.into_optimized()?;

        // Should still have MoeFfn (not OptMoeFfn)
        let has_moe = opt_model.nodes().iter().any(|n| n.op_is::<MoeFfn>());
        assert!(has_moe, "Expected MoeFfn to remain when weights are not constants");

        Ok(())
    }

    #[test]
    fn test_e2e_nnef_qwen3_moe() -> TractResult<()> {
        use crate::WithTractTransformers;
        use std::io::Cursor;

        // Load the Qwen3 MoE model exported from transformers
        let model_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../harness/nnef-test-cases/moe-ffn/qwen3-tiny");

        let nnef = tract_nnef::nnef().with_tract_transformers();
        let model = nnef.model_for_path(&model_path)?;
        let model = model.into_optimized()?;

        // Verify OptMoeFfn is present after optimization
        let has_opt = model.nodes().iter().any(|n: &TypedNode| n.op_is::<OptMoeFfn>());
        assert!(has_opt, "Expected OptMoeFfn in optimized model");

        let plan = SimplePlan::new(model)?;

        // Load input and expected output from io.npz
        let npz_path = model_path.join("io.npz");
        let npz_bytes = std::fs::read(&npz_path)?;
        let mut npz = ndarray_npy::NpzReader::new(Cursor::new(npz_bytes))?;

        let input: tract_ndarray::ArrayD<f32> = npz.by_name("input_0.npy")?;
        let expected_output: tract_ndarray::ArrayD<f32> = npz.by_name("output_0.npy")?;

        // Run inference
        let result = plan.spawn()?.run(tvec![input.into_tensor().into_tvalue()])?;

        // Compare against PyTorch reference output
        result[0].close_enough(&expected_output.into_tensor(), Approximation::Approximate)?;

        Ok(())
    }
}
