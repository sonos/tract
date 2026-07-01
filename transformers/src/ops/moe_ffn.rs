use std::hash::{Hash, Hasher};
use std::sync::Arc;

use tract_ndarray::{Array2, ArrayView2, ArrayViewD, Axis, s};
use tract_nnef::internal::*;
use tract_nnef::tract_core::ops::einsum::EinSum;
use tract_nnef::tract_core::ops::konst::Const;
use tract_nnef::tract_core::ops::math::mul;
use tract_nnef::tract_core::ops::{FrozenOpState, OpStateFreeze};
use tract_nnef::tract_core::tract_linalg::block_quant::{BlockQuantStorage, block_quant_slice};

use super::gelu_approximate::gelu_approximate;
use super::silu::silu;

pub fn register(registry: &mut Registry) {
    registry.register_primitive(
        "tract_moe_ffn",
        &[
            TypeName::Scalar.tensor().named("x"),
            TypeName::Scalar.tensor().named("wg"),
            TypeName::Scalar.tensor().named("w1"),
            TypeName::Scalar.tensor().named("w2"),
            TypeName::Scalar.tensor().named("w3"),
            // Optional biases (e.g. gpt-oss): router bias [E], gate bias
            // (w1) [E, H], up bias (w3) [E, H], down bias (w2) [E, D].
            TypeName::Scalar.tensor().named("wg_bias"),
            TypeName::Scalar.tensor().named("w1_bias"),
            TypeName::Scalar.tensor().named("w3_bias"),
            TypeName::Scalar.tensor().named("w2_bias"),
            TypeName::Integer.named("k"),
            TypeName::String.named("activation"),
            // How router logits become the top-k gate weights:
            //   softmax_topk: softmax over the top-k logits (Mixtral, Qwen w/
            //                 norm_topk_prob, gpt-oss)
            //   softmax_all:  softmax over ALL experts, gather top-k, no
            //                 renormalization (OLMoE, Qwen w/o norm_topk_prob)
            //   sigmoid:      per-expert sigmoid of the top-k logits (Llama4)
            //   raw:          raw top-k logits as weights
            TypeName::String.named("gate"),
            // Optional clamped-SwiGLU params (gpt-oss): when act_limit is set
            // the activation becomes gate.clamp(max=limit) /
            // up.clamp(+-limit) / glu = gate*sigmoid(alpha*gate) /
            // out = (up + 1) * glu.
            TypeName::Scalar.named("act_alpha"),
            TypeName::Scalar.named("act_limit"),
        ],
        // single output: tract is inference-only, so router_logits (a
        // training-time load-balancing-loss signal) is computed internally for
        // expert selection but never surfaced as an output.
        &[("output", TypeName::Scalar.tensor())],
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
    let wg_bias: Option<OutletId> = invocation.get_named_arg_as(builder, "wg_bias")?;
    let w1_bias: Option<OutletId> = invocation.get_named_arg_as(builder, "w1_bias")?;
    let w3_bias: Option<OutletId> = invocation.get_named_arg_as(builder, "w3_bias")?;
    let w2_bias: Option<OutletId> = invocation.get_named_arg_as(builder, "w2_bias")?;
    let k: i64 = invocation.named_arg_as(builder, "k")?;
    let activation: String = invocation.named_arg_as(builder, "activation")?;
    let gate_str: String = invocation.named_arg_as(builder, "gate")?;
    let gate = GateMode::parse(&gate_str)?;
    let act_alpha: Option<f32> = invocation.get_named_arg_as(builder, "act_alpha")?;
    let act_limit: Option<f32> = invocation.get_named_arg_as(builder, "act_limit")?;

    // Inputs are pushed in a fixed order so eval can recover their positions
    // from the has_* flags: x, wg, w1, w2, [w3], [wg_bias], [w1_bias],
    // [w3_bias], [w2_bias].
    let mut inputs = vec![x, wg, w1, w2];
    let has_w3 = w3.is_some();
    let has_wg_bias = wg_bias.is_some();
    let has_w1_bias = w1_bias.is_some();
    let has_w3_bias = w3_bias.is_some();
    let has_w2_bias = w2_bias.is_some();
    for opt in [w3, wg_bias, w1_bias, w3_bias, w2_bias].into_iter().flatten() {
        inputs.push(opt);
    }

    builder.wire(
        MoeFfn {
            k: k as usize,
            activation,
            gate,
            has_w3,
            has_wg_bias,
            has_w1_bias,
            has_w3_bias,
            has_w2_bias,
            act_alpha_bits: act_alpha.map(f32::to_bits),
            act_limit_bits: act_limit.map(f32::to_bits),
        },
        &inputs,
    )
}

/// How router logits are turned into the top-k gate weights.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum GateMode {
    /// softmax over the top-k logits (Mixtral, Qwen norm_topk_prob, gpt-oss).
    SoftmaxTopk,
    /// softmax over ALL experts, gather the top-k, no renormalization (OLMoE).
    SoftmaxAll,
    /// per-expert sigmoid of the top-k logits, no normalization (Llama4).
    Sigmoid,
    /// raw top-k logits as weights.
    Raw,
}

impl GateMode {
    fn parse(s: &str) -> TractResult<Self> {
        match s {
            "softmax_topk" => Ok(GateMode::SoftmaxTopk),
            "softmax_all" => Ok(GateMode::SoftmaxAll),
            "sigmoid" => Ok(GateMode::Sigmoid),
            "raw" => Ok(GateMode::Raw),
            other => bail!("unknown tract_moe_ffn gate mode {other:?}"),
        }
    }

    /// Weights for the selected experts given the full logits row and the
    /// (expert_id, logit) pairs chosen by top-k.
    fn weights(&self, full_logits: &[f32], selected: &[(usize, f32)]) -> Vec<f32> {
        match self {
            GateMode::Raw => selected.iter().map(|(_, l)| *l).collect(),
            GateMode::Sigmoid => {
                selected.iter().map(|(_, l)| 1.0 / (1.0 + (-l).exp())).collect()
            }
            GateMode::SoftmaxTopk => {
                let m = selected
                    .iter()
                    .map(|(_, l)| *l)
                    .fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = selected.iter().map(|(_, l)| (l - m).exp()).collect();
                let sum: f32 = exps.iter().sum();
                exps.iter().map(|e| e / sum).collect()
            }
            GateMode::SoftmaxAll => {
                let m = full_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let denom: f32 = full_logits.iter().map(|l| (l - m).exp()).sum();
                selected.iter().map(|(_, l)| (l - m).exp() / denom).collect()
            }
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct MoeFfn {
    pub k: usize,
    pub activation: String,
    pub gate: GateMode,
    pub has_w3: bool,
    pub has_wg_bias: bool,
    pub has_w1_bias: bool,
    pub has_w3_bias: bool,
    pub has_w2_bias: bool,
    // f32 has no Hash/Eq, so the clamped-SwiGLU params are stored as bit
    // patterns; None means "use the plain activation_op path".
    pub act_alpha_bits: Option<u32>,
    pub act_limit_bits: Option<u32>,
}

/// Resolved positions of the optional inputs within the `inputs` slice,
/// derived from the `has_*` flags (see `deser_moe_ffn` for the push order).
struct MoeInputIdx {
    w3: Option<usize>,
    wg_bias: Option<usize>,
    w1_bias: Option<usize>,
    w3_bias: Option<usize>,
    w2_bias: Option<usize>,
}

impl MoeFfn {
    /// Bias-free, plain-activation op for unit tests (kept future-proof
    /// against new optional fields).
    #[cfg(test)]
    fn basic(k: usize, activation: &str, gate: GateMode, has_w3: bool) -> Self {
        MoeFfn {
            k,
            activation: activation.to_string(),
            gate,
            has_w3,
            has_wg_bias: false,
            has_w1_bias: false,
            has_w3_bias: false,
            has_w2_bias: false,
            act_alpha_bits: None,
            act_limit_bits: None,
        }
    }

    fn is_clamped_act(&self) -> bool {
        self.act_limit_bits.is_some()
    }

    fn has_any_bias(&self) -> bool {
        self.has_wg_bias || self.has_w1_bias || self.has_w3_bias || self.has_w2_bias
    }

    fn act_alpha(&self) -> f32 {
        self.act_alpha_bits.map(f32::from_bits).unwrap_or(1.0)
    }

    fn act_limit(&self) -> f32 {
        self.act_limit_bits.map(f32::from_bits).unwrap_or(f32::INFINITY)
    }

    fn input_idx(&self) -> MoeInputIdx {
        let mut i = 4;
        let mut next = |present: bool| {
            if present {
                let v = i;
                i += 1;
                Some(v)
            } else {
                None
            }
        };
        MoeInputIdx {
            w3: next(self.has_w3),
            wg_bias: next(self.has_wg_bias),
            w1_bias: next(self.has_w1_bias),
            w3_bias: next(self.has_w3_bias),
            w2_bias: next(self.has_w2_bias),
        }
    }
}

impl Op for MoeFfn {
    fn name(&self) -> StaticName {
        "MoeFfn".to_string().into()
    }
    op_as_typed_op!();
}

/// Scatter-add expert outputs into `output`, weighted by the gate weights.
/// Accumulation is in f32: the router and experts run in f32 (matching
/// PyTorch's CPU f32-upcast of f16 matmuls), so the whole op is f32 internally
/// and only the final output is cast back to the model dtype.
fn scatter_add_weighted(
    output: &mut Array2<f32>,
    y_view: &ArrayView2<f32>,
    tokens: &[(usize, f32)],
) {
    for (i, &(t, gw)) in tokens.iter().enumerate() {
        let y_row = y_view.row(i);
        let mut out_row = output.row_mut(t);
        out_row.scaled_add(gw, &y_row);
    }
}

/// Add a per-expert bias row (`bias_full[eid]`, length H or D) to every row of
/// `arr` ([n, H] or [n, D]).
fn add_expert_bias(
    arr: &mut Array2<f32>,
    bias_full: &ArrayViewD<f32>,
    eid: usize,
) -> TractResult<()> {
    let brow = bias_full.index_axis(Axis(0), eid);
    let bslice = brow.as_slice().context("expert bias row not contiguous")?;
    for mut row in arr.rows_mut() {
        row.iter_mut().zip(bslice).for_each(|(v, &bv)| *v += bv);
    }
    Ok(())
}

impl EvalOp for MoeFfn {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        // inputs: x [T,D], wg [E,D] or [1,E,D], w1 [E,D,H], w2 [E,H,D],
        // then optionally w3 [E,D,H], wg_bias [E], w1_bias [E,H],
        // w3_bias [E,H], w2_bias [E,D] (order set in deser_moe_ffn).
        let idx = self.input_idx();
        let orig_dt = inputs[0].datum_type();
        // The reference path computes in f32. Cast every input up front (this
        // upcasts f16/bf16 weights, so it is the un-optimized fallback; the
        // codegen optimized path keeps experts in their native dtype).
        let cast = |i: usize| -> TractResult<Tensor> {
            let input = &inputs[i];
            if input.is_plain() {
                return Ok(input.cast_to::<f32>()?.into_owned());
            }
            let s = input.shape();
            let k = *s.last().context("block-quant tensor has no last axis")?;
            // Leading dims are group/batch dims; last two are [M, K].
            let num_groups: usize = if s.len() > 2 { s[..s.len() - 2].iter().product() } else { 1 };
            let m_per_group: usize = if s.len() >= 2 { s[s.len() - 2] } else { 1 };
            let bqs = input.try_storage_as::<BlockQuantStorage>()?;
            let mut unpacked = (0..num_groups)
                .map(|g| {
                    let slice = block_quant_slice(bqs.value(), bqs.format(), m_per_group, k, g);
                    bqs.format().dequant_f32(slice)
                })
                .collect::<TractResult<Vec<_>>>()?;
            unpacked.iter_mut().try_for_each(|t| t.insert_axis(0))?;
            let stacked = if unpacked.len() > 1 {
                Tensor::stack_tensors(0, &unpacked)?
            } else {
                unpacked.into_iter().next().unwrap()
            };
            stacked.into_shape(s)
        };
        let x_t = cast(0)?;
        let wg_t = cast(1)?;
        let w1_t = cast(2)?;
        let w2_t = cast(3)?;
        let w3_t = idx.w3.map(cast).transpose()?;
        let wg_bias_t = idx.wg_bias.map(cast).transpose()?;
        let w1_bias_t = idx.w1_bias.map(cast).transpose()?;
        let w3_bias_t = idx.w3_bias.map(cast).transpose()?;
        let w2_bias_t = idx.w2_bias.map(cast).transpose()?;
        let x = x_t.to_plain_array_view::<f32>()?;
        let wg_raw = wg_t.to_plain_array_view::<f32>()?;
        let w1 = w1_t.to_plain_array_view::<f32>()?;
        let w2 = w2_t.to_plain_array_view::<f32>()?;
        let w3 = w3_t
            .as_ref()
            .map(|t| t.to_plain_array_view::<f32>())
            .transpose()?;
        let wg_bias = wg_bias_t
            .as_ref()
            .map(|t| t.to_plain_array_view::<f32>())
            .transpose()?;
        let w1_bias = w1_bias_t
            .as_ref()
            .map(|t| t.to_plain_array_view::<f32>())
            .transpose()?;
        let w3_bias = w3_bias_t
            .as_ref()
            .map(|t| t.to_plain_array_view::<f32>())
            .transpose()?;
        let w2_bias = w2_bias_t
            .as_ref()
            .map(|t| t.to_plain_array_view::<f32>())
            .transpose()?;

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
            x.into_shape_with_order((x_orig_shape[0] * x_orig_shape[1], x_orig_shape[2]))?
                .into_dimensionality()?
        } else {
            x.into_dimensionality()?
        };

        let t_tokens = x.shape()[0];
        let d_model = x.shape()[1];
        let num_experts = wg.shape()[0];
        let _d_hidden = w1.shape()[2];

        // ---- Step 1: Router ----
        // logits = x @ wg.T  [T, D] @ [D, E] -> [T, E]  (+ optional bias [E])
        let mut router_logits: Array2<f32> = x.dot(&wg.t());
        if let Some(ref b) = wg_bias {
            let bias = b.as_slice().context("wg_bias not contiguous")?;
            for mut row in router_logits.rows_mut() {
                row.iter_mut().zip(bias).for_each(|(v, &bv)| *v += bv);
            }
        }

        // ---- Step 2: Top-k selection + gate weights per token ----
        // assignments[token] = Vec<(expert_id, gate_weight)>
        let mut assignments: Vec<Vec<(usize, f32)>> = Vec::with_capacity(t_tokens);
        for t in 0..t_tokens {
            let row = router_logits.row(t);
            let full: Vec<f32> = row.iter().copied().collect();
            let mut scores: Vec<(usize, f32)> =
                row.iter().enumerate().map(|(e, &s)| (e, s)).collect();
            scores.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
            scores.truncate(self.k);

            let gate_weights = self.gate.weights(&full, &scores);

            assignments
                .push(scores.iter().zip(gate_weights).map(|((eid, _), gw)| (*eid, gw)).collect());
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

        for (eid, tokens) in expert_tokens.iter().enumerate() {
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

            // gate branch: h = x_batch @ w1_e (+ w1_bias)  -> [n, H]
            let mut h: Array2<f32> = x_batch.dot(&w1_e);
            if let Some(ref b) = w1_bias {
                add_expert_bias(&mut h, b, eid)?;
            }

            if self.is_clamped_act() {
                // gpt-oss clamped SwiGLU: gate = w1 branch, up = w3 branch.
                //   gate = clamp(gate, max=limit)
                //   up   = clamp(up, -limit, limit)
                //   glu  = gate * sigmoid(alpha * gate)
                //   h    = (up + 1) * glu
                let w3 = w3.as_ref().context("clamped activation requires w3 (up branch)")?;
                let w3_e = w3.slice(s![eid, .., ..]); // [D, H]
                let mut up: Array2<f32> = x_batch.dot(&w3_e);
                if let Some(ref b) = w3_bias {
                    add_expert_bias(&mut up, b, eid)?;
                }
                let alpha = self.act_alpha();
                let limit = self.act_limit();
                h.iter_mut().zip(up.iter()).for_each(|(g, &u)| {
                    let gate = g.min(limit);
                    let up = u.clamp(-limit, limit);
                    let glu = gate / (1.0 + (-alpha * gate).exp());
                    *g = (up + 1.0) * glu;
                });
            } else if let Some(ref w3) = w3 {
                // SwiGLU: h = silu(h) * (x_batch @ w3_e + w3_bias)
                let w3_e = w3.slice(s![eid, .., ..]); // [D, H]
                let mut gate: Array2<f32> = x_batch.dot(&w3_e); // [n, H]
                if let Some(ref b) = w3_bias {
                    add_expert_bias(&mut gate, b, eid)?;
                }
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

            // y_expert = h @ w2_e (+ w2_bias)  -> [n, D]  (BLAS-backed GEMM)
            let mut y_expert: Array2<f32> = h.dot(&w2_e);
            if let Some(ref b) = w2_bias {
                add_expert_bias(&mut y_expert, b, eid)?;
            }

            // ---- Step 5: Scatter-add weighted results back ----
            scatter_add_weighted(&mut output, &y_expert.view(), tokens);
        }

        // Restore original rank if input was 3D, and the original dtype.
        let output_tensor = if x_ndim == 3 {
            output.into_shape_with_order((x_orig_shape[0], x_orig_shape[1], d_model))?.into_tensor()
        } else {
            output.into_tensor()
        };
        let output_tensor = output_tensor.cast_to_dt(orig_dt)?.into_owned();
        Ok(tvec![output_tensor.into_tvalue()])
    }
}

impl TypedOp for MoeFfn {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        // single output: same shape as input x
        let x_fact = inputs[0];
        let output_fact = x_fact.datum_type.fact(x_fact.shape.clone());
        Ok(tvec!(output_fact))
    }

    fn codegen(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        // The optimized plan-based path only models the bias-free, plain
        // activation case. gpt-oss (biases + clamped SwiGLU) stays on the
        // reference eval path, which is correct if slower.
        if self.has_any_bias() || self.is_clamped_act() {
            return Ok(None);
        }
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

        // Optimized sub-plans slice each expert tensor and build plain
        // constants. Q40 NNEF variables use packed storage, so leave them on the
        // reference eval path for now instead of rejecting the model at codegen.
        if !w1_tensor.is_plain()
            || !w2_tensor.is_plain()
            || w3_tensor.as_ref().is_some_and(|t| !t.is_plain())
        {
            return Ok(None);
        }

        // Bail if the activation is not supported by the optimized path
        if activation_op(&self.activation, self.has_w3).is_none() {
            return Ok(None);
        }

        let num_experts = w1_tensor.shape()[0];
        let d_model = w1_tensor.shape()[1];
        let d_hidden = w1_tensor.shape()[2];

        // Build router plan: x [T, D] @ wg.T -> [T, E]
        let router_plan =
            build_router_plan(&wg_tensor, &model.symbols).context("Building router plan")?;

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
                &model.symbols,
            )
            .with_context(|| format!("Building expert plan for expert {eid}"))?;
            expert_plans.push(plan);
        }

        let opt_op = OptMoeFfn {
            k: self.k,
            gate: self.gate.clone(),
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
        Ok(Some(patch))
    }

    as_op!();
}

// ---------------------------------------------------------------------------
// Activation helper
// ---------------------------------------------------------------------------

fn activation_op(name: &str, has_w3: bool) -> Option<Box<dyn TypedOp>> {
    match name {
        "silu" => Some(Box::new(silu())),
        // SwiGLU: the inner activation is silu, w3 provides the gate branch
        "swiglu" if has_w3 => Some(Box::new(silu())),
        "gelu" => Some(Box::new(gelu_approximate(false))),
        "relu" => Some(Box::new(tract_nnef::tract_core::ops::nn::leaky_relu(0.0))),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Sub-model builders
// ---------------------------------------------------------------------------

fn build_router_plan(
    wg: &Arc<Tensor>,
    symbols: &SymbolScope,
) -> TractResult<Arc<TypedSimplePlan>> {
    let mut model = TypedModel { symbols: symbols.clone(), ..Default::default() };
    let n_sym = symbols.sym("moe_t");

    // The router runs in f32 regardless of the model dtype: expert selection
    // is a discrete top-k, so f16 rounding in the router matmul would flip
    // which experts borderline tokens pick versus PyTorch (which upcasts f16
    // matmuls to f32 on CPU), making a few tokens diverge completely and
    // compound across layers. The router weight is tiny so f32 is cheap.
    let dt = f32::datum_type();

    // wg is [E, D] or [1, E, D] — normalize to [E, D], in f32.
    let wg_2d = if wg.rank() == 3 {
        wg.slice(0, 0, 1)?.into_shape(&[wg.shape()[1], wg.shape()[2]])?
    } else {
        (**wg).clone()
    };
    let wg_2d = wg_2d.cast_to::<f32>()?.into_owned();

    let d_model = wg_2d.shape()[1];

    // x: [T, D]
    let x = model.add_source("x", dt.fact([n_sym.to_dim(), d_model.to_dim()]))?;
    // wg: [E, D] as constant
    let wg_const = model.add_const("wg", wg_2d)?;

    // router_logits = x @ wg.T  -> [T, E]
    // EinSum: "ij,kj->ik" means i=T, j=D (contracted), k=E
    let axes: AxesMapping = "ij,kj->ik".parse()?;
    let logits = model.wire_node("router_logits", EinSum::new(axes, dt), &[x, wg_const])?[0];

    model.select_output_outlets(&[logits])?;
    SimplePlan::new(model.into_optimized()?)
}

fn build_expert_plan(
    w1: &Tensor,
    w2: &Tensor,
    w3: Option<&Tensor>,
    activation: &str,
    symbols: &SymbolScope,
) -> TractResult<Arc<TypedSimplePlan>> {
    let mut model = TypedModel { symbols: symbols.clone(), ..Default::default() };
    let n_sym = symbols.sym("moe_n");

    // Experts run in f32 so the matmuls accumulate in f32, matching PyTorch's
    // CPU behaviour (it upcasts f16 matmuls to f32). Native-f16 accumulation
    // diverged enough to derail greedy decoding. Weights are upcast to f32
    // const here; the NNEF file stays f16, so this is purely a runtime choice
    // (it does roughly double the in-memory expert size).
    let dt = f32::datum_type();
    let to_f32 = |t: &Tensor| -> TractResult<Tensor> { Ok(t.cast_to::<f32>()?.into_owned()) };

    let d_model = w1.shape()[0]; // w1: [D, H]

    // Input: x_batch [n, D]
    let x = model.add_source("x", dt.fact([n_sym.to_dim(), d_model.to_dim()]))?;

    // w1 matmul: x_batch [n,D] @ w1 [D,H] -> [n,H]
    let w1_const = model.add_const("w1", to_f32(w1)?)?;
    let axes_mm: AxesMapping = "ij,jk->ik".parse()?;
    let h = model.wire_node("w1_matmul", EinSum::new(axes_mm.clone(), dt), &[x, w1_const])?[0];

    // Activation (caller guarantees activation_op returns Some via codegen check)
    let act_op = activation_op(activation, w3.is_some())
        .ok_or_else(|| format_err!("Unsupported activation: {activation}"))?;
    let h = model.wire_node("activation", act_op, &[h])?[0];

    // Optional SwiGLU: gate = x @ w3, h = h * gate
    let h = if let Some(w3) = w3 {
        let w3_const = model.add_const("w3", to_f32(w3)?)?;
        let gate =
            model.wire_node("w3_matmul", EinSum::new(axes_mm.clone(), dt), &[x, w3_const])?[0];
        model.wire_node("swiglu_mul", mul(), &[h, gate])?[0]
    } else {
        h
    };

    // w2 matmul: h [n,H] @ w2 [H,D] -> [n,D]
    let w2_const = model.add_const("w2", to_f32(w2)?)?;
    let y = model.wire_node("w2_matmul", EinSum::new(axes_mm, dt), &[h, w2_const])?[0];

    model.select_output_outlets(&[y])?;
    SimplePlan::new(model.into_optimized()?)
}

// ---------------------------------------------------------------------------
// OptMoeFfn — optimized MoE FFN with pre-compiled expert sub-plans
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct OptMoeFfn {
    pub k: usize,
    pub gate: GateMode,
    pub num_experts: usize,
    pub d_model: usize,
    pub d_hidden: usize,
    pub router_plan: Arc<TypedSimplePlan>,
    pub expert_plans: Vec<Arc<TypedSimplePlan>>,
}

impl Hash for OptMoeFfn {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.k.hash(state);
        self.gate.hash(state);
        self.num_experts.hash(state);
        self.d_model.hash(state);
        self.d_hidden.hash(state);
    }
}

impl PartialEq for OptMoeFfn {
    fn eq(&self, other: &Self) -> bool {
        // Compare scalar config only; the compiled plans are derived from it.
        self.k == other.k
            && self.gate == other.gate
            && self.num_experts == other.num_experts
            && self.d_model == other.d_model
            && self.d_hidden == other.d_hidden
    }
}

impl Eq for OptMoeFfn {}

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
        let expert_states =
            self.expert_plans.iter().map(|p| p.spawn()).collect::<TractResult<Vec<_>>>()?;
        Ok(Some(Box::new(OptMoeFfnState { op: self.clone(), router_state, expert_states })))
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
        // The router/expert sub-plans run in the model dtype (e.g. f16) so the
        // big expert weights stay un-upcast; only the small per-token glue
        // (gather, top-k, gate weights, scatter) runs in f32.
        let dt = x_input.datum_type();
        let x_f32 = x_input.cast_to::<f32>()?;
        let x_view = x_f32.to_plain_array_view::<f32>()?;
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

        // ---- Step 1: Router via pre-spawned state (f32, see build_router_plan) ----
        let x_2d_tensor =
            x_f32.clone().into_owned().into_shape(&[t_tokens, d_model])?;

        let router_result = self.router_state.run(tvec![x_2d_tensor.into_tvalue()])?;
        let router_logits_f32 = router_result[0].cast_to::<f32>()?;
        let router_logits: ArrayView2<f32> =
            router_logits_f32.to_plain_array_view::<f32>()?.into_dimensionality()?;

        // ---- Step 2: Top-k selection + gate weights ----
        let mut assignments: Vec<Vec<(usize, f32)>> = Vec::with_capacity(t_tokens);
        for t in 0..t_tokens {
            let row = router_logits.row(t);
            let full: Vec<f32> = row.iter().copied().collect();
            let mut scores: Vec<(usize, f32)> =
                row.iter().enumerate().map(|(e, &s)| (e, s)).collect();
            scores.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
            scores.truncate(op.k);

            let gate_weights = op.gate.weights(&full, &scores);

            assignments
                .push(scores.iter().zip(gate_weights).map(|((eid, _), gw)| (*eid, gw)).collect());
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

        for (eid, tokens) in expert_tokens.iter().enumerate() {
            if tokens.is_empty() {
                continue;
            }
            let n = tokens.len();

            // Gather: build x_batch [n, D] in f32 and feed the f32 expert plan.
            let mut x_batch = Tensor::zero_dt(f32::datum_type(), &[n, d_model])?;
            {
                let mut x_batch_plain = x_batch.try_as_plain_mut()?;
                let x_batch_slice = x_batch_plain.as_slice_mut::<f32>()?;
                for (i, &(t, _)) in tokens.iter().enumerate() {
                    let src = x.row(t);
                    x_batch_slice[i * d_model..(i + 1) * d_model]
                        .copy_from_slice(src.as_slice().unwrap());
                }
            }

            // Run expert plan (f32; reusing pre-spawned state)
            let y_expert = self.expert_states[eid].run(tvec![x_batch.into_tvalue()])?;
            let y_view: ArrayView2<f32> =
                y_expert[0].to_plain_array_view::<f32>()?.into_dimensionality()?;
            scatter_add_weighted(&mut output, &y_view, tokens);
        }

        // ---- Restore shapes + dtype ----
        let output_tensor = if x_ndim == 3 {
            output.into_shape_with_order((x_orig_shape[0], x_orig_shape[1], d_model))?.into_tensor()
        } else {
            output.into_tensor()
        };
        let output_tensor = output_tensor.cast_to_dt(dt)?.into_owned();
        Ok(tvec![output_tensor.into_tvalue()])
    }
}

impl TypedOp for OptMoeFfn {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let x_fact = inputs[0];
        let output_fact = x_fact.datum_type.fact(x_fact.shape.clone());
        Ok(tvec!(output_fact))
    }

    as_op!();
}

#[cfg(test)]
mod tests {
    use super::*;
    use tract_nnef::tract_core::tract_linalg::block_quant::{
        BlockQuant, BlockQuantFact, BlockQuantStorage, Q4_0,
    };

    fn make_moe_model(
        t_tokens: usize,
        d_model: usize,
        d_hidden: usize,
        num_experts: usize,
        k: usize,
        has_w3: bool,
    ) -> TractResult<(TypedModel, Tensor)> {
        let mut model = TypedModel::default();

        let x = model.add_source("x", f32::datum_type().fact([t_tokens, d_model]))?;

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

        let op = MoeFfn::basic(k, "silu", GateMode::SoftmaxTopk, has_w3);
        let outputs = model.wire_node("moe", op, &inputs)?;
        model.select_output_outlets(&outputs)?;

        // Create input tensor
        let x_data = make_tensor(&[t_tokens, d_model], &mut next_f32);

        Ok((model, x_data))
    }

    fn add_q40_const(model: &mut TypedModel, name: &str, tensor: Tensor) -> TractResult<OutletId> {
        let shape = tensor.shape().to_vec();
        let k = *shape.last().context("Q40 tensor has no last axis")?;
        ensure!(k % Q4_0.block_len() == 0, "Q40 K axis must be a multiple of 32");
        let m = shape[..shape.len() - 1].iter().product();
        let quant = Q4_0.quant_f32(tensor.try_as_plain()?.as_slice::<f32>()?)?;
        let storage = BlockQuantStorage::new(Box::new(Q4_0), m, k, Arc::new(quant))?;
        let packed = Arc::new(storage.into_tensor_with_shape(f32::datum_type(), &shape));
        let fact = BlockQuantFact::new(Box::new(Q4_0), shape.iter().copied().collect());
        Ok(model.wire_node(name, Const::new_with_exotic_fact(packed, Box::new(fact))?, &[])?[0])
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

        Ok(())
    }

    #[test]
    fn test_codegen_fallback_on_non_const_weights() -> TractResult<()> {
        // When weights are inputs (not constants), codegen should not fire
        let mut model = TypedModel::default();
        let x = model.add_source("x", f32::datum_type().fact([4, 8]))?;
        let wg = model.add_source("wg", f32::datum_type().fact([2, 8]))?;
        let w1 = model.add_source("w1", f32::datum_type().fact([2, 8, 16]))?;
        let w2 = model.add_source("w2", f32::datum_type().fact([2, 16, 8]))?;

        let op = MoeFfn::basic(1, "silu", GateMode::SoftmaxTopk, false);
        let outputs = model.wire_node("moe", op, &[x, wg, w1, w2])?;
        model.select_output_outlets(&outputs)?;

        let opt_model = model.into_optimized()?;

        // Should still have MoeFfn (not OptMoeFfn)
        let has_moe = opt_model.nodes().iter().any(|n| n.op_is::<MoeFfn>());
        assert!(has_moe, "Expected MoeFfn to remain when weights are not constants");

        Ok(())
    }

    #[test]
    fn test_moe_ffn_runs_with_q40_expert_constants() -> TractResult<()> {
        let mut model = TypedModel::default();
        let t_tokens = 4;
        let d_model = 32;
        let d_hidden = 64;
        let num_experts = 4;

        let x = model.add_source("x", f32::datum_type().fact([t_tokens, d_model]))?;

        let mut rng_state: u64 = 1337;
        let mut next_f32 = || -> f32 {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng_state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        };
        let make_tensor = |shape: &[usize], rng: &mut dyn FnMut() -> f32| -> Tensor {
            let n: usize = shape.iter().product();
            let data: Vec<f32> = (0..n).map(|_| rng()).collect();
            tract_ndarray::ArrayD::from_shape_vec(shape, data).unwrap().into_tensor()
        };

        let wg = model.add_const("wg", make_tensor(&[num_experts, d_model], &mut next_f32))?;
        let w1 = add_q40_const(
            &mut model,
            "w1",
            make_tensor(&[num_experts, d_model, d_hidden], &mut next_f32),
        )?;
        let w2 = add_q40_const(
            &mut model,
            "w2",
            make_tensor(&[num_experts, d_hidden, d_model], &mut next_f32),
        )?;
        let w3 = add_q40_const(
            &mut model,
            "w3",
            make_tensor(&[num_experts, d_model, d_hidden], &mut next_f32),
        )?;

        let op = MoeFfn::basic(2, "silu", GateMode::SoftmaxTopk, true);
        let outputs = model.wire_node("moe", op, &[x, wg, w1, w2, w3])?;
        model.select_output_outlets(&outputs)?;

        let opt_model = model.into_optimized()?;
        let has_moe = opt_model.nodes().iter().any(|n| n.op_is::<MoeFfn>());
        let has_opt = opt_model.nodes().iter().any(|n| n.op_is::<OptMoeFfn>());
        assert!(has_moe, "Expected Q40 experts to stay on MoeFfn reference eval");
        assert!(!has_opt, "Q40 experts should not use OptMoeFfn yet");

        let x_data = make_tensor(&[t_tokens, d_model], &mut next_f32);
        let result = SimplePlan::new(opt_model)?.spawn()?.run(tvec![x_data.into_tvalue()])?;
        let output = result[0].to_plain_array_view::<f32>()?;
        assert!(output.iter().all(|v| v.is_finite()));

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
