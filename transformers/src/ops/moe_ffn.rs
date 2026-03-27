use tract_ndarray::{s, Array2, ArrayView2, Axis};
use tract_nnef::internal::*;

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

    as_op!();
}
