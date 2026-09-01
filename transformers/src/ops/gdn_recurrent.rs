use tract_nnef::internal::*;

use super::cast_f32::{from_f32, to_f32_vec};

pub fn register(registry: &mut Registry) {
    fn deserialize(
        builder: &mut ModelBuilder,
        invocation: &ResolvedInvocation,
    ) -> TractResult<Value> {
        let inputs = ["query", "key", "value", "log_decay", "beta", "initial_state"]
            .map(|name| invocation.named_arg_as(builder, name))
            .into_iter()
            .collect::<TractResult<TVec<_>>>()?;
        builder.wire(GatedDeltaNetRecurrent::default(), &inputs)
    }
    fn serialize(
        ast: &mut IntoAst,
        node: &TypedNode,
        op: &GatedDeltaNetRecurrent,
    ) -> TractResult<Option<Arc<RValue>>> {
        // sigmoid_beta is a device-transform-internal variant (the beta
        // sigmoid folded into the kernel); it has no NNEF form.
        ensure!(
            !op.sigmoid_beta,
            "GatedDeltaNetRecurrent with sigmoid_beta cannot be serialized to NNEF"
        );
        let inputs: Vec<Arc<RValue>> =
            node.inputs.iter().map(|i| ast.mapping[i].clone()).collect();
        Ok(Some(invocation("tract_transformers_gdn_recurrent", &inputs, &[])))
    }
    registry.register_dumper(serialize);
    // Generic name first (primary, what serialization emits); the historical
    // qwen35-specific name stays as a deserialization alias.
    for name in ["tract_transformers_gdn_recurrent", "tract_qwen35_gdn_recurrent"] {
        registry.register_primitive(
            name,
            &[
                TypeName::Scalar.tensor().named("query"),
                TypeName::Scalar.tensor().named("key"),
                TypeName::Scalar.tensor().named("value"),
                TypeName::Scalar.tensor().named("log_decay"),
                TypeName::Scalar.tensor().named("beta"),
                TypeName::Scalar.tensor().named("initial_state"),
            ],
            &[("output", TypeName::Scalar.tensor()), ("final_state", TypeName::Scalar.tensor())],
            deserialize,
        );
    }
}

/// Gated delta rule recurrence (Qwen3.5 linear attention core).
///
/// Semantics match HF transformers `torch_recurrent_gated_delta_rule` with
/// `use_qk_l2norm_in_kernel=True`, applied sequentially over the S axis:
///
/// per step: `state *= exp(g)`; `kv_mem = k . state` (over the key axis);
/// `delta = (v - kv_mem) * beta`; `state += k (x) delta`;
/// `out = (q / sqrt(w)) . state`, with q and k L2-normalized (eps 1e-6).
///
/// Layout: query/key `[b, S, hk, w]`, value `[b, S, hv, w]` (key width ==
/// value width), log_decay/beta `[b, S, hv]`, initial_state `[b, hv, w, w]`,
/// with `hv = G * hk` for an integer group count G resolved from the shapes
/// (GQA: value head h reads query/key head h / G, matching HF's
/// `repeat_interleave` on axis 2). hk == hv is the ungrouped case, so
/// pre-GQA graphs load and run unchanged. Outputs: `[b, S, hv, w]` in the
/// query datum type and the final state in the initial_state datum type.
/// All compute is f32.
///
/// With `sigmoid_beta` set the `beta` input carries the raw pre-activation
/// values and the op applies the sigmoid itself: runtimes can then fold the
/// singleton sigmoid dispatch that otherwise feeds beta into their kernel.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct GatedDeltaNetRecurrent {
    pub sigmoid_beta: bool,
}

impl Op for GatedDeltaNetRecurrent {
    fn name(&self) -> StaticName {
        "GatedDeltaNetRecurrent".into()
    }
    fn info(&self) -> TractResult<Vec<String>> {
        Ok(if self.sigmoid_beta { vec!["sigmoid_beta: true".to_string()] } else { vec![] })
    }
    op_as_typed_op!();
}


impl EvalOp for GatedDeltaNetRecurrent {
    op_out_of_plan!();

    fn eval(&self, _ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        ensure!(inputs.len() == 6, "GDN expects q, k, v, log_decay, beta, state");
        let q_shape: TVec<usize> = inputs[0].shape().into();
        let v_shape: TVec<usize> = inputs[2].shape().into();
        let state_shape: TVec<usize> = inputs[5].shape().into();
        ensure!(
            q_shape.len() == 4,
            "GDN query must be [b, S, hk, w], got {q_shape:?}"
        );
        ensure!(inputs[1].shape() == &*q_shape);
        let (b, s_len, k_heads, width) = (q_shape[0], q_shape[1], q_shape[2], q_shape[3]);
        ensure!(
            v_shape.len() == 4
                && v_shape[0] == b
                && v_shape[1] == s_len
                && v_shape[3] == width
                && v_shape[2].is_multiple_of(k_heads),
            "GDN value must be [b, S, G*hk, w] with query/key [b, S, hk, w], \
             got value {v_shape:?} vs query {q_shape:?}"
        );
        let heads = v_shape[2];
        let groups = heads / k_heads;
        ensure!(
            inputs[3].len() == b * s_len * heads && inputs[4].len() == b * s_len * heads,
            "GDN log_decay/beta must have b*S*hv elements"
        );
        ensure!(
            state_shape.len() == 4
                && state_shape[0] == b
                && state_shape[1] == heads
                && state_shape[2] == width
                && state_shape[3] == width,
            "GDN state must be [b, hv, w, w], got {state_shape:?}"
        );

        let q = to_f32_vec(&inputs[0])?;
        let k = to_f32_vec(&inputs[1])?;
        let v = to_f32_vec(&inputs[2])?;
        let g = to_f32_vec(&inputs[3])?;
        let mut beta = to_f32_vec(&inputs[4])?;
        if self.sigmoid_beta {
            for b in beta.iter_mut() {
                *b = 1.0 / (1.0 + (-*b).exp());
            }
        }
        let mut state = to_f32_vec(&inputs[5])?;

        let scale: f32 = 1.0 / (width as f32).sqrt();
        let mut output = vec![0f32; v.len()];
        let mut qn = vec![0f32; width];
        let mut kn = vec![0f32; width];
        for bi in 0..b {
            for si in 0..s_len {
                for h in 0..heads {
                    let vb = ((bi * s_len + si) * heads + h) * width;
                    // GQA: value head h reads query/key head h / groups.
                    let qkb = ((bi * s_len + si) * k_heads + h / groups) * width;
                    let sb = (bi * heads + h) * width * width;
                    let gb = (bi * s_len + si) * heads + h;
                    let q_inv = 1.0
                        / (q[qkb..qkb + width].iter().map(|x| x * x).sum::<f32>() + 1e-6).sqrt();
                    let k_inv = 1.0
                        / (k[qkb..qkb + width].iter().map(|x| x * x).sum::<f32>() + 1e-6).sqrt();
                    for i in 0..width {
                        qn[i] = q[qkb + i] * q_inv * scale;
                        kn[i] = k[qkb + i] * k_inv;
                    }
                    let decay = g[gb].exp();
                    let bta = beta[gb];
                    for col in 0..width {
                        let mut kv_mem = 0f32;
                        for row in 0..width {
                            kv_mem += kn[row] * state[sb + row * width + col] * decay;
                        }
                        let delta = (v[vb + col] - kv_mem) * bta;
                        let mut result = 0f32;
                        for row in 0..width {
                            let ix = sb + row * width + col;
                            let next = state[ix] * decay + kn[row] * delta;
                            state[ix] = next;
                            result += qn[row] * next;
                        }
                        output[vb + col] = result;
                    }
                }
            }
        }
        Ok(tvec![
            from_f32(output, &v_shape, inputs[0].datum_type())?.into_tvalue(),
            from_f32(state, &state_shape, inputs[5].datum_type())?.into_tvalue(),
        ])
    }
}

impl TypedOp for GatedDeltaNetRecurrent {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(inputs.len() == 6);
        let dts: Vec<DatumType> = inputs.iter().map(|i| i.datum_type).collect();
        // The two intended combinations, matching what the GPU kernels
        // accept (falling back to this CPU op, which upcasts to f32
        // internally regardless, for anything else): uniformly f32, or the
        // fused-kernel mix (query/key/value/beta f16, log_decay f32, state
        // either f16 or f32).
        let all_f32 = dts.iter().all(|dt| *dt == DatumType::F32);
        let fused_mix = dts[..4] == [DatumType::F16, DatumType::F16, DatumType::F16, DatumType::F32]
            && dts[4] == DatumType::F16
            && matches!(dts[5], DatumType::F16 | DatumType::F32);
        ensure!(all_f32 || fused_mix, "unsupported GDN dtype combination: {dts:?}");
        ensure!(inputs[0].rank() == 4, "GDN query must be [b, S, hk, w]");
        ensure!(inputs[0].shape == inputs[1].shape);
        ensure!(inputs[2].rank() == 4, "GDN value must be [b, S, hv, w]");
        // Output takes the VALUE shape (hv heads, possibly G * hk) with the
        // query datum type; head-count divisibility is checked at eval time
        // (dims may be symbolic here).
        ensure!(inputs[5].rank() == 4, "GDN state must be [b, hv, w, w]");
        let mut out = inputs[2].without_value();
        out.datum_type = inputs[0].datum_type;
        Ok(tvec![out, inputs[5].without_value()])
    }
    as_op!();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::test_utils::arb;

    fn run(
        s_len: usize,
        heads: usize,
        width: usize,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        g: &Tensor,
        beta: &Tensor,
        state: &Tensor,
    ) -> TractResult<(Tensor, Tensor)> {
        let _ = (s_len, heads, width);
        let outputs = GatedDeltaNetRecurrent::default().eval(tvec![
            q.clone().into_tvalue(),
            k.clone().into_tvalue(),
            v.clone().into_tvalue(),
            g.clone().into_tvalue(),
            beta.clone().into_tvalue(),
            state.clone().into_tvalue(),
        ])?;
        Ok((outputs[0].clone().into_tensor(), outputs[1].clone().into_tensor()))
    }

    /// repeat_interleave along the head axis (2) of a [b, S, h, w] tensor.
    fn repeat_heads(t: &Tensor, groups: usize) -> Tensor {
        let shape = t.shape();
        let (b, s_len, heads, width) = (shape[0], shape[1], shape[2], shape[3]);
        let src = t.to_plain_array_view::<f32>().unwrap();
        let src = src.as_slice().unwrap();
        let mut data = vec![0f32; b * s_len * heads * groups * width];
        for bi in 0..b {
            for si in 0..s_len {
                for h in 0..heads * groups {
                    let dst_base = ((bi * s_len + si) * heads * groups + h) * width;
                    let src_base = ((bi * s_len + si) * heads + h / groups) * width;
                    data[dst_base..dst_base + width]
                        .copy_from_slice(&src[src_base..src_base + width]);
                }
            }
        }
        Tensor::from_shape(&[b, s_len, heads * groups, width], &data).unwrap()
    }

    /// The S-axis loop must be exactly equivalent to threading the state
    /// through S single-step calls, for both the ungrouped (G=1) and the
    /// GQA (G=2) head layouts.
    fn multi_step_matches_sequential_single_steps_case(groups: usize) -> TractResult<()> {
        let (b, s_len, k_heads, width) = (1, 5, 3, 16);
        let heads = k_heads * groups;
        let q = arb(&[b, s_len, k_heads, width], 1);
        let k = arb(&[b, s_len, k_heads, width], 2);
        let v = arb(&[b, s_len, heads, width], 3);
        let g = arb(&[b, s_len, heads], 4);
        let beta = arb(&[b, s_len, heads], 5);
        let state0 = arb(&[b, heads, width, width], 6);

        let (out_multi, final_multi) = run(s_len, heads, width, &q, &k, &v, &g, &beta, &state0)?;

        let mut state = state0.clone();
        let mut outs: Vec<Tensor> = vec![];
        for si in 0..s_len {
            let slice = |t: &Tensor| t.slice(1, si, si + 1).unwrap();
            let (o, st) = run(
                1,
                heads,
                width,
                &slice(&q),
                &slice(&k),
                &slice(&v),
                &slice(&g),
                &slice(&beta),
                &state,
            )?;
            outs.push(o);
            state = st;
        }
        let seq_out = Tensor::stack_tensors(1, &outs.iter().map(|o| o.clone().into()).collect::<Vec<TValue>>())?;
        // stacked [b, S, 1, h, w] vs [b, S, h, w]: reshape
        let seq_out = seq_out.into_shape(&[b, s_len, heads, width])?;

        out_multi.close_enough(&seq_out, Approximation::Close)?;
        final_multi.close_enough(&state, Approximation::Close)?;
        Ok(())
    }

    #[test]
    fn multi_step_matches_sequential_single_steps() -> TractResult<()> {
        multi_step_matches_sequential_single_steps_case(1)
    }

    #[test]
    fn multi_step_matches_sequential_single_steps_grouped() -> TractResult<()> {
        multi_step_matches_sequential_single_steps_case(2)
    }

    /// Grouped q/k (hk heads) must give bitwise the same result as the old
    /// ungrouped call on repeat-interleaved q/k (hv heads), i.e. exactly
    /// what HF materializes before the op boundary.
    #[test]
    fn grouped_matches_repeated_reference() -> TractResult<()> {
        let (b, s_len, k_heads, groups, width) = (1, 4, 2, 2, 16);
        let heads = k_heads * groups;
        let q = arb(&[b, s_len, k_heads, width], 11);
        let k = arb(&[b, s_len, k_heads, width], 12);
        let v = arb(&[b, s_len, heads, width], 13);
        let g = arb(&[b, s_len, heads], 14);
        let beta = arb(&[b, s_len, heads], 15);
        let state0 = arb(&[b, heads, width, width], 16);

        let (out_grouped, state_grouped) =
            run(s_len, heads, width, &q, &k, &v, &g, &beta, &state0)?;
        let q_rep = repeat_heads(&q, groups);
        let k_rep = repeat_heads(&k, groups);
        let (out_ref, state_ref) =
            run(s_len, heads, width, &q_rep, &k_rep, &v, &g, &beta, &state0)?;

        out_grouped.close_enough(&out_ref, Approximation::Exact)?;
        state_grouped.close_enough(&state_ref, Approximation::Exact)?;
        Ok(())
    }
}
