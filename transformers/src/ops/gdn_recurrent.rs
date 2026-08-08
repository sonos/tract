use tract_nnef::internal::*;

pub fn register(registry: &mut Registry) {
    fn deserialize(
        builder: &mut ModelBuilder,
        invocation: &ResolvedInvocation,
    ) -> TractResult<Value> {
        let inputs = ["query", "key", "value", "log_decay", "beta", "initial_state"]
            .map(|name| invocation.named_arg_as(builder, name))
            .into_iter()
            .collect::<TractResult<TVec<_>>>()?;
        builder.wire(GatedDeltaNetRecurrent, &inputs)
    }
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
/// Layout: query/key/value `[b, S, h, w]` (heads already repeated to the
/// value-head count, key width == value width), log_decay/beta `[b, S, h]`,
/// initial_state `[b, h, w, w]`. Outputs: `[b, S, h, w]` in the query datum
/// type and the final state in the initial_state datum type. All compute is
/// f32.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GatedDeltaNetRecurrent;

impl Op for GatedDeltaNetRecurrent {
    fn name(&self) -> StaticName {
        "GatedDeltaNetRecurrent".into()
    }
    op_as_typed_op!();
}

fn to_f32_vec(t: &TValue) -> TractResult<Vec<f32>> {
    let cow = t.cast_to::<f32>()?;
    Ok(cow.to_plain_array_view::<f32>()?.iter().copied().collect())
}

fn from_f32(data: Vec<f32>, shape: &[usize], dt: DatumType) -> TractResult<Tensor> {
    let t = Tensor::from_shape(shape, &data)?;
    Ok(t.cast_to_dt(dt)?.into_owned())
}

impl EvalOp for GatedDeltaNetRecurrent {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        ensure!(inputs.len() == 6, "GDN expects q, k, v, log_decay, beta, state");
        let q_shape: TVec<usize> = inputs[0].shape().into();
        let state_shape: TVec<usize> = inputs[5].shape().into();
        ensure!(
            q_shape.len() == 4,
            "GDN query must be [b, S, h, w], got {q_shape:?}"
        );
        ensure!(inputs[1].shape() == &*q_shape && inputs[2].shape() == &*q_shape);
        let (b, s_len, heads, width) = (q_shape[0], q_shape[1], q_shape[2], q_shape[3]);
        ensure!(
            inputs[3].len() == b * s_len * heads && inputs[4].len() == b * s_len * heads,
            "GDN log_decay/beta must have b*S*h elements"
        );
        ensure!(
            state_shape.len() == 4
                && state_shape[0] == b
                && state_shape[1] == heads
                && state_shape[2] == width
                && state_shape[3] == width,
            "GDN state must be [b, h, w, w], got {state_shape:?}"
        );

        let q = to_f32_vec(&inputs[0])?;
        let k = to_f32_vec(&inputs[1])?;
        let v = to_f32_vec(&inputs[2])?;
        let g = to_f32_vec(&inputs[3])?;
        let beta = to_f32_vec(&inputs[4])?;
        let mut state = to_f32_vec(&inputs[5])?;

        let scale: f32 = 1.0 / (width as f32).sqrt();
        let mut output = vec![0f32; q.len()];
        let mut qn = vec![0f32; width];
        let mut kn = vec![0f32; width];
        for bi in 0..b {
            for si in 0..s_len {
                for h in 0..heads {
                    let vb = ((bi * s_len + si) * heads + h) * width;
                    let sb = (bi * heads + h) * width * width;
                    let gb = (bi * s_len + si) * heads + h;
                    let q_inv =
                        1.0 / (q[vb..vb + width].iter().map(|x| x * x).sum::<f32>() + 1e-6).sqrt();
                    let k_inv =
                        1.0 / (k[vb..vb + width].iter().map(|x| x * x).sum::<f32>() + 1e-6).sqrt();
                    for i in 0..width {
                        qn[i] = q[vb + i] * q_inv * scale;
                        kn[i] = k[vb + i] * k_inv;
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
            from_f32(output, &q_shape, inputs[0].datum_type())?.into_tvalue(),
            from_f32(state, &state_shape, inputs[5].datum_type())?.into_tvalue(),
        ])
    }
}

impl TypedOp for GatedDeltaNetRecurrent {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(inputs.len() == 6);
        for input in inputs {
            ensure!(
                matches!(input.datum_type, DatumType::F16 | DatumType::F32),
                "GDN inputs must be f16 or f32, got {:?}",
                input.datum_type
            );
        }
        ensure!(inputs[0].rank() == 4, "GDN query must be [b, S, h, w]");
        ensure!(inputs[0].shape == inputs[1].shape && inputs[0].shape == inputs[2].shape);
        ensure!(inputs[5].rank() == 4, "GDN state must be [b, h, w, w]");
        Ok(tvec![inputs[0].without_value(), inputs[5].without_value()])
    }
    as_op!();
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let outputs = GatedDeltaNetRecurrent.eval(tvec![
            q.clone().into_tvalue(),
            k.clone().into_tvalue(),
            v.clone().into_tvalue(),
            g.clone().into_tvalue(),
            beta.clone().into_tvalue(),
            state.clone().into_tvalue(),
        ])?;
        Ok((outputs[0].clone().into_tensor(), outputs[1].clone().into_tensor()))
    }

    fn arb(shape: &[usize], seed: u64) -> Tensor {
        // simple deterministic pseudo-random floats in [-1, 1]
        let len: usize = shape.iter().product();
        let mut x = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let data: Vec<f32> = (0..len)
            .map(|_| {
                x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((x >> 33) as f32 / (1u64 << 31) as f32) - 1.0
            })
            .collect();
        Tensor::from_shape(shape, &data).unwrap()
    }

    /// The S-axis loop must be exactly equivalent to threading the state
    /// through S single-step calls.
    #[test]
    fn multi_step_matches_sequential_single_steps() -> TractResult<()> {
        let (b, s_len, heads, width) = (1, 5, 3, 16);
        let q = arb(&[b, s_len, heads, width], 1);
        let k = arb(&[b, s_len, heads, width], 2);
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
}
