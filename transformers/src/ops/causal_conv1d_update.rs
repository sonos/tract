use tract_nnef::internal::*;

pub fn register(registry: &mut Registry) {
    fn deserialize(
        builder: &mut ModelBuilder,
        invocation: &ResolvedInvocation,
    ) -> TractResult<Value> {
        let inputs = ["input", "weight", "initial_state"]
            .map(|name| invocation.named_arg_as(builder, name))
            .into_iter()
            .collect::<TractResult<TVec<_>>>()?;
        builder.wire(CausalConv1dUpdate, &inputs)
    }
    fn serialize(
        ast: &mut IntoAst,
        node: &TypedNode,
        _op: &CausalConv1dUpdate,
    ) -> TractResult<Option<Arc<RValue>>> {
        let inputs: Vec<Arc<RValue>> =
            node.inputs.iter().map(|i| ast.mapping[i].clone()).collect();
        Ok(Some(invocation("tract_transformers_causal_conv1d_update", &inputs, &[])))
    }
    registry.register_dumper(serialize);
    // Generic name first (primary, what serialization emits); the historical
    // qwen35-specific name stays as a deserialization alias.
    for name in ["tract_transformers_causal_conv1d_update", "tract_qwen35_causal_conv1d_update"] {
        registry.register_primitive(
            name,
            &[
                TypeName::Scalar.tensor().named("input"),
                TypeName::Scalar.tensor().named("weight"),
                TypeName::Scalar.tensor().named("initial_state"),
            ],
            &[("output", TypeName::Scalar.tensor()), ("final_state", TypeName::Scalar.tensor())],
            deserialize,
        );
    }
}

/// Stateful causal depthwise conv1d + SiLU (Qwen3.5 linear attention front).
///
/// Semantics match HF transformers `torch_causal_conv1d_update`:
/// `full = concat(state, input)` along time; `output[t] = silu(sum_tap
/// weight[c, tap] * full[c, t + 1 + tap])` for the last S positions;
/// `final_state = full[..., -k:]`.
///
/// Layout: input `[b, C, S]`, weight `[C, k]`, state `[b, C, k]`.
/// Outputs: `[b, C, S]` in the input datum type, final state in the state
/// datum type. Compute is f32.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CausalConv1dUpdate;

impl Op for CausalConv1dUpdate {
    fn name(&self) -> StaticName {
        "CausalConv1dUpdate".into()
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

impl EvalOp for CausalConv1dUpdate {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        ensure!(inputs.len() == 3, "causal conv update expects input, weight, state");
        let input_shape: TVec<usize> = inputs[0].shape().into();
        let weight_shape: TVec<usize> = inputs[1].shape().into();
        let state_shape: TVec<usize> = inputs[2].shape().into();
        ensure!(input_shape.len() == 3, "input must be [b, C, S], got {input_shape:?}");
        let (b, channels, s_len) = (input_shape[0], input_shape[1], input_shape[2]);
        let kernel_width =
            *weight_shape.last().context("conv weight must have a kernel axis")?;
        ensure!(
            weight_shape.iter().product::<usize>() == channels * kernel_width,
            "weight must be [C, k], got {weight_shape:?}"
        );
        ensure!(
            state_shape.len() == 3
                && state_shape[0] == b
                && state_shape[1] == channels
                && state_shape[2] == kernel_width,
            "state must be [b, C, k], got {state_shape:?}"
        );

        let input = to_f32_vec(&inputs[0])?;
        let weight = to_f32_vec(&inputs[1])?;
        let state = to_f32_vec(&inputs[2])?;

        let mut output = vec![0f32; input.len()];
        let mut final_state = vec![0f32; state.len()];
        let mut full = vec![0f32; kernel_width + s_len];
        for bi in 0..b {
            for c in 0..channels {
                let ib = (bi * channels + c) * s_len;
                let sb = (bi * channels + c) * kernel_width;
                let wb = c * kernel_width;
                full[..kernel_width].copy_from_slice(&state[sb..sb + kernel_width]);
                full[kernel_width..].copy_from_slice(&input[ib..ib + s_len]);
                for t in 0..s_len {
                    let mut sum = 0f32;
                    for tap in 0..kernel_width {
                        sum += weight[wb + tap] * full[t + 1 + tap];
                    }
                    output[ib + t] = sum / (1.0 + (-sum).exp());
                }
                final_state[sb..sb + kernel_width]
                    .copy_from_slice(&full[s_len..s_len + kernel_width]);
            }
        }
        Ok(tvec![
            from_f32(output, &input_shape, inputs[0].datum_type())?.into_tvalue(),
            from_f32(final_state, &state_shape, inputs[2].datum_type())?.into_tvalue(),
        ])
    }
}

impl TypedOp for CausalConv1dUpdate {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(inputs.len() == 3);
        for input in inputs {
            ensure!(
                matches!(input.datum_type, DatumType::F16 | DatumType::F32),
                "causal conv update inputs must be f16 or f32, got {:?}",
                input.datum_type
            );
        }
        ensure!(inputs[0].rank() == 3, "input must be [b, C, S]");
        ensure!(inputs[2].rank() == 3, "state must be [b, C, k]");
        Ok(tvec![inputs[0].without_value(), inputs[2].without_value()])
    }
    as_op!();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arb(shape: &[usize], seed: u64) -> Tensor {
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

    fn run(
        input: &Tensor,
        weight: &Tensor,
        state: &Tensor,
    ) -> TractResult<(Tensor, Tensor)> {
        let outputs = CausalConv1dUpdate.eval(tvec![
            input.clone().into_tvalue(),
            weight.clone().into_tvalue(),
            state.clone().into_tvalue(),
        ])?;
        Ok((outputs[0].clone().into_tensor(), outputs[1].clone().into_tensor()))
    }

    /// The S-axis loop must be exactly equivalent to threading the state
    /// through S single-step calls.
    #[test]
    fn multi_step_matches_sequential_single_steps() -> TractResult<()> {
        let (b, channels, s_len, k) = (1, 6, 5, 4);
        let input = arb(&[b, channels, s_len], 1);
        let weight = arb(&[channels, k], 2);
        let state0 = arb(&[b, channels, k], 3);

        let (out_multi, final_multi) = run(&input, &weight, &state0)?;

        let mut state = state0.clone();
        let mut outs: Vec<Tensor> = vec![];
        for t in 0..s_len {
            let step = input.slice(2, t, t + 1)?;
            let (o, st) = run(&step, &weight, &state)?;
            outs.push(o);
            state = st;
        }
        let seq_out = Tensor::stack_tensors(
            2,
            &outs.iter().map(|o| o.clone().into()).collect::<Vec<TValue>>(),
        )?;
        let seq_out = seq_out.into_shape(&[b, channels, s_len])?;
        out_multi.close_enough(&seq_out, Approximation::Close)?;
        final_multi.close_enough(&state, Approximation::Close)?;
        Ok(())
    }
}
