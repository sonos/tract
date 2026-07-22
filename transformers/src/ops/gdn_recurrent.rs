use tract_nnef::internal::*;

pub fn register(registry: &mut Registry) {
    registry.register_primitive(
        "tract_transformers_gdn_recurrent",
        &[
            TypeName::Scalar.tensor().named("query"),
            TypeName::Scalar.tensor().named("key"),
            TypeName::Scalar.tensor().named("value"),
            TypeName::Scalar.tensor().named("log_decay"),
            TypeName::Scalar.tensor().named("beta"),
            TypeName::Scalar.tensor().named("initial_state"),
        ],
        &[("output", TypeName::Scalar.tensor()), ("final_state", TypeName::Scalar.tensor())],
        |builder, invocation| {
            let inputs = ["query", "key", "value", "log_decay", "beta", "initial_state"]
                .map(|name| invocation.named_arg_as(builder, name))
                .into_iter()
                .collect::<TractResult<TVec<_>>>()?;
            builder.wire(GatedDeltaNetRecurrent, &inputs)
        },
    );
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GatedDeltaNetRecurrent;

impl Op for GatedDeltaNetRecurrent {
    fn name(&self) -> StaticName {
        "GatedDeltaNetRecurrent".into()
    }
    op_as_typed_op!();
}

impl EvalOp for GatedDeltaNetRecurrent {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        ensure!(inputs.len() == 6, "GDN expects q, k, v, log_decay, beta, state");
        let q = inputs[0].to_plain_array_view::<f16>()?;
        let k = inputs[1].to_plain_array_view::<f16>()?;
        let v = inputs[2].to_plain_array_view::<f16>()?;
        let g = inputs[3].to_plain_array_view::<f32>()?;
        let beta = inputs[4].to_plain_array_view::<f16>()?;
        let state = inputs[5].to_plain_array_view::<f32>()?;
        let width = *q.shape().last().context("GDN query must have a last axis")?;
        ensure!(width == 128, "the Qwen3.5 recurrent op requires width=128");
        ensure!(q.shape() == k.shape() && q.shape() == v.shape());
        let heads = q.len() / width;
        ensure!(g.len() == heads && beta.len() == heads);
        ensure!(state.len() == heads * width * width);

        let q = q.as_slice().context("query must be contiguous")?;
        let k = k.as_slice().context("key must be contiguous")?;
        let v = v.as_slice().context("value must be contiguous")?;
        let g = g.as_slice().context("log_decay must be contiguous")?;
        let beta = beta.as_slice().context("beta must be contiguous")?;
        let state = state.as_slice().context("state must be contiguous")?;
        let mut output = vec![f16::ZERO; q.len()];
        let mut final_state = vec![0f32; state.len()];
        for head in 0..heads {
            let vb = head * width;
            let mb = head * width * width;
            let q_inv = 1.0
                / (q[vb..vb + width].iter().map(|x| x.to_f32().powi(2)).sum::<f32>() + 1e-6).sqrt();
            let k_inv = 1.0
                / (k[vb..vb + width].iter().map(|x| x.to_f32().powi(2)).sum::<f32>() + 1e-6).sqrt();
            let decay = g[head].exp();
            for col in 0..width {
                let predicted = (0..width)
                    .map(|row| k[vb + row].to_f32() * k_inv * state[mb + row * width + col] * decay)
                    .sum::<f32>();
                let residual = (v[vb + col].to_f32() - predicted) * beta[head].to_f32();
                let mut result = 0f32;
                for row in 0..width {
                    let ix = mb + row * width + col;
                    let next = state[ix] * decay + k[vb + row].to_f32() * k_inv * residual;
                    final_state[ix] = next;
                    result += q[vb + row].to_f32() * q_inv * next;
                }
                output[vb + col] = f16::from_f32(result / (width as f32).sqrt());
            }
        }
        Ok(tvec![
            Tensor::from_shape(inputs[0].shape(), &output)?.into_tvalue(),
            Tensor::from_shape(inputs[5].shape(), &final_state)?.into_tvalue(),
        ])
    }
}

impl TypedOp for GatedDeltaNetRecurrent {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(inputs.len() == 6);
        ensure!(inputs[0].datum_type == DatumType::F16);
        ensure!(inputs[1].datum_type == DatumType::F16);
        ensure!(inputs[2].datum_type == DatumType::F16);
        ensure!(inputs[3].datum_type == DatumType::F32);
        ensure!(inputs[4].datum_type == DatumType::F16);
        ensure!(inputs[5].datum_type == DatumType::F32);
        ensure!(inputs[0].shape == inputs[1].shape && inputs[0].shape == inputs[2].shape);
        Ok(tvec![inputs[0].without_value(), inputs[5].without_value()])
    }
    as_op!();
}
