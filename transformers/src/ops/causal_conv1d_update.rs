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

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CausalConv1dUpdate;

impl Op for CausalConv1dUpdate {
    fn name(&self) -> StaticName {
        "CausalConv1dUpdate".into()
    }
    op_as_typed_op!();
}

impl EvalOp for CausalConv1dUpdate {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let (input, weight, state) = args_3!(inputs);
        let input = input.to_plain_array_view::<f16>()?;
        let weight = weight.to_plain_array_view::<f16>()?;
        let state = state.to_plain_array_view::<f16>()?;
        let kernel_width = *weight.shape().last().context("conv weight must have a kernel axis")?;
        ensure!(kernel_width == 4, "Qwen3.5 requires a four-tap convolution");
        let channels = input.len();
        ensure!(weight.len() == channels * kernel_width);
        ensure!(state.len() == channels * kernel_width);
        let input_shape = input.shape().to_vec();
        let state_shape = state.shape().to_vec();
        let input = input.as_slice().context("input must be contiguous")?;
        let weight = weight.as_slice().context("weight must be contiguous")?;
        let state = state.as_slice().context("state must be contiguous")?;
        let mut output = vec![f16::ZERO; channels];
        let mut final_state = vec![f16::ZERO; state.len()];
        for channel in 0..channels {
            let base = channel * kernel_width;
            let mut sum = 0f32;
            for tap in 0..kernel_width - 1 {
                final_state[base + tap] = state[base + tap + 1];
                sum += state[base + tap + 1].to_f32() * weight[base + tap].to_f32();
            }
            final_state[base + kernel_width - 1] = input[channel];
            sum += input[channel].to_f32() * weight[base + kernel_width - 1].to_f32();
            output[channel] = f16::from_f32(sum / (1.0 + (-sum).exp()));
        }
        Ok(tvec![
            Tensor::from_shape(&input_shape, &output)?.into_tvalue(),
            Tensor::from_shape(&state_shape, &final_state)?.into_tvalue(),
        ])
    }
}

impl TypedOp for CausalConv1dUpdate {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(inputs.len() == 3);
        ensure!(inputs.iter().all(|fact| fact.datum_type == DatumType::F16));
        Ok(tvec![inputs[0].without_value(), inputs[2].without_value()])
    }
    as_op!();
}
