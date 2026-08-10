use crate::kernels::matmul::{
    RoutedQ40InputMode, RoutedSwigluAct, dispatch_routed_q40_swiglu_f32,
};
use anyhow::ensure;
use tract_core::internal::*;
use tract_core::tract_linalg::block_quant::Q4_0;
use tract_gpu::tensor::DeviceTensorExt;
use tract_gpu::utils::{as_quant_fact, facts_to_device_facts};
use tract_transformers::ops::moe_ffn::RoutedInputMode;

/// Fused routed expert up-projection: g = w1 x (+bias1), u = w3 x (+bias3),
/// output = act(g, u), in one kernel dispatch at decode and with a shared
/// expert sort + activation gather at prefill. Replaces the unfused
/// w1-matmul / w3-matmul / (bias adds) / activation dispatch chain.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct MetalRoutedQ40SwiGlu {
    pub input_mode: RoutedInputMode,
    /// Clamped-swiglu parameters (gpt-oss style) as bit patterns; None is the
    /// plain silu(g)*u epilogue.
    pub act_alpha_bits: Option<u32>,
    pub act_limit_bits: Option<u32>,
    pub has_bias: bool,
    /// Same command-buffer-boundary medicine as MetalRoutedQ40MatMul (see
    /// that op for the full story); one boundary per MoE block instead of
    /// the three the unfused lowering paid.
    pub sync_after_dispatch: bool,
}

impl MetalRoutedQ40SwiGlu {
    fn kernel_input_mode(&self) -> RoutedQ40InputMode {
        match self.input_mode {
            RoutedInputMode::TokenRows => RoutedQ40InputMode::TokenRows,
            RoutedInputMode::RouteRows => RoutedQ40InputMode::RouteRows,
        }
    }

    fn act(&self) -> RoutedSwigluAct {
        match (self.act_alpha_bits, self.act_limit_bits) {
            (alpha, Some(limit)) => RoutedSwigluAct::Clamped {
                alpha: alpha.map(f32::from_bits).unwrap_or(1.0),
                limit: f32::from_bits(limit),
            },
            _ => RoutedSwigluAct::Plain,
        }
    }

    fn input_count(&self) -> usize {
        5 + 2 * self.has_bias as usize
    }

    fn output_facts_inner(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(inputs.len() == self.input_count());
        ensure!(inputs[0].rank() == 2, "MetalRoutedQ40SwiGlu input must be rank 2");
        for w in [inputs[1], inputs[2]] {
            ensure!(w.rank() == 3, "MetalRoutedQ40SwiGlu weights must be rank 3 [E,N,K]");
            ensure!(
                as_quant_fact(w, &Q4_0).is_some(),
                "MetalRoutedQ40SwiGlu weights must be Q4_0"
            );
        }
        ensure!(inputs[3].rank() == 1 && inputs[4].rank() == 1);
        ensure!(inputs[3].datum_type == i64::datum_type());
        ensure!(inputs[4].datum_type == i64::datum_type());
        if self.has_bias {
            for b in [inputs[5], inputs[6]] {
                ensure!(b.datum_type == f32::datum_type() && b.rank() == 2);
            }
        }
        let route_count = inputs[3].shape.to_tvec()[0].clone();
        let out_dim = inputs[1].shape.to_tvec()[1].clone();
        Ok(tvec!(f32::datum_type().fact(&[route_count, out_dim])))
    }
}

impl Op for MetalRoutedQ40SwiGlu {
    fn name(&self) -> StaticName {
        "MetalRoutedQ40SwiGlu".into()
    }
    op_as_typed_op!();
}

impl EvalOp for MetalRoutedQ40SwiGlu {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        node_id: usize,
        session: &TurnState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        ensure!(inputs.len() == self.input_count());
        let tensors = inputs
            .iter()
            .map(|value| value.to_device_tensor())
            .collect::<TractResult<TVec<_>>>()?;
        let (input, w1, w3, route_token_ids, route_expert_ids) =
            (tensors[0], tensors[1], tensors[2], tensors[3], tensors[4]);
        let biases = self.has_bias.then(|| (tensors[5], tensors[6]));

        ensure!(route_token_ids.rank() == 1);
        ensure!(w1.rank() == 3);
        let output = tract_gpu::session_handler::make_tensor_for_node(
            session,
            node_id,
            f32::datum_type(),
            &[route_token_ids.shape()[0], w1.shape()[1]],
        )?;

        crate::with_metal_stream(|stream| {
            dispatch_routed_q40_swiglu_f32(
                stream,
                input,
                w1,
                w3,
                biases,
                route_token_ids,
                route_expert_ids,
                self.kernel_input_mode(),
                self.act(),
                &output,
            )?;
            // Same prefill-only command-buffer boundary as
            // MetalRoutedQ40MatMul (see that op).
            let min_routes = std::env::var("TRACT_METAL_MOE_COMMIT_MIN_ROUTES")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(64);
            if self.sync_after_dispatch && route_token_ids.shape()[0] > min_routes {
                stream.commit_current()?;
            }
            Ok(())
        })?;

        Ok(tvec![output.into_tensor().into_tvalue()])
    }
}

impl TypedOp for MetalRoutedQ40SwiGlu {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        facts_to_device_facts(inputs, |input_facts| self.output_facts_inner(input_facts))
            .with_context(|| format!("Error while computing output facts for {}", self.name()))
    }

    as_op!();
}
