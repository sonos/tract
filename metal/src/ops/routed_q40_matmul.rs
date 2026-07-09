use crate::kernels::matmul::{RoutedQ40InputMode, dispatch_routed_q40_f32};
use anyhow::ensure;
use tract_core::internal::*;
use tract_core::tract_linalg::block_quant::Q4_0;
use tract_gpu::tensor::DeviceTensorExt;
use tract_gpu::utils::{as_quant_fact, facts_to_device_facts};
use tract_transformers::ops::moe_ffn::{RoutedInputMode, RoutedQ40MatMul};

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct MetalRoutedQ40MatMul {
    pub input_mode: RoutedInputMode,
}

impl MetalRoutedQ40MatMul {
    fn kernel_input_mode(&self) -> RoutedQ40InputMode {
        match self.input_mode {
            RoutedInputMode::TokenRows => RoutedQ40InputMode::TokenRows,
            RoutedInputMode::RouteRows => RoutedQ40InputMode::RouteRows,
        }
    }

    fn output_facts_inner(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(inputs.len() == 4);
        ensure!(inputs[0].rank() == 2, "MetalRoutedQ40MatMul input must be rank 2");
        ensure!(inputs[1].rank() == 3, "MetalRoutedQ40MatMul weights must be rank 3 [E,N,K]");
        ensure!(
            as_quant_fact(inputs[1], &Q4_0).is_some(),
            "MetalRoutedQ40MatMul weights must be Q4_0"
        );
        ensure!(inputs[2].rank() == 1 && inputs[3].rank() == 1);
        ensure!(inputs[2].datum_type == i64::datum_type());
        ensure!(inputs[3].datum_type == i64::datum_type());
        let route_count = inputs[2].shape.to_tvec()[0].clone();
        let out_dim = inputs[1].shape.to_tvec()[1].clone();
        Ok(tvec!(f32::datum_type().fact(&[route_count, out_dim])))
    }
}

impl Op for MetalRoutedQ40MatMul {
    fn name(&self) -> StaticName {
        "MetalRoutedQ40MatMul".into()
    }
    op_as_typed_op!();
}

impl EvalOp for MetalRoutedQ40MatMul {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        node_id: usize,
        session: &TurnState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let (input_raw, weights_raw, route_token_ids_raw, route_expert_ids_raw) = args_4!(inputs);
        let input = input_raw
            .to_device_tensor()
            .with_context(|| format!("input is not a Metal tensor: {input_raw:?}"))?;
        let weights = weights_raw
            .to_device_tensor()
            .with_context(|| format!("weights are not a Metal tensor: {weights_raw:?}"))?;
        let route_token_ids = route_token_ids_raw.to_device_tensor().with_context(|| {
            format!("route_token_ids are not a Metal tensor: {route_token_ids_raw:?}")
        })?;
        let route_expert_ids = route_expert_ids_raw.to_device_tensor().with_context(|| {
            format!("route_expert_ids are not a Metal tensor: {route_expert_ids_raw:?}")
        })?;

        ensure!(route_token_ids.rank() == 1);
        ensure!(weights.rank() == 3);
        let output = tract_gpu::session_handler::make_tensor_for_node(
            session,
            node_id,
            f32::datum_type(),
            &[route_token_ids.shape()[0], weights.shape()[1]],
        )?;

        crate::with_metal_stream(|stream| {
            dispatch_routed_q40_f32(
                stream,
                input,
                weights,
                route_token_ids,
                route_expert_ids,
                self.kernel_input_mode(),
                &output,
            )
        })?;

        Ok(tvec![output.into_tensor().into_tvalue()])
    }
}

impl TypedOp for MetalRoutedQ40MatMul {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        facts_to_device_facts(inputs, |input_facts| self.output_facts_inner(input_facts))
            .with_context(|| format!("Error while computing output facts for {}", self.name()))
    }

    as_op!();
}

crate::register_metal_op!(RoutedQ40MatMul, |source, node, op| {
    let facts = source.node_input_facts(node.id)?;
    rule_if!(facts[0].datum_type == f32::datum_type());
    rule_if!(facts[2].datum_type == i64::datum_type());
    rule_if!(facts[3].datum_type == i64::datum_type());
    rule_if!(as_quant_fact(&facts[1], &Q4_0).is_some());
    Ok(Some(Box::new(MetalRoutedQ40MatMul { input_mode: op.input_mode })))
});

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{MetalRuntime, MetalTransform};
    use tract_core::ops::konst::Const;
    use tract_core::tract_linalg::block_quant::{BlockQuant, BlockQuantFact, BlockQuantStorage};
    use tract_core::transform::ModelTransform;
    use tract_transformers::ops::moe_ffn::{ExpertLayout, GateMode, MoeFfn};

    fn add_q40_const(
        model: &mut TypedModel,
        name: &str,
        shape: &[usize],
        data: &[f32],
    ) -> TractResult<OutletId> {
        let k = *shape.last().context("Q40 tensor has no last axis")?;
        ensure!(k % Q4_0.block_len() == 0);
        let rows = shape[..shape.len() - 1].iter().product::<usize>();
        let quant = Q4_0.quant_f32(data)?;
        let storage = BlockQuantStorage::new(Box::new(Q4_0), rows, k, Arc::new(quant))?;
        let tensor = Arc::new(storage.into_tensor_with_shape(f32::datum_type(), shape));
        let fact = BlockQuantFact::new(Box::new(Q4_0), shape.iter().copied().collect());
        Ok(model.wire_node(name, Const::new_with_exotic_fact(tensor, Box::new(fact))?, &[])?[0])
    }

    fn tensor_from_f32(shape: &[usize], data: &[f32], dt: DatumType) -> TractResult<Tensor> {
        match dt {
            DatumType::F32 => Tensor::from_shape(shape, data),
            DatumType::F16 => {
                let data = data.iter().map(|&v| f16::from_f32(v)).collect::<Vec<_>>();
                Tensor::from_shape(shape, &data)
            }
            _ => bail!("unsupported test datum type: {dt:?}"),
        }
    }

    fn make_model(input_mode: RoutedInputMode) -> TractResult<(TypedModel, Tensor)> {
        let experts = 3;
        let tokens = 5;
        let routes = 6;
        let n = 17;
        let k = 64;
        let input_rows = match input_mode {
            RoutedInputMode::TokenRows => tokens,
            RoutedInputMode::RouteRows => routes,
        };
        let input_data =
            (0..input_rows * k).map(|i| ((i * 13 % 97) as f32 - 48.0) / 64.0).collect::<Vec<_>>();
        let weight_data =
            (0..experts * n * k).map(|i| ((i * 17 % 101) as f32 - 50.0) / 80.0).collect::<Vec<_>>();
        let route_token_ids = match input_mode {
            RoutedInputMode::TokenRows => vec![3, 0, 4, 1, 3, 2],
            RoutedInputMode::RouteRows => (0..routes as i64).collect(),
        };
        let route_expert_ids = vec![1i64, 0, 2, 1, 2, 0];

        let mut model = TypedModel::default();
        let x = model.add_source("x", f32::datum_type().fact([input_rows, k]))?;
        let weights = add_q40_const(&mut model, "weights", &[experts, n, k], &weight_data)?;
        let route_token_ids =
            model.add_const("route_token_ids", Tensor::from_shape(&[routes], &route_token_ids)?)?;
        let route_expert_ids = model
            .add_const("route_expert_ids", Tensor::from_shape(&[routes], &route_expert_ids)?)?;
        let y = model.wire_node(
            "routed_q40",
            RoutedQ40MatMul { input_mode },
            &[x, weights, route_token_ids, route_expert_ids],
        )?;
        model.select_output_outlets(&y)?;

        Ok((model, Tensor::from_shape(&[input_rows, k], &input_data)?))
    }

    fn check_graph(input_mode: RoutedInputMode) -> TractResult<()> {
        let (model, input) = make_model(input_mode)?;

        let mut transformed = model.clone();
        MetalTransform::default().transform(&mut transformed)?;
        let has_metal_routed =
            transformed.nodes().iter().any(|node| node.op_is::<MetalRoutedQ40MatMul>());
        ensure!(has_metal_routed, "Metal transform did not pick MetalRoutedQ40MatMul");

        let expected =
            DefaultRuntime.prepare(model.clone())?.run(tvec![input.clone().into_tvalue()])?;
        let actual = MetalRuntime.prepare(model)?.run(tvec![input.into_tvalue()])?;
        actual[0]
            .clone()
            .into_tensor()
            .close_enough(&expected[0].clone().into_tensor(), Approximation::Approximate)
    }

    #[test]
    fn graph_routed_q40_token_rows() -> TractResult<()> {
        check_graph(RoutedInputMode::TokenRows)
    }

    #[test]
    fn graph_routed_q40_route_rows() -> TractResult<()> {
        check_graph(RoutedInputMode::RouteRows)
    }

    fn check_q40_moe_ffn_lowers_to_metal_primitives(
        rank3_input: bool,
        input_dt: DatumType,
        router_dt: DatumType,
    ) -> TractResult<()> {
        let experts = 4;
        let tokens = 3;
        let d_model = 32;
        let d_hidden = 64;
        let input_shape =
            if rank3_input { vec![1, tokens, d_model] } else { vec![tokens, d_model] };
        let mut rng_state: u64 = 20260702;
        let mut next_f32 = || -> f32 {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng_state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        };
        let input_data = (0..tokens * d_model).map(|_| next_f32()).collect::<Vec<_>>();
        let wg_data = (0..experts * d_model).map(|_| next_f32()).collect::<Vec<_>>();
        let w1_data = (0..experts * d_hidden * d_model).map(|_| next_f32()).collect::<Vec<_>>();
        let w2_data = (0..experts * d_model * d_hidden).map(|_| next_f32()).collect::<Vec<_>>();
        let w3_data = (0..experts * d_hidden * d_model).map(|_| next_f32()).collect::<Vec<_>>();

        let mut model = TypedModel::default();
        let x = model.add_source("x", input_dt.fact(&input_shape))?;
        let wg =
            model.add_const("wg", tensor_from_f32(&[experts, d_model], &wg_data, router_dt)?)?;
        let w1 = add_q40_const(&mut model, "w1", &[experts, d_hidden, d_model], &w1_data)?;
        let w2 = add_q40_const(&mut model, "w2", &[experts, d_model, d_hidden], &w2_data)?;
        let w3 = add_q40_const(&mut model, "w3", &[experts, d_hidden, d_model], &w3_data)?;
        let op = MoeFfn {
            k: 2,
            activation: "silu".to_string(),
            gate: GateMode::SoftmaxTopk,
            has_w3: true,
            has_wg_bias: false,
            has_w1_bias: false,
            has_w3_bias: false,
            has_w2_bias: false,
            act_alpha_bits: None,
            act_limit_bits: None,
            expert_layout: ExpertLayout::Linear,
        };
        let y = model.wire_node("moe", op, &[x, wg, w1, w2, w3])?;
        model.select_output_outlets(&y)?;

        let mut transformed = model.clone();
        MetalTransform::default().transform(&mut transformed)?;
        let routed_count = transformed
            .nodes()
            .iter()
            .filter(|node| node.op().name() == "MetalRoutedQ40MatMul")
            .count();
        let transformed_ops = || {
            transformed
                .nodes()
                .iter()
                .map(|node| format!("{}: {}", node.name, node.op().name()))
                .collect::<Vec<_>>()
                .join("\n")
        };
        ensure!(
            routed_count == 3,
            "expected 3 MetalRoutedQ40MatMul nodes, got {routed_count}\n{}",
            transformed_ops()
        );
        let has_combine =
            transformed.nodes().iter().any(|node| node.op_is::<crate::ops::MetalRoutedCombine>());
        ensure!(has_combine, "expected MetalRoutedCombine in lowered MoE graph");

        let input = tensor_from_f32(&input_shape, &input_data, input_dt)?;
        let expected =
            DefaultRuntime.prepare(model.clone())?.run(tvec![input.clone().into_tvalue()])?;
        let actual = MetalRuntime.prepare(model)?.run(tvec![input.into_tvalue()])?;
        actual[0]
            .clone()
            .into_tensor()
            .close_enough(&expected[0].clone().into_tensor(), Approximation::Approximate)
    }

    #[test]
    fn graph_q40_moe_ffn_lowers_to_metal_primitives() -> TractResult<()> {
        check_q40_moe_ffn_lowers_to_metal_primitives(false, DatumType::F32, DatumType::F32)
    }

    #[test]
    fn graph_q40_moe_ffn_lowers_rank3_f16_to_metal_primitives() -> TractResult<()> {
        check_q40_moe_ffn_lowers_to_metal_primitives(true, DatumType::F16, DatumType::F16)
    }

    #[test]
    fn graph_gpt_oss_q40_moe_ffn_lowers_to_metal_primitives() -> TractResult<()> {
        let experts = 4;
        let tokens = 3;
        let d_model = 32;
        let d_hidden = 64;
        let input_shape = vec![tokens, d_model];
        let mut rng_state: u64 = 20260709;
        let mut next_f32 = || -> f32 {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng_state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        };
        let input_data = (0..tokens * d_model).map(|_| next_f32()).collect::<Vec<_>>();
        let wg_data = (0..experts * d_model).map(|_| next_f32()).collect::<Vec<_>>();
        let w1_data = (0..experts * d_hidden * d_model).map(|_| next_f32()).collect::<Vec<_>>();
        let w2_data = (0..experts * d_model * d_hidden).map(|_| next_f32()).collect::<Vec<_>>();
        let w3_data = (0..experts * d_hidden * d_model).map(|_| next_f32()).collect::<Vec<_>>();
        let wg_bias_data = (0..experts).map(|_| next_f32()).collect::<Vec<_>>();
        let w1_bias_data = (0..experts * d_hidden).map(|_| next_f32()).collect::<Vec<_>>();
        let w3_bias_data = (0..experts * d_hidden).map(|_| next_f32()).collect::<Vec<_>>();
        let w2_bias_data = (0..experts * d_model).map(|_| next_f32()).collect::<Vec<_>>();

        let mut model = TypedModel::default();
        let x = model.add_source("x", f16::datum_type().fact(&input_shape))?;
        let wg = model
            .add_const("wg", tensor_from_f32(&[experts, d_model], &wg_data, DatumType::F16)?)?;
        let w1 = add_q40_const(&mut model, "w1", &[experts, d_hidden, d_model], &w1_data)?;
        let w2 = add_q40_const(&mut model, "w2", &[experts, d_model, d_hidden], &w2_data)?;
        let w3 = add_q40_const(&mut model, "w3", &[experts, d_hidden, d_model], &w3_data)?;
        let wg_bias = model
            .add_const("wg_bias", tensor_from_f32(&[experts], &wg_bias_data, DatumType::F16)?)?;
        let w1_bias = model.add_const(
            "w1_bias",
            tensor_from_f32(&[experts, d_hidden], &w1_bias_data, DatumType::F16)?,
        )?;
        let w3_bias = model.add_const(
            "w3_bias",
            tensor_from_f32(&[experts, d_hidden], &w3_bias_data, DatumType::F16)?,
        )?;
        let w2_bias = model.add_const(
            "w2_bias",
            tensor_from_f32(&[experts, d_model], &w2_bias_data, DatumType::F16)?,
        )?;
        let op = MoeFfn {
            k: 2,
            activation: "swiglu".to_string(),
            gate: GateMode::SoftmaxTopk,
            has_w3: true,
            has_wg_bias: true,
            has_w1_bias: true,
            has_w3_bias: true,
            has_w2_bias: true,
            act_alpha_bits: Some(1.702f32.to_bits()),
            act_limit_bits: Some(7.0f32.to_bits()),
            expert_layout: ExpertLayout::Linear,
        };
        let y =
            model.wire_node("moe", op, &[x, wg, w1, w2, w3, wg_bias, w1_bias, w3_bias, w2_bias])?;
        model.select_output_outlets(&y)?;

        let mut transformed = model.clone();
        MetalTransform::default().transform(&mut transformed)?;
        let routed_count = transformed
            .nodes()
            .iter()
            .filter(|node| node.op().name() == "MetalRoutedQ40MatMul")
            .count();
        ensure!(routed_count == 3, "expected 3 MetalRoutedQ40MatMul nodes, got {routed_count}");
        ensure!(
            transformed.nodes().iter().any(|node| node.op_is::<crate::ops::MetalClampedSwiGlu>()),
            "expected MetalClampedSwiGlu in lowered MoE graph"
        );
        ensure!(
            transformed.nodes().iter().any(|node| node.op_is::<crate::ops::MetalRoutedCombine>()),
            "expected MetalRoutedCombine in lowered MoE graph"
        );

        let input = tensor_from_f32(&input_shape, &input_data, DatumType::F16)?;
        let expected =
            DefaultRuntime.prepare(model.clone())?.run(tvec![input.clone().into_tvalue()])?;
        let actual = MetalRuntime.prepare(model)?.run(tvec![input.into_tvalue()])?;
        actual[0]
            .clone()
            .into_tensor()
            .close_enough(&expected[0].clone().into_tensor(), Approximation::Approximate)
    }
}
