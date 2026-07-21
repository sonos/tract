use anyhow::ensure;
use tract_core::internal::*;
use tract_gpu::tensor::{DeviceTensor, DeviceTensorExt};
use tract_gpu::utils::facts_to_device_facts;
use tract_transformers::ops::moe_ffn::{GateMode, RouteTopK};

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct MetalRouteTopK {
    pub k: usize,
    pub gate: GateMode,
}

impl MetalRouteTopK {
    fn token_count_dim(shape: &ShapeFact) -> TDim {
        let dims = shape.to_tvec();
        dims[..dims.len() - 1].iter().cloned().product()
    }

    fn output_facts_inner(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(inputs.len() == 2 || inputs.len() == 3);
        ensure!(self.k <= 16, "MetalRouteTopK supports k <= 16, got {}", self.k);
        ensure!(inputs[0].rank() == 2 || inputs[0].rank() == 3);
        ensure!(inputs[1].rank() == 2 || inputs[1].rank() == 3);
        ensure!(inputs[0].datum_type == f32::datum_type());
        ensure!(inputs[1].datum_type == f32::datum_type());
        if inputs.len() == 3 {
            ensure!(inputs[2].rank() == 1);
            ensure!(inputs[2].datum_type == f32::datum_type());
            ensure!(inputs[2].shape[0] == inputs[1].shape[inputs[1].rank() - 2]);
        }
        let route_count = Self::token_count_dim(&inputs[0].shape) * self.k;
        Ok(tvec![
            i64::datum_type().fact(&[route_count.clone()]),
            i64::datum_type().fact(&[route_count.clone()]),
            f32::datum_type().fact(&[route_count]),
        ])
    }
}

impl Op for MetalRouteTopK {
    fn name(&self) -> StaticName {
        "MetalRouteTopK".into()
    }
    op_as_typed_op!();
}

impl EvalOp for MetalRouteTopK {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        _node_id: usize,
        _session: &TurnState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        ensure!(inputs.len() == 2 || inputs.len() == 3);
        let x_raw = &inputs[0];
        let wg_raw = &inputs[1];
        let x = x_raw
            .to_device_tensor()
            .with_context(|| format!("x is not a Metal tensor: {x_raw:?}"))?;
        let wg = wg_raw
            .to_device_tensor()
            .with_context(|| format!("wg is not a Metal tensor: {wg_raw:?}"))?;
        let wg_bias = if inputs.len() == 3 {
            let wg_bias_raw = &inputs[2];
            Some(
                wg_bias_raw
                    .to_device_tensor()
                    .with_context(|| format!("wg_bias is not a Metal tensor: {wg_bias_raw:?}"))?,
            )
        } else {
            None
        };

        let d_model = *x.shape().last().context("x has no feature axis")?;
        let token_count = x.len() / d_model;
        let route_count = token_count * self.k;
        let route_token_ids = DeviceTensor::uninitialized_dt(i64::datum_type(), &[route_count])?;
        let route_expert_ids = DeviceTensor::uninitialized_dt(i64::datum_type(), &[route_count])?;
        let route_weights = DeviceTensor::uninitialized_dt(f32::datum_type(), &[route_count])?;

        crate::with_metal_stream(|stream| {
            crate::kernels::moe::dispatch_route_topk_f32(
                stream,
                x,
                wg,
                wg_bias,
                self.k,
                &self.gate,
                &route_token_ids,
                &route_expert_ids,
                &route_weights,
            )
        })?;

        Ok(tvec![
            route_token_ids.into_tensor().into_tvalue(),
            route_expert_ids.into_tensor().into_tvalue(),
            route_weights.into_tensor().into_tvalue(),
        ])
    }
}

impl TypedOp for MetalRouteTopK {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        facts_to_device_facts(inputs, |input_facts| self.output_facts_inner(input_facts))
            .with_context(|| format!("Error while computing output facts for {}", self.name()))
    }

    as_op!();
}

crate::register_metal_op!(RouteTopK, |source, node, op| {
    let facts = source.node_input_facts(node.id)?;
    rule_if!(op.k <= 16);
    rule_if!(facts[0].datum_type == f32::datum_type());
    rule_if!(facts[1].datum_type == f32::datum_type());
    Ok(Some(Box::new(MetalRouteTopK { k: op.k, gate: op.gate.clone() })))
});

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MetalRuntime, MetalTransform};
    use tract_core::transform::ModelTransform;

    fn make_model(gate: GateMode, rank3: bool) -> TractResult<(TypedModel, Tensor)> {
        let tokens = 4;
        let d_model = 32;
        let experts = 7;
        let k = 3;
        let x_shape: TVec<usize> =
            if rank3 { tvec!(1, tokens, d_model) } else { tvec!(tokens, d_model) };
        let x_data = (0..tokens * d_model)
            .map(|i| ((i * 13 % 101) as f32 - 50.0) / 37.0)
            .collect::<Vec<_>>();
        let wg_data = (0..experts * d_model)
            .map(|i| ((i * 17 % 107) as f32 - 53.0) / 41.0)
            .collect::<Vec<_>>();

        let mut model = TypedModel::default();
        let x = model.add_source("x", f32::datum_type().fact(&x_shape))?;
        let wg = model.add_const("wg", Tensor::from_shape(&[experts, d_model], &wg_data)?)?;
        let routes = model.wire_node("route_topk", RouteTopK { k, gate }, &[x, wg])?;
        model.select_output_outlets(&routes)?;
        Ok((model, Tensor::from_shape(&x_shape, &x_data)?))
    }

    fn check_graph(gate: GateMode, rank3: bool) -> TractResult<()> {
        let (model, input) = make_model(gate, rank3)?;

        let mut transformed = model.clone();
        MetalTransform::default().transform(&mut transformed)?;
        let has_metal_route = transformed.nodes().iter().any(|node| node.op_is::<MetalRouteTopK>());
        ensure!(has_metal_route, "Metal transform did not pick MetalRouteTopK");

        let expected =
            DefaultRuntime.prepare(model.clone())?.run(tvec![input.clone().into_tvalue()])?;
        let actual = MetalRuntime.prepare(model)?.run(tvec![input.into_tvalue()])?;
        actual[0]
            .clone()
            .into_tensor()
            .close_enough(&expected[0].clone().into_tensor(), Approximation::Exact)?;
        actual[1]
            .clone()
            .into_tensor()
            .close_enough(&expected[1].clone().into_tensor(), Approximation::Exact)?;
        actual[2]
            .clone()
            .into_tensor()
            .close_enough(&expected[2].clone().into_tensor(), Approximation::Approximate)
    }

    #[test]
    fn graph_route_topk_softmax_topk_rank2() -> TractResult<()> {
        check_graph(GateMode::SoftmaxTopk, false)
    }

    #[test]
    fn graph_route_topk_softmax_all_rank3() -> TractResult<()> {
        check_graph(GateMode::SoftmaxAll, true)
    }

    #[test]
    fn graph_route_topk_sigmoid_rank2() -> TractResult<()> {
        check_graph(GateMode::Sigmoid, false)
    }

    #[test]
    fn graph_route_topk_raw_rank2() -> TractResult<()> {
        check_graph(GateMode::Raw, false)
    }
}
