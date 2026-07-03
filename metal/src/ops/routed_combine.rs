use anyhow::ensure;
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensorExt;
use tract_gpu::utils::facts_to_device_facts;
use tract_transformers::ops::moe_ffn::RoutedCombine;

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct MetalRoutedCombine;

impl MetalRoutedCombine {
    fn output_facts_inner(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(inputs.len() == 4);
        ensure!(
            inputs[0].rank() == 2 || inputs[0].rank() == 3,
            "MetalRoutedCombine shape_like must be rank 2 or 3"
        );
        ensure!(inputs[0].datum_type == f32::datum_type());
        ensure!(inputs[1].rank() == 2);
        ensure!(inputs[1].datum_type == f32::datum_type());
        ensure!(inputs[2].rank() == 1);
        ensure!(inputs[2].datum_type == i64::datum_type());
        ensure!(inputs[3].rank() == 1);
        ensure!(inputs[3].datum_type == f32::datum_type());
        Ok(tvec![f32::datum_type().fact(inputs[0].shape.clone())])
    }
}

impl Op for MetalRoutedCombine {
    fn name(&self) -> StaticName {
        "MetalRoutedCombine".into()
    }
    op_as_typed_op!();
}

impl EvalOp for MetalRoutedCombine {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        node_id: usize,
        session: &TurnState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let (shape_like_raw, route_values_raw, route_token_ids_raw, route_weights_raw) =
            args_4!(inputs);
        let shape_like = shape_like_raw
            .to_device_tensor()
            .with_context(|| format!("shape_like is not a Metal tensor: {shape_like_raw:?}"))?;
        let route_values = route_values_raw.to_device_tensor().with_context(|| {
            format!("route_values are not a Metal tensor: {route_values_raw:?}")
        })?;
        let route_token_ids = route_token_ids_raw.to_device_tensor().with_context(|| {
            format!("route_token_ids are not a Metal tensor: {route_token_ids_raw:?}")
        })?;
        let route_weights = route_weights_raw.to_device_tensor().with_context(|| {
            format!("route_weights are not a Metal tensor: {route_weights_raw:?}")
        })?;

        let output = tract_gpu::session_handler::make_tensor_for_node(
            session,
            node_id,
            f32::datum_type(),
            shape_like.shape(),
        )?;

        crate::with_metal_stream(|stream| {
            crate::kernels::moe::dispatch_routed_combine_f32(
                stream,
                route_values,
                route_token_ids,
                route_weights,
                &output,
            )
        })?;

        Ok(tvec![output.into_tensor().into_tvalue()])
    }
}

impl TypedOp for MetalRoutedCombine {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        facts_to_device_facts(inputs, |input_facts| self.output_facts_inner(input_facts))
            .with_context(|| format!("Error while computing output facts for {}", self.name()))
    }

    as_op!();
}

crate::register_metal_op!(RoutedCombine, |source, node, _op| {
    let facts = source.node_input_facts(node.id)?;
    rule_if!(facts[0].datum_type == f32::datum_type());
    rule_if!(facts[1].datum_type == f32::datum_type());
    rule_if!(facts[2].datum_type == i64::datum_type());
    rule_if!(facts[3].datum_type == f32::datum_type());
    Ok(Some(Box::new(MetalRoutedCombine)))
});

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MetalRuntime, MetalTransform};
    use tract_core::transform::ModelTransform;

    fn make_model(rank3: bool) -> TractResult<(TypedModel, Tensor)> {
        let d_model = 5;
        let shape_like_shape: TVec<usize> =
            if rank3 { tvec!(1, 3, d_model) } else { tvec!(3, d_model) };
        let route_values = Tensor::from_shape(
            &[6, d_model],
            &(0..6 * d_model).map(|i| ((i * 7 % 29) as f32 - 14.0) / 11.0).collect::<Vec<_>>(),
        )?;
        let route_token_ids = Tensor::from_shape(&[6], &[2i64, 0, 1, 2, 1, 0])?;
        let route_weights = Tensor::from_shape(&[6], &[0.2f32, 1.0, -0.5, 0.75, 0.25, -0.3])?;

        let mut model = TypedModel::default();
        let shape_like =
            model.add_source("shape_like", f32::datum_type().fact(&shape_like_shape))?;
        let route_values = model.add_const("route_values", route_values)?;
        let route_token_ids = model.add_const("route_token_ids", route_token_ids)?;
        let route_weights = model.add_const("route_weights", route_weights)?;
        let y = model.wire_node(
            "combine",
            RoutedCombine,
            &[shape_like, route_values, route_token_ids, route_weights],
        )?;
        model.select_output_outlets(&y)?;
        Ok((model, Tensor::zero_dt(f32::datum_type(), &shape_like_shape)?))
    }

    fn check_graph(rank3: bool) -> TractResult<()> {
        let (model, shape_like) = make_model(rank3)?;

        let mut transformed = model.clone();
        MetalTransform::default().transform(&mut transformed)?;
        let has_metal_combine =
            transformed.nodes().iter().any(|node| node.op_is::<MetalRoutedCombine>());
        ensure!(has_metal_combine, "Metal transform did not pick MetalRoutedCombine");

        let expected =
            DefaultRuntime.prepare(model.clone())?.run(tvec![shape_like.clone().into_tvalue()])?;
        let actual = MetalRuntime.prepare(model)?.run(tvec![shape_like.into_tvalue()])?;
        actual[0]
            .clone()
            .into_tensor()
            .close_enough(&expected[0].clone().into_tensor(), Approximation::Approximate)
    }

    #[test]
    fn graph_routed_combine_rank2() -> TractResult<()> {
        check_graph(false)
    }

    #[test]
    fn graph_routed_combine_rank3() -> TractResult<()> {
        check_graph(true)
    }
}
