/// `FoldUniformTDim` optimizer pass.
///
/// Analogous to `PropConst` for `uniform`: when a wire's `uniform_tdim` is
/// known, this pass replaces its entire producer subgraph with a single
/// `UniformTDim` node that evaluates the expression at runtime using the
/// session's resolved symbol values.
///
/// This collapses chains like:
///   Range → Cast → Div → Floor → Cast → Unsqueeze → Sub → Le/Ge → And
/// into a single `UniformTDim(expr, shape)` node, leaving the Iff intact
/// and the model runnable.
use crate::internal::*;
use crate::ops::uniform_tdim::UniformTDim;
use crate::optim::OptimizerSession;

#[derive(Clone, Debug, Default)]
pub struct FoldUniformTDim(usize);

impl super::TypedPass for FoldUniformTDim {
    fn reset(&mut self) -> TractResult<()> {
        self.0 = 0;
        Ok(())
    }

    fn next(
        &mut self,
        _session: &mut OptimizerSession,
        model: &TypedModel,
    ) -> TractResult<Option<TypedModelPatch>> {
        let order = model.eval_order()?;
        for (order_ix, &node_id) in order.iter().enumerate().skip(self.0) {
            self.0 = order_ix + 1;
            let node = &model.nodes()[node_id];

            // Skip nodes that are already UniformTDim or Const — no-op.
            if node.op_as::<UniformTDim>().is_some() {
                continue;
            }
            if node.op_as::<crate::ops::konst::Const>().is_some() {
                continue;
            }

            for slot in 0..node.outputs.len() {
                let outlet = OutletId::new(node_id, slot);
                let fact = model.outlet_fact(outlet)?;

                let expr = match fact.uniform_tdim.as_ref() {
                    Some(e) => e.clone(),
                    None => continue,
                };

                // Don't bother if the shape is 0-d (scalar) — nothing to fold.
                if fact.shape.rank() == 0 {
                    continue;
                }

                // Only fold bool wires.  Non-bool wires may carry uniform_tdim as
                // metadata for other passes, but UniformTDim can only materialise
                // bool tensors (TDim is integer-valued).
                if fact.datum_type != DatumType::Bool {
                    continue;
                }

                let shape = fact.shape.clone();
                let dt = fact.datum_type;

                let mut patch = TypedModelPatch::default();
                patch.push_context(format!("FoldUniformTDim/{node_id}/{slot}"));

                // Wire a model input as a dummy dependency so that UniformTDim
                // is topologically ordered after Source nodes, ensuring that
                // symbols like S are resolved in session.resolved_symbols before
                // UniformTDim::eval_with_session tries to evaluate the shape.
                let shape_has_symbols = shape.iter().any(|d| !d.symbols().is_empty());
                let dummy_inputs: TVec<OutletId> = if shape_has_symbols && !model.inputs.is_empty()
                {
                    tvec![patch.tap_model(model, model.inputs[0])?]
                } else {
                    tvec![]
                };

                let uniform_node = patch.wire_node(
                    &node.name,
                    UniformTDim::new(expr, shape, dt),
                    &dummy_inputs,
                )?[0];

                patch.shunt_outside(model, outlet, uniform_node)?;
                return Ok(Some(patch));
            }
        }
        Ok(None)
    }
}
