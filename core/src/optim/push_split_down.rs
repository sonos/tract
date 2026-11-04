use crate::internal::*;

use crate::optim::OptimizerSession;
use tract_itertools::Itertools;

#[derive(Clone, Debug)]
pub struct PushSplitDown;

impl super::TypedPass for PushSplitDown {
    fn reset(&mut self) -> TractResult<()> {
        Ok(())
    }
    fn next(
        &mut self,
        _session: &mut OptimizerSession,
        model: &TypedModel,
    ) -> TractResult<Option<TypedModelPatch>> {
        let mut patch = TypedModelPatch::default();
        for node in model.eval_order()? {
            for output in &model.node(node).outputs {
                for [a, b] in output.successors.iter().array_combinations() {
                    if a.node == b.node {
                        // found where a square is implemented using a mul with duplicate input
                        continue;
                    }
                    if patch.obliterate.contains(&a.node) || patch.obliterate.contains(&b.node) {
                        continue;
                    }
                    // dont merge outputs.
                    if model.outputs.contains(&a.node.into())
                        && model.outputs.contains(&b.node.into())
                    {
                        continue;
                    }
                    let a = model.node(a.node);
                    let b = model.node(b.node);
                    if a.same_as(b) {
                        for slot in 0..b.outputs.len() {
                            let tap = patch.tap_model(model, OutletId::new(a.id, slot))?;
                            patch.shunt_outside(model, OutletId::new(b.id, slot), tap)?;
                            patch.obliterate(b.id)?;
                        }
                    }
                }
            }
        }
        Ok(Some(patch).filter(|p| !p.is_empty()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::math::add;
    use crate::ops::math::mul;

    #[test]
    fn twins_seen_in_both_orders_leave_one_survivor() -> TractResult<()> {
        let mut model = TypedModel::default();
        let p = model.add_source("p", f32::fact([2]))?;
        let q = model.add_source("q", f32::fact([2]))?;
        let fact = f32::fact([2]);
        let x = model.add_node("x", add(), tvec!(fact.clone()))?;
        let y = model.add_node("y", add(), tvec!(fact))?;
        model.add_edge(p, InletId::new(x, 0))?;
        model.add_edge(p, InletId::new(y, 0))?;
        model.add_edge(q, InletId::new(y, 1))?;
        model.add_edge(q, InletId::new(x, 1))?;
        let z = model.wire_node("z", mul(), &[x.into(), y.into()])?[0];
        model.select_output_outlets(&[z])?;

        let model = model.into_decluttered()?;
        let plan = SimplePlan::new(model)?;
        let out = plan.run(tvec!(tensor1(&[1f32, 2.]).into(), tensor1(&[3f32, 4.]).into()))?;
        out[0].close_enough(&tensor1(&[16f32, 36.]), Approximation::Exact)
    }
}
