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
                for (a, b) in output.successors.iter().tuple_combinations() {
                    if a.node == b.node {
                        // found where a square is implemented using a mul with duplicate input
                        continue;
                    }
                    if patch.obliterate.contains(&b.node) {
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
