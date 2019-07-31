use crate::internal::*;

use itertools::Itertools;

#[derive(Debug)]
pub struct PushSplitDown;

impl super::TypedPass for PushSplitDown {
    fn pass(&self, model: &mut TypedModel) -> TractResult<bool> {
        let mut done_something = false;
        loop {
            let mut patch = TypedModelPatch::default();
            for node in model.eval_order()? {
                for output in &model.node(node).outputs {
                    for (a, b) in output.successors.iter().tuple_combinations() {
                        if patch.obliterate.contains(&b.node) {
                            continue;
                        }
                        let a = model.node(a.node);
                        let b = model.node(b.node);
                        if a.same_as(b) {
                            for slot in 0..b.outputs.len() {
                                let tap = patch.tap_model(model, OutletId::new(a.id, slot))?;
                                patch.shunt_outside(OutletId::new(b.id, slot), tap)?;
                                patch.obliterate(b.id)?;
                            }
                        }
                    }
                }
            }
            if !patch.is_empty() {
                done_something = true;
                patch.apply(model)?;
            } else {
                break;
            }
        }
        Ok(done_something)
    }
}
