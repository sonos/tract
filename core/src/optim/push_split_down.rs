use std::collections::HashMap;

use crate::model::{InletId, OutletId};
use crate::ops::prelude::*;
use crate::Model;

use itertools::Itertools;

#[derive(Debug)]
pub struct PushSplitDown;

impl super::OptimizerPass for PushSplitDown {
    fn pass(&self, model: &mut Model) -> TractResult<bool> {
        let mut done_something = false;
        loop {
            let mut remap = HashMap::<usize, usize>::new();
            for node in model.eval_order()? {
                for output in &model.node(node).outputs {
                    for (a, b) in output.successors.iter().tuple_combinations() {
                        if remap.contains_key(&a.node) {
                            continue;
                        }
                        let a = model.node(a.node);
                        let b = model.node(b.node);
                        if a.same_as(b) {
                            remap.insert(b.id, a.id);
                        }
                    }
                }
            }
            if remap.len() > 0 {
                for (&killed, &kept) in remap.iter() {
                    trace!("collapsing {} into {}", killed, kept);
                    let successors: Vec<InletId> = model
                        .node(killed)
                        .outputs
                        .iter()
                        .flat_map(|s| s.successors.iter())
                        .cloned()
                        .collect();
                    for succ in successors {
                        for input_ix in 0..model.node(succ.node).inputs.len() {
                            let outlet = model.node(succ.node).inputs[input_ix];
                            if outlet.node == killed {
                                model.add_edge(
                                    OutletId::new(kept, outlet.slot),
                                    InletId::new(succ.node, input_ix),
                                )?;
                            }
                        }
                    }
                    model.clear_inputs(killed)?;
                    if cfg!(debug_assertions) {
                        model.check_edges()?;
                    }
                    done_something = true;
                }
            } else {
                break;
            }
        }
        Ok(done_something)
    }
}
