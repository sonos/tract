use ops::prelude::*;
use Model;

#[derive(Debug)]
pub struct Reduce(pub ReductionPhase);

impl super::OptimizerPass for Reduce {
    fn pass(&self, model: &mut Model) -> TractResult<bool> {
        let mut done_something = false;
        loop {
            let mut done_something_this_time = false;
            for id in model.eval_order()? {
                let reduced = {
                    let node = &model.nodes()[id];
                    debug!(
                        "Consider {:?} {} #{} ({})",
                        self,
                        node.name,
                        node.id,
                        node.op().name()
                    );
                    let input_facts: TVec<&TensorFact> = node
                        .inputs
                        .iter()
                        .map(|o| model.fact(*o))
                        .inspect(|fact| trace!("   Input {:?}", fact))
                        .collect::<TractResult<_>>()?;
                    let output_facts: TVec<&TensorFact> =
                        node.outputs.iter().map(|o| &o.fact).collect();
                    node.op
                        .reduce(input_facts, output_facts, self.0)
                        .map_err(|e| format!("Reduce {:?} node {:?}, {:?}", self.0, node, e))?
                };
                if let Some(red) = reduced {
                    debug!("  Unarize to {:?}", red);
                    use model::dsl::ModelDsl;
                    use model::{ InletId, OutletId };

                    let ::ops::ReducedOpRewire { mut ops, rewired } = red;
                    let inputs:Vec<OutletId> = rewired.into_iter().map(|ix| model.node(id).inputs[ix]).collect();
                    if ops.len() == 1 {
                        model.node_mut(id).op = ops.remove(0);
                        model.clear_inputs(id)?;
                        for (ix, i) in inputs.iter().enumerate() {
                            model.add_edge(*i, InletId::new(id, ix))?;
                        }
                    } else {
                        model.mut_nodes()[id].op = ops.pop().unwrap();
                        let name = format!("{}-{}", model.node(id).name, ops.len());
                        let mut created_node_id = model.add_node(name, ops.remove(0))?;
                        for (ix, i) in inputs.iter().enumerate() {
                            model.add_edge(*i, InletId::new(created_node_id, ix))?;
                        }
                        while ops.len() > 0 {
                            let name = format!("{}-{}", model.node(id).name, ops.len());
                            created_node_id = model.chain(name, ops.remove(0))?;
                        }
                        model.clear_inputs(id)?;
                        model.add_edge(OutletId::new(created_node_id, 0), InletId::new(id, 0))?;
                    }
                    if cfg!(debug_assertions) {
                        model.check_edges()?;
                    }
                    done_something_this_time = true
                }
            }
            done_something = done_something || done_something_this_time;
            if !done_something_this_time {
                break;
            }
        }
        Ok(done_something)
    }
}

