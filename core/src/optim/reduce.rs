use ops::prelude::*;
use Model;

#[derive(Debug)]
pub struct Reduce(pub ReductionPhase);

impl super::OptimizerPass for Reduce {
    fn pass(&self, model: &mut Model) -> TractResult<bool> {
        let mut done_something = false;
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
                let inputs = rewired.into_iter().map(|ix| model.node(id).inputs[ix]).collect();
                if ops.len() == 1 {
                    let mut node = &mut model.mut_nodes()[id];
                    node.op = ops.remove(0);
                    node.inputs = inputs;
                } else {
                    model.mut_nodes()[id].op = ops.pop().unwrap();
                    let name = format!("{}-{}", model.node(id).name, ops.len());
                    let mut created_node_id = model.add_node(name, ops.remove(0))?;
                    model.mut_nodes()[created_node_id].inputs = inputs;
                    while ops.len() > 0 {
                        let name = format!("{}-{}", model.node(id).name, ops.len());
                        created_node_id = model.chain(name, ops.remove(0))?;
                    }
                    model.mut_nodes()[id].inputs.clear();
                    model.add_edge(OutletId::new(created_node_id, 0), InletId::new(id, 0))?;
                }
                done_something = true
            }
        }
        Ok(done_something)
    }
}

