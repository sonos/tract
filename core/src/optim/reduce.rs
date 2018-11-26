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
                debug!("  Unarize to {:?}", red.new_op.name());
                let mut node = &mut model.mut_nodes()[id];
                let ::ops::ReducedOpRewire { new_op, rewired } = red;
                node.op = new_op;
                let new_inputs = rewired.into_iter().map(|ix| node.inputs[ix]).collect();
                node.inputs = new_inputs;
                done_something = true
            }
        }
        Ok(done_something)
    }
}

