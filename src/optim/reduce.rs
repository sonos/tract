use ops::prelude::*;
use ::Model;

pub struct Reduce;

impl super::OptimizerPass for Reduce {
    fn pass(model: &mut Model) -> TfdResult<bool> {
        let mut done_something = false;
        for id in model.eval_order()? {
            let reduced = {
                let node = &model.nodes()[id];
                debug!("Consider unarize {:?}", node);
                let input_facts: TVec<&TensorFact> = node
                    .inputs
                    .iter()
                    .map(|o| model.fact(*o))
                    .inspect(|fact| trace!("   Input {:?}", fact))
                    .collect::<TfdResult<_>>()?;
                let output_facts: TVec<&TensorFact> =
                    node.outputs.iter().map(|o| &o.fact).collect();
                node.op
                    .reduce(input_facts, output_facts)
                    .map_err(|e| format!("Unarizing node {:?}, {:?}", node, e))?
            };
            if let Some(red) = reduced {
                debug!("  Unarize {:?}", red);
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
