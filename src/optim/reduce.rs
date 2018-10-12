use { Model, TfdResult, TensorFact, TVec };

pub struct Reduce;

impl super::OptimizerPass for Reduce {

    fn pass(model: &mut Model) -> TfdResult<bool> {
        let mut done_something = false;
        for id in model.eval_order()? {
            let reduced = {
                let node = &model.nodes()[id];
                let input_facts: TVec<&TensorFact> = node
                    .inputs
                    .iter()
                    .map(|o| model.fact(*o))
                    .collect::<TfdResult<_>>()?;
                let output_facts: TVec<&TensorFact> =
                    node.outputs.iter().map(|o| &o.fact).collect();
                node.op.reduce(input_facts, output_facts)?
            };
            if let Some(red) = reduced {
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
