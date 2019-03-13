use crate::model::*;
use crate::ops::prelude::*;
use std::ops::{Deref, DerefMut};

#[derive(Clone, Debug)]
pub struct ModelPatch<TI: TensorInfo> {
    model: Model<TI>,
    incoming: HashMap<OutletId, OutletId>,
    shunt_outlet_by: HashMap<OutletId, OutletId>,
}

impl<TI: TensorInfo> Default for ModelPatch<TI> {
    fn default() -> ModelPatch<TI> {
        ModelPatch {
            model: Model::default(),
            incoming: HashMap::new(),
            shunt_outlet_by: HashMap::new(),
        }
    }
}

impl<TI: TensorInfo> Deref for ModelPatch<TI> {
    type Target = Model<TI>;
    fn deref(&self) -> &Model<TI> {
        &self.model
    }
}

impl<TI: TensorInfo> DerefMut for ModelPatch<TI> {
    fn deref_mut(&mut self) -> &mut Model<TI> {
        &mut self.model
    }
}

impl<TI: TensorInfo> ModelPatch<TI> {
    pub fn tap_model(&mut self, model: &Model<TI>, outlet: OutletId) -> TractResult<OutletId> {
        let fact = model.fact(outlet)?;
        let node_id =
            self.add_source_fact(format!("incoming-{}/{}", outlet.node, outlet.slot), fact.clone())?;
        let inside = OutletId::new(node_id, 0);
        self.incoming.insert(inside, outlet);
        Ok(inside)
    }

    pub fn shunt_outside(&mut self, outlet: OutletId, by: OutletId) -> TractResult<()> {
        self.shunt_outlet_by.insert(outlet, by);
        Ok(())
    }

    pub fn single_unary_op<O: Into<Box<Op>>>(
        patched_model: &Model<TI>,
        node: &Node<TI>,
        new_op: O,
    ) -> TractResult<ModelPatch<TI>> {
        let mut patch = ModelPatch::default();
        patch.tap_model(&patched_model, node.inputs[0])?;
        let new_op = new_op.into();
        let by = patch.chain_facts(
            format!("{}-as-{}", node.name, new_op.name()),
            new_op,
            tvec!(node.outputs[0].fact.clone()),
        )?;
        patch.shunt_outside(OutletId::new(node.id, 0), OutletId::new(by, 0))?;
        Ok(patch)
    }

    pub fn apply(self, model: &mut Model<TI>) -> TractResult<()> {
        let ModelPatch { model: patch, incoming: mut mapping, shunt_outlet_by } = self;
        for node in patch.nodes {
            if node.op_is::<crate::ops::source::Source>() {
                continue;
            }
            let Node { id, name, inputs, op, outputs } = node;
            let n_outputs = outputs.len();
            let facts = outputs.into_iter().map(|of| of.fact).collect();
            let added_node_id = model.add_node_disable_output_guess(name, op, facts, true)?;
            for (ix, input) in inputs.into_iter().enumerate() {
                model.add_edge(mapping[&input], InletId::new(added_node_id, ix))?;
            }
            for ix in 0..n_outputs {
                mapping.insert(OutletId::new(id, ix), OutletId::new(added_node_id, ix));
            }
        }
        for (outlet, by) in shunt_outlet_by {
            let fixed_by = mapping[&by];
            let succs = model.nodes[outlet.node].outputs[outlet.slot].successors.clone();
            for succ in succs {
                model.add_edge(fixed_by, succ)?;
            }
            for o in model.outputs.iter_mut() {
                if *o == outlet {
                    *o = fixed_by;
                }
            }
        }
        Ok(())
    }
}
