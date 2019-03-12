use crate::model::*;
use crate::ops::prelude::*;
use std::ops::{Deref, DerefMut};

#[derive(Clone, Debug, Default)]
struct ModelPatch {
    model: Model,
    incoming: HashMap<OutletId, OutletId>,
    replacement: HashMap<OutletId, OutletId>,
}

impl Deref for ModelPatch {
    type Target = Model;
    fn deref(&self) -> &Model {
        &self.model
    }
}

impl DerefMut for ModelPatch {
    fn deref_mut(&mut self) -> &mut Model {
        &mut self.model
    }
}

impl ModelPatch {
    pub fn tap_outside(&mut self, outside: OutletId, fact: TensorFact) -> TractResult<OutletId> {
        let node_id =
            self.add_source_fact(format!("incoming-{}/{}", outside.node, outside.slot), fact)?;
        let inside = OutletId::new(node_id, 0);
        self.incoming.insert(inside, outside);
        Ok(inside)
    }

    pub fn shunt_outside(&mut self, old: OutletId, new: OutletId) -> TractResult<()> {
        self.replacement.insert(old, new);
        Ok(())
    }

    pub fn single_unary_op<O: Op>(
        patched_model: &Model,
        node: &Node,
        new_op: O,
    ) -> TractResult<ModelPatch> {
        let mut patch = ModelPatch::default();
        let input: OutletId = node.inputs[0];
        let fact = patched_model.fact(input)?;
        patch.tap_outside(input, fact.clone());
        let replacement =
            patch.chain(format!("{}-as-{}", node.name, new_op.name()), Box::new(new_op))?;
        patch.shunt_outside(OutletId::new(replacement, 0), OutletId::new(node.id, 0))?;
        Ok(patch)
    }
}
