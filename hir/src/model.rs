use crate::internal::*;
use crate::infer::*;

impl ModelWireNode<InferenceFact, Box<dyn InferenceOp>> for InferenceModel {
    fn wire_node(
        &mut self,
        name: impl Into<String>,
        op: impl Into<Box<dyn InferenceOp>>,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let op = op.into();
        let output_facts: TVec<InferenceFact> =
            (0..op.nboutputs()?).map(|_| InferenceFact::default()).collect();
        let id = self.add_node(name, op, output_facts)?;
        inputs
            .iter()
            .enumerate()
            .try_for_each(|(ix, i)| self.add_edge(*i, InletId::new(id, ix)))?;
        Ok(self.node(id).outputs.iter().enumerate().map(|(ix, _)| OutletId::new(id, ix)).collect())
    }
}

