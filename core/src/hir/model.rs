use crate::internal::*;
use crate::infer::*;
use std::convert::TryFrom;
use std::fmt;

impl<TI: Fact + Clone + 'static, O, E> ModelDslConst for ModelImpl<TI, O>
where
    TractError: From<E>,
    TI: Fact + Clone + 'static + for<'a> TryFrom<&'a InferenceFact, Error = E>,
    O: fmt::Debug
        + fmt::Display
        + From<crate::ops::konst::Const>
        + AsRef<dyn Op>
        + AsMut<dyn Op>
        + Clone
        + 'static,
{
    fn add_const(
        &mut self,
        name: impl Into<String>,
        v: impl IntoArcTensor,
    ) -> TractResult<OutletId> {
        let v = v.into_arc_tensor();
        let fact = TI::try_from(&InferenceFact::from(&*v))?;
        self.add_node(name, crate::ops::konst::Const::new(v), tvec!(fact)).map(|id| id.into())
    }
}

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

