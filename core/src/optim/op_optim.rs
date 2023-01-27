use crate::internal::*;

use super::OptimizerSession;

#[derive(Clone)]
pub struct OpOptim(
    pub &'static str,
    pub  fn(
        op: &dyn TypedOp,
        session: &mut OptimizerSession,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>>,
    pub usize,
);

impl OpOptim {
    fn full_pass(
        &mut self,
        session: &mut OptimizerSession,
        new: &TypedModel,
    ) -> TractResult<Option<TypedModelPatch>> {
        for (ix, &id) in new.eval_order()?.iter().enumerate().skip(self.2) {
            let node = &new.nodes()[id];
            let patch = (self.1)(node.op.as_ref(), session, new, node)
                .with_context(|| format!("{self:?} node {node}"))?;
            if let Some(mut p) = patch {
                p.push_context(format!("{self:?} {node}"));
                self.2 = ix + p.dont_apply_twice.is_some() as usize;
                return Ok(Some(p));
            }
        }
        Ok(None)
    }
}

impl std::fmt::Debug for OpOptim {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(fmt, "{}", self.0)
    }
}

impl super::TypedPass for OpOptim {
    fn reset(&mut self) -> TractResult<()> {
        self.2 = 0;
        Ok(())
    }

    fn next(
        &mut self,
        session: &mut OptimizerSession,
        model: &TypedModel,
    ) -> TractResult<Option<TypedModelPatch>> {
        self.full_pass(session, model)
    }
}
