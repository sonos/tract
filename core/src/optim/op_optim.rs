use crate::internal::*;

pub struct OpOptim(
    pub &'static str,
    pub  fn(
        op: &dyn TypedOp,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>>,
    pub usize,
);

impl OpOptim {
    fn full_pass(&mut self, new: &TypedModel) -> TractResult<Option<TypedModelPatch>> {
        for (ix, &id) in new.eval_order()?.iter().enumerate().skip(self.2) {
            let node = &new.nodes()[id];
            let patch = (self.1)(node.op.as_ref(), &new, node)
                .with_context(|| format!("{:?} node {}", self, node))?;
            if let Some(mut p) = patch {
                p.push_context(format!("{:?} {}", self, node));
                self.2 = ix;
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

    fn next(&mut self, model: &TypedModel) -> TractResult<Option<TypedModelPatch>> {
        self.full_pass(model)
    }
}
