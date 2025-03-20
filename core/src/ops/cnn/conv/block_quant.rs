use tract_linalg::block_quant::BlockQuantFact;

use crate::internal::*;

#[derive(Debug, Clone, new, Hash)]
pub struct BlockQuantIntoShape {
    pub shape: TVec<usize>,
}

impl Op for BlockQuantIntoShape {
    fn name(&self) -> Cow<str> {
        "BlockQuantIntoShape".into()
    }
    op_as_typed_op!();
}

impl EvalOp for BlockQuantIntoShape {
    fn is_stateless(&self) -> bool {
        true
    }

    fn state(
        &self,
        _session: &mut SessionState,
        _node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(None)
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        Ok(inputs)
    }
}

impl TypedOp for BlockQuantIntoShape {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let old = inputs[0]
            .opaque_fact
            .as_ref()
            .and_then(|of| of.downcast_ref::<BlockQuantFact>())
            .context("Expects BlockQuantFact")?
            .clone();
        let new = BlockQuantFact::new(old.format, self.shape.clone());
        let fact = inputs[0].clone().with_opaque_fact(new);
        Ok(tvec!(fact))
    }
    as_op!();
}
