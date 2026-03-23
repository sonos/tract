use tract_linalg::block_quant::{BlockQuantFact, BlockQuantStorage};

use crate::internal::*;

#[derive(Debug, Clone, new, Hash)]
pub struct BlockQuantIntoShape {
    pub shape: TVec<usize>,
}

impl Op for BlockQuantIntoShape {
    fn name(&self) -> StaticName {
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
        _session: &TurnState,
        _node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(None)
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs).into_tensor();
        let bqs = input.try_storage_as::<BlockQuantStorage>()?;
        let new_k: usize = self.shape[1..].iter().product();
        let new = bqs.with_shape(self.shape[0], new_k)?;
        Ok(tvec!(new.into_tensor().into_tvalue()))
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

#[derive(Debug, Clone, new, Hash)]
pub struct SplitGroupBlockQuant {
    pub group: usize,
}

impl Op for SplitGroupBlockQuant {
    fn name(&self) -> StaticName {
        "SplitGroupBlockQuant".into()
    }

    op_as_typed_op!();
}

impl EvalOp for SplitGroupBlockQuant {
    fn is_stateless(&self) -> bool {
        true
    }

    fn state(
        &self,
        _session: &TurnState,
        _node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(None)
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        let bqs = input.try_storage_as::<BlockQuantStorage>()?;
        let output = bqs.split_m(self.group)?;
        Ok(tvec!(output.into_tensor().into_tvalue()))
    }
}

impl TypedOp for SplitGroupBlockQuant {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        // Output is a scalar Opaque tensor: logical m×k dimensions live inside
        // BlockQuantStorage/BlockQuantFact, not in the tensor shape.
        let input = inputs[0];
        ensure!(input.shape.rank() == 0);
        let bqf = input
            .opaque_fact
            .as_ref()
            .and_then(|of| of.downcast_ref::<BlockQuantFact>())
            .context("Expect BlockQuantFact")?;
        let mut shape: TVec<usize> = bqf.shape().into();
        shape[0] /= self.group;
        let opaque_fact = BlockQuantFact::new(bqf.format.clone(), shape);
        let fact = Opaque::fact::<[usize; 0]>([]).with_opaque_fact(opaque_fact);
        Ok(tvec!(fact))
    }
    as_op!();
}
