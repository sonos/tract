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
        let new_m = self.shape[0];
        let new_k: usize = self.shape[1..].iter().product();
        let new = bqs.with_shape(new_m, new_k)?;
        Ok(tvec!(new.into_tensor().into_tvalue()))
    }
}

impl TypedOp for BlockQuantIntoShape {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let input = inputs[0];
        let old = input
            .opaque_fact
            .as_ref()
            .and_then(|of| of.downcast_ref::<BlockQuantFact>())
            .context("Expects BlockQuantFact")?;
        let g: usize = input.shape[0].to_usize()?;
        let new_m = self.shape[0];
        let new_k: usize = self.shape[1..].iter().product();
        let bqf_shape = tvec!(g, new_m, new_k);
        let new = BlockQuantFact::new(old.format.clone(), bqf_shape.clone());
        let shape: TVec<TDim> = bqf_shape.iter().map(|d| d.to_dim()).collect();
        let fact = DatumType::Opaque.fact(&*shape).with_opaque_fact(new);
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
        let input = inputs[0];
        ensure!(input.shape.rank() == 3);
        let bqf = input
            .opaque_fact
            .as_ref()
            .and_then(|of| of.downcast_ref::<BlockQuantFact>())
            .context("Expect BlockQuantFact")?;
        let m: usize = input.shape[1].to_usize()?;
        let k: usize = input.shape[2].to_usize()?;
        ensure!(m % self.group == 0);
        let new_shape = tvec!(self.group, m / self.group, k);
        let opaque_fact = BlockQuantFact::new(bqf.format.clone(), new_shape.clone());
        let fact = DatumType::Opaque
            .fact(&*new_shape.iter().map(|d| d.to_dim()).collect::<TVec<_>>())
            .with_opaque_fact(opaque_fact);
        Ok(tvec!(fact))
    }
    as_op!();
}
