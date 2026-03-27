use tract_linalg::block_quant::{BlockQuantFact, BlockQuantStorage};

use crate::internal::*;

#[derive(Debug, Clone, new, Hash, PartialEq, Eq)]
pub struct BlockQuantIntoShape {
    pub shape: TVec<usize>,
}

impl Op for BlockQuantIntoShape {
    fn name(&self) -> StaticName {
        "BlockQuantIntoShape".into()
    }
    op_as_typed_op!();
    impl_op_same_as!();
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
        let g = input.shape()[0];
        let bqs = input.try_storage_as::<BlockQuantStorage>()?.clone();
        let new_m = self.shape[0];
        let new_k: usize = self.shape[1..].iter().product();
        Ok(tvec!(bqs.into_tensor_with_shape(input.datum_type(), &[g, new_m, new_k]).into_tvalue()))
    }
}

impl TypedOp for BlockQuantIntoShape {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let input = inputs[0];
        let old = input
            .exotic_fact
            .as_ref()
            .and_then(|of| of.downcast_ref::<BlockQuantFact>())
            .context("Expects BlockQuantFact")?;
        let g: usize = input.shape[0].to_usize()?;
        let new_m = self.shape[0];
        let new_k: usize = self.shape[1..].iter().product();
        let bqf_shape = tvec!(g, new_m, new_k);
        let new = BlockQuantFact::new(old.format.clone(), bqf_shape.clone());
        let shape: TVec<TDim> = bqf_shape.iter().map(|d| d.to_dim()).collect();
        let fact = inputs[0].datum_type.fact(&*shape).with_exotic_fact(new);
        Ok(tvec!(fact))
    }
    as_op!();
}

#[derive(Debug, Clone, new, Hash, PartialEq, Eq)]
pub struct SplitGroupBlockQuant {
    pub group: usize,
}

impl Op for SplitGroupBlockQuant {
    fn name(&self) -> StaticName {
        "SplitGroupBlockQuant".into()
    }

    op_as_typed_op!();
    impl_op_same_as!();
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
        let bqs = input.try_storage_as::<BlockQuantStorage>()?.clone();
        let mut new_shape: TVec<usize> = input.shape().into();
        let o = new_shape[0];
        new_shape[0] = o / self.group;
        new_shape.insert(0, self.group);
        Ok(tvec!(bqs.into_tensor_with_shape(input.datum_type(), &new_shape).into_tvalue()))
    }
}

impl TypedOp for SplitGroupBlockQuant {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let input = inputs[0];
        let bqf = input
            .exotic_fact
            .as_ref()
            .and_then(|of| of.downcast_ref::<BlockQuantFact>())
            .context("Expect BlockQuantFact")?;
        let o: usize = input.shape[0].to_usize()?;
        ensure!(o % self.group == 0);
        let mut new_shape: TVec<usize> =
            input.shape.iter().map(|d| d.to_usize()).collect::<TractResult<_>>()?;
        new_shape[0] = o / self.group;
        new_shape.insert(0, self.group);
        let exotic_fact = BlockQuantFact::new(bqf.format.clone(), new_shape.clone());
        let fact = inputs[0]
            .datum_type
            .fact(&*new_shape.iter().map(|d| d.to_dim()).collect::<TVec<_>>())
            .with_exotic_fact(exotic_fact);
        Ok(tvec!(fact))
    }
    as_op!();
}
