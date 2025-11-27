use std::ops::Range;

use tract_linalg::block_quant::BlockQuantFact;

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
        _session: &mut SessionState,
        _node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(None)
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let mut input = args_1!(inputs).into_tensor();
        for o in input.as_slice_mut::<Opaque>()? {
            let old =
                o.0.downcast_ref::<BlobWithFact>().context("Expects only BlobWithFact")?;
            let bqf = old.fact.downcast_ref::<BlockQuantFact>().context("Expects only BlockQuantFact")?;
            let new = BlobWithFact {
                value: old.value.clone(),
                fact: Box::new(BlockQuantFact::new(bqf.format.clone(), self.shape.clone())),
            };
            *o = Opaque(Arc::new(new))
        }
        Ok(tvec!(input.into_tvalue()))
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

fn split_rows(bqf: &BlockQuantFact, blob: &Blob, range: Range<usize>) -> TractResult<BlobWithFact> {
    let row_bytes =
        bqf.k() / bqf.format.block_len() * bqf.format.block_bytes();
    let mut value =
        unsafe { Blob::new_for_size_and_align(range.len() * row_bytes, vector_size()) };
    value.copy_from_slice(&blob[range.start * row_bytes..][..range.len() * row_bytes]);
    let mut shape: TVec<usize> = bqf.shape().into();
    shape[0] = range.len();
    Ok(BlobWithFact {
        fact: Box::new(BlockQuantFact::new(bqf.format.clone(), shape)),
        value: Arc::new(value),
    })
}

impl EvalOp for SplitGroupBlockQuant {
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
        let input = args_1!(inputs);
        let bwf = input
            .to_scalar::<Opaque>()?
            .downcast_ref::<BlobWithFact>()
            .context("Expect BlobWithFact")?;
        let bqf = bwf.fact.downcast_ref::<BlockQuantFact>().context("Expect BlockQuantFact")?;
        let rows_per_group = bqf.m() / self.group;
        let splits = (0..self.group)
            .map(|g| {
                Ok(Opaque(Arc::new(split_rows(bqf, &bwf.value, g * rows_per_group..(g + 1) * rows_per_group)?)))
            })
            .collect::<TractResult<Vec<_>>>()?;
        let output = tensor1(&splits);
        Ok(tvec!(output.into_tvalue()))
    }
}

impl TypedOp for SplitGroupBlockQuant {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
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
        let fact = Opaque::fact([self.group]).with_opaque_fact(opaque_fact);
        Ok(tvec!(fact))
    }
    as_op!();
}
