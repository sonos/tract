use crate::infer::*;
use crate::internal::*;

pub use tract_core::ops::cnn::{MaxPool, PoolSpec, SumPool};

impl InferenceRulesOp for SumPool {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 1)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
        rules_for_shape(&self.pool_spec, s, inputs, outputs)
    }

    as_op!();
    to_typed!();
}

impl InferenceRulesOp for MaxPool {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_output_arity(&outputs, 1 + self.with_index_outputs.is_some() as usize)?;
        s.equals(&outputs[0].rank, &inputs[0].rank)?;
        s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
        if let Some(idt) = self.with_index_outputs {
            s.equals(&outputs[1].datum_type, idt)?;
            s.equals(&outputs[1].shape, &outputs[0].shape)?;
        }
        rules_for_shape(&self.pool_spec, s, inputs, outputs)
    }

    fn nboutputs(&self) -> TractResult<usize> {
        Ok(1 + self.with_index_outputs.is_some() as usize)
    }

    as_op!();
    to_typed!();
}

pub fn rules_for_shape<'r, 'p: 'r, 's: 'r>(
    pool_spec: &'s PoolSpec,
    s: &mut Solver<'r>,
    inputs: &'p [TensorProxy],
    outputs: &'p [TensorProxy],
) -> InferenceResult {
    s.equals(&outputs[0].rank, &inputs[0].rank)?;
    s.given(&inputs[0].shape, move |s, ishape| {
        let ishape = pool_spec.data_format.shape(ishape)?;
        let ones = tvec![1; ishape.hw_rank()];
        let computed = pool_spec.padding.compute(
            ishape.hw_dims(),
            &*pool_spec.kernel_shape,
            pool_spec.dilations.as_ref().unwrap_or(&ones),
            pool_spec.strides.as_ref().unwrap_or(&ones),
        );
        for o in 0..outputs.len() {
            for (ix, d) in computed.iter().enumerate() {
                s.equals(&outputs[o].shape[ix + ishape.h_axis()], &d.convoluted)?;
            }
            if ishape.n_axis().is_some() {
                s.equals(&outputs[o].shape[ishape.n_axis().unwrap()], ishape.n_dim().unwrap())?;
            }
            if let Some(c) = pool_spec.output_channel_override {
                s.equals(&outputs[o].shape[ishape.c_axis()], c.to_dim())?;
            } else {
                s.equals(&outputs[o].shape[ishape.c_axis()], ishape.c_dim())?;
            }
        }
        Ok(())
    })
}
