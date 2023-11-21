use crate::infer::*;
use crate::internal::*;

use tract_core::ops::cnn::MaxPool;
use tract_core::ops::cnn::PoolSpec;
use tract_core::ops::cnn::SumPool;

#[derive(Debug, Clone, new, Hash)]
pub struct HirSumPool {
    pub pool_spec: PoolSpec,
    pub count_include_pad: bool,
    pub normalize: bool,
}

impl Expansion for HirSumPool {
    fn name(&self) -> Cow<str> {
        "SumPool".into()
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(inputs, 1)?;
        check_output_arity(outputs, 1)?;
        s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
        rules_for_shape(&self.pool_spec, s, inputs, outputs)
    }

    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let c = self
            .pool_spec
            .data_format
            .shape(&model.outlet_fact(inputs[0])?.shape)?
            .c()
            .to_usize()
            .context("Expect constant integer depth")?;
        let pool_spec =
            PoolSpec { input_channels: c, output_channels: c, ..self.pool_spec.clone() };
        model.wire_node(
            prefix,
            SumPool {
                pool_spec,
                count_include_pad: self.count_include_pad,
                normalize: self.normalize,
            },
            inputs,
        )
    }
}

#[derive(Debug, Clone, new, Hash)]
pub struct HirMaxPool {
    pub pool_spec: PoolSpec,
    pub with_index_outputs: Option<DatumType>,
}

impl Expansion for HirMaxPool {
    fn name(&self) -> Cow<str> {
        "MaxPool".into()
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_output_arity(outputs, 1 + self.with_index_outputs.is_some() as usize)?;
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

    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let c = self
            .pool_spec
            .data_format
            .shape(&model.outlet_fact(inputs[0])?.shape)?
            .c()
            .to_usize()
            .context("Expect constant integer depth")?;
        let pool_spec =
            PoolSpec { input_channels: c, output_channels: c, ..self.pool_spec.clone() };
        model.wire_node(
            prefix,
            MaxPool { pool_spec, with_index_outputs: self.with_index_outputs },
            inputs,
        )
    }
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
            &pool_spec.kernel_shape,
            pool_spec.dilations.as_ref().unwrap_or(&ones),
            pool_spec.strides.as_ref().unwrap_or(&ones),
        );
        for o in outputs {
            for (ix, d) in computed.iter().enumerate() {
                s.equals(&o.shape[ix + ishape.h_axis()], &d.convoluted)?;
            }
            if ishape.n_axis().is_some() {
                s.equals(&o.shape[ishape.n_axis().unwrap()], ishape.n_dim().unwrap())?;
            }
            // hack for max and sum pool, convolutions know this and deal with it on their side
            if pool_spec.input_channels == 0 && pool_spec.output_channels == 0 {
                s.equals(&o.shape[ishape.c_axis()], ishape.c_dim())?;
            }
        }
        Ok(())
    })
}
