use crate::tensor::DeviceTensorExt;
use crate::utils::compute_broadcast_strides;
use tract_core::internal::*;
use tract_core::ops::downsample::Downsample;

/// [`Downsample`] with a forward stride, as one strided copy: output index `i`
/// along `axis` reads input index `modulo + i * stride`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GpuDownsample {
    pub inner: Downsample,
}

impl GpuDownsample {
    pub fn from_core(op: &Downsample) -> Option<Self> {
        (op.stride > 0).then(|| Self { inner: op.clone() })
    }
}

impl Op for GpuDownsample {
    fn name(&self) -> StaticName {
        "GpuDownsample".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        self.inner.info()
    }

    op_as_typed_op!();
}

impl EvalOp for GpuDownsample {
    op_out_of_plan!();

    fn eval(&self, ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input_value = args_1!(inputs);
        let input = input_value.to_device_tensor()?;
        let Downsample { axis, stride, modulo } = self.inner;

        let mut o_shape: TVec<usize> = input.shape().into();
        o_shape[axis] = o_shape[axis].saturating_sub(modulo).div_ceil(stride as usize);
        let output = crate::turn_handler::make_tensor_for_node(ctx, input.datum_type(), &o_shape)?;

        if o_shape[axis] != 0 {
            let mut strides: TVec<isize> = input.strides().into();
            let offset = modulo * strides[axis] as usize * input.datum_type().size_of();
            strides[axis] *= stride;
            let strides = compute_broadcast_strides(&o_shape, &strides)?;
            let ctx = crate::device::get_context()?;
            ctx.copy_nd(input, offset, &strides, &output, 0, output.shape(), output.strides())?;
        }
        Ok(tvec![output.into_tensor().into_tvalue()])
    }
}

impl TypedOp for GpuDownsample {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        crate::utils::facts_to_device_facts(inputs, |facts| self.inner.output_facts(facts))
            .with_context(|| format!("Error while computing facts for {:?}", self.name()))
    }

    as_op!();
}
