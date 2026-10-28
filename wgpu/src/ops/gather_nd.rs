use tract_core::internal::*;
use tract_core::ops::array::GatherNd;
use tract_gpu::tensor::DeviceTensorExt;

use crate::kernels::gather_nd::wgpu_gather_nd_dispatch;

/// [`GatherNd`] on the device, reading its indices as f32.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WgpuGatherNd {
    pub batch_dims: usize,
}

impl Op for WgpuGatherNd {
    fn name(&self) -> StaticName {
        "WgpuGatherNd".into()
    }

    op_as_typed_op!();
}

impl EvalOp for WgpuGatherNd {
    op_out_of_plan!();

    fn eval(&self, ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let (data, indices) = args_2!(inputs);
        let data = data.to_device_tensor()?;
        let indices = indices.to_device_tensor()?;
        let n = *indices.shape().last().context("GatherNd indices of rank 0")?;
        let mut shape: TVec<usize> = indices.shape()[..indices.rank() - 1].into();
        shape.extend(data.shape()[self.batch_dims + n..].iter().copied());
        let output = tract_gpu::turn_handler::make_tensor_for_node(ctx, data.datum_type(), &shape)?;
        if output.len() > 0 {
            wgpu_gather_nd_dispatch(self.batch_dims, data, indices, &output)?;
        }
        Ok(tvec!(output.into_tensor().into_tvalue()))
    }
}

impl TypedOp for WgpuGatherNd {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        tract_gpu::utils::facts_to_device_facts(inputs, |facts| {
            GatherNd::new(self.batch_dims).output_facts(facts)
        })
        .with_context(|| "Error while computing facts for WgpuGatherNd")
    }
}
