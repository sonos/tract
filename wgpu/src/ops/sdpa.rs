use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensorExt;

use crate::kernels::sdpa::wgpu_flash_sdpa_dispatch;

/// Unmasked, non-causal scaled dot-product attention on the flash kernel:
/// inputs Q, K and V, output shaped as Q with the value head dim.
#[derive(Debug, Clone, PartialEq)]
pub struct WgpuFlashSdpa {
    pub scale: f32,
}

impl Eq for WgpuFlashSdpa {}

impl std::hash::Hash for WgpuFlashSdpa {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.scale.to_bits().hash(state);
    }
}

impl Op for WgpuFlashSdpa {
    fn name(&self) -> StaticName {
        "WgpuFlashSdpa".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("scale: {}", self.scale)])
    }

    op_as_typed_op!();
}

impl EvalOp for WgpuFlashSdpa {
    op_out_of_plan!();

    fn eval(&self, ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let (q, k, v) = args_3!(inputs);
        let (q, k, v) = (q.to_device_tensor()?, k.to_device_tensor()?, v.to_device_tensor()?);
        let mut shape = q.shape().to_vec();
        *shape.last_mut().unwrap() = *v.shape().last().unwrap();
        let output = tract_gpu::turn_handler::make_tensor_for_node(ctx, q.datum_type(), &shape)?;
        if output.len() > 0 {
            wgpu_flash_sdpa_dispatch(self.scale, q, k, v, &output)?;
        }
        Ok(tvec!(output.into_tensor().into_tvalue()))
    }
}

impl TypedOp for WgpuFlashSdpa {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        tract_gpu::utils::facts_to_device_facts(inputs, |facts| {
            let mut shape: TVec<_> = facts[0].shape.to_tvec();
            *shape.last_mut().unwrap() = facts[2].shape.last().unwrap().clone();
            Ok(tvec!(facts[0].datum_type.fact(shape)))
        })
        .with_context(|| "Error while computing facts for WgpuFlashSdpa")
    }
}
