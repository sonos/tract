pub mod apply_rope;
pub mod gelu_approximate;
pub mod leaky_relu;
pub mod reduce;
pub mod rms_norm;
pub mod scaled_masked_softmax;
pub mod silu;
pub mod softmax;

pub use apply_rope::{ApplyRope, metal_apply_rope_dispatch};
pub use gelu_approximate::GeluApproximate;
pub use gelu_approximate::metal_gelu_approximate_dispatch;
pub use leaky_relu::LeakyRelu;
pub use leaky_relu::metal_leaky_relu_dispatch;
pub use reduce::{Reducer, metal_reduce_launch};
pub use rms_norm::RmsNorm;
pub use rms_norm::metal_rms_norm_dispatch;
pub use scaled_masked_softmax::{ScaledMaskedSoftmax, metal_scaled_masked_softmax_dispatch};
pub use silu::Silu;
pub use softmax::Softmax;
pub use softmax::metal_softmax_dispatch;

use crate::kernels::BroadcastKind;

pub fn all_functions() -> Vec<String> {
    use std::collections::HashSet;
    let mut functions = HashSet::<String>::new();

    functions.extend(
        Reducer::ALL
            .into_iter()
            .flat_map(|op| {
                tract_gpu::tensor::DeviceTensor::SUPPORTED_DT.into_iter().map(move |dt| (op, dt))
            })
            .flat_map(|(op, dt)| reduce::kernel_name(&op, dt).into_iter()),
    );
    functions.extend(
        tract_gpu::tensor::DeviceTensor::SUPPORTED_DT
            .into_iter()
            .flat_map(|dt| Softmax.kernel_name(dt).into_iter()),
    );

    functions.extend(
        tract_gpu::tensor::DeviceTensor::SUPPORTED_DT
            .into_iter()
            .flat_map(|dt| ScaledMaskedSoftmax.kernel_name(dt).into_iter()),
    );

    functions.extend(
        tract_gpu::tensor::DeviceTensor::SUPPORTED_DT
            .into_iter()
            .flat_map(|dt| [true, false].into_iter().map(move |is_l4| (dt, is_l4)))
            .flat_map(|(dt, is_l4)| RmsNorm.kernel_name(dt, is_l4).into_iter()),
    );

    functions.extend(
        BroadcastKind::ALL
            .into_iter()
            .flat_map(|brdcast| {
                tract_gpu::tensor::DeviceTensor::SUPPORTED_DT
                    .into_iter()
                    .map(move |dt| (dt, brdcast))
            })
            .flat_map(|(dt, brdcast)| ApplyRope.kernel_name(dt, brdcast).into_iter()),
    );

    functions.extend(
        tract_gpu::tensor::DeviceTensor::SUPPORTED_DT
            .into_iter()
            .flat_map(|dt| [true, false].into_iter().map(move |fast_impl| (dt, fast_impl)))
            .flat_map(|(dt, fast_impl)| GeluApproximate { fast_impl }.kernel_name(dt).into_iter()),
    );

    functions.extend(
        tract_gpu::tensor::DeviceTensor::SUPPORTED_DT
            .into_iter()
            .flat_map(|dt| [true, false].into_iter().map(move |is_l4| (dt, is_l4)))
            .flat_map(|(dt, is_l4)| Silu.kernel_name(dt, is_l4).into_iter()),
    );

    functions.extend(
        tract_gpu::tensor::DeviceTensor::SUPPORTED_DT
            .into_iter()
            .flat_map(|dt| LeakyRelu.kernel_name(dt).into_iter()),
    );
    functions.into_iter().collect()
}
