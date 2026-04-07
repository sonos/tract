mod apply_rope;
mod gelu_approximate;
mod leaky_relu;
mod reduce;
mod rms_norm;
mod scaled_masked_softmax;
mod softmax;

pub use apply_rope::{ApplyRope, cuda_apply_rope_dispatch};
pub use gelu_approximate::GeluApproximate;
pub use gelu_approximate::cuda_gelu_approximate_dispatch;
pub use leaky_relu::LeakyRelu;
pub use leaky_relu::cuda_leaky_relu_dispatch;
pub use reduce::{Reducer, cuda_reduce_launch};
pub use rms_norm::RmsNorm;
pub use rms_norm::cuda_rms_norm_dispatch;
pub use scaled_masked_softmax::{ScaledMaskedSoftmax, cuda_scaled_masked_softmax_dispatch};
pub use softmax::Softmax;
pub use softmax::cuda_softmax_dispatch;

use crate::kernels::{BroadcastKind, MAX_THREADS};

fn sms_block_sizes() -> Vec<i32> {
    let mut range = vec![0i32];

    for i in 5..=15 {
        range.push(2i32.pow(i));
    }

    range
}

pub fn all_functions() -> Vec<String> {
    use std::collections::HashSet;
    let mut functions = HashSet::<String>::new();

    functions.extend(
        Reducer::ALL
            .into_iter()
            .flat_map(|op| {
                tract_gpu::tensor::DeviceTensor::SUPPORTED_DT.into_iter().map(move |dt| (op, dt))
            })
            .flat_map(|(op, dt)| [0, MAX_THREADS].into_iter().map(move |n_cols| (op, dt, n_cols)))
            .flat_map(|(op, dt, n_cols)| reduce::kernel_name(&op, dt, n_cols).into_iter()),
    );
    functions.extend(
        tract_gpu::tensor::DeviceTensor::SUPPORTED_DT
            .into_iter()
            .flat_map(|dt| [0, MAX_THREADS].into_iter().map(move |n_cols| (dt, n_cols)))
            .flat_map(|(dt, n_cols)| Softmax.kernel_name(dt, n_cols).into_iter()),
    );

    functions.extend(
        tract_gpu::tensor::DeviceTensor::SUPPORTED_DT
            .into_iter()
            .flat_map(|dt| sms_block_sizes().into_iter().map(move |bs| (dt, bs as usize)))
            .flat_map(|(dt, bs)| ScaledMaskedSoftmax.kernel_name(dt, bs).into_iter()),
    );

    functions.extend(
        tract_gpu::tensor::DeviceTensor::SUPPORTED_DT
            .into_iter()
            .flat_map(|dt| [0, MAX_THREADS].into_iter().map(move |n_cols| (dt, n_cols)))
            .flat_map(|(dt, n_cols)| RmsNorm.kernel_name(dt, n_cols).into_iter()),
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
            .flat_map(|dt| LeakyRelu.kernel_name(dt).into_iter()),
    );

    functions.into_iter().collect()
}
