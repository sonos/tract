mod apply_rope;
mod gelu_approximate;
mod leaky_relu;
mod reduce;
mod rms_norm;
mod scaled_masked_softmax;
mod softmax;

pub use apply_rope::ApplyRope;
pub use gelu_approximate::GeluApproximate;
pub use leaky_relu::LeakyRelu;
pub use reduce::Reducer;
pub use rms_norm::RmsNorm;
pub use scaled_masked_softmax::ScaledMaskedSoftmax;
pub use softmax::Softmax;

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
            .flat_map(|(op, dt, n_cols)| op.kernel_name(dt, n_cols).into_iter()),
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
