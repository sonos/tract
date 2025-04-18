pub mod apply_rope;
pub mod gelu_approximate;
pub mod reduce;
pub mod rms_norm;
pub mod scaled_masked_softmax;
pub mod silu;
pub mod softmax;

pub use apply_rope::ApplyRope;
pub use gelu_approximate::GeluApproximate;
pub use reduce::Reducer;
pub use rms_norm::RmsNorm;
pub use scaled_masked_softmax::ScaledMaskedSoftmax;
pub use silu::Silu;
pub use softmax::Softmax;

pub fn all_functions() -> Vec<String> {
    use std::collections::HashSet;
    let mut functions = HashSet::<String>::new();

    functions.extend(
        Reducer::ALL
            .into_iter()
            .flat_map(|op| {
                tract_gpu::tensor::DeviceTensor::SUPPORTED_DT.into_iter().map(move |dt| (op, dt))
            })
            .flat_map(|(op, dt)| op.kernel_name(dt).into_iter()),
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

    functions.into_iter().collect()
}
