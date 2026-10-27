mod cast;
mod copy;
mod cumsum;
mod diag_gather;
mod dispatch;
mod gather;
mod gather_nd;
mod resize;
mod rotate_half;

pub use cast::Cast;
pub use cast::cuda_cast_dispatch;
pub use copy::Memcpy;
pub use cumsum::cuda_cumsum_dispatch;
pub use diag_gather::DiagGather;
pub use diag_gather::cuda_diag_gather_dispatch;
pub use dispatch::cuda_copy_nd_dispatch;
pub use gather::Gather;
pub use gather::cuda_gather_dispatch;
pub use gather_nd::cuda_gather_nd_dispatch;
pub use resize::cuda_resize;
pub use rotate_half::RotateHalf;
pub use rotate_half::cuda_rotate_half_dispatch;

pub fn all_functions() -> Vec<String> {
    use std::collections::HashSet;
    use tract_gpu::utils::BroadcastKind;
    let mut functions = HashSet::<String>::new();

    functions.extend(BroadcastKind::all_copy_kernel_names(""));

    functions.extend(
        tract_gpu::tensor::DeviceTensor::SUPPORTED_DT
            .into_iter()
            .flat_map(|dt1| {
                tract_gpu::tensor::DeviceTensor::SUPPORTED_DT.into_iter().map(move |dt2| (dt1, dt2))
            })
            .flat_map(|(dt1, dt2)| Cast.kernel_name(dt1, dt2).into_iter()),
    );

    functions.extend(
        tract_gpu::tensor::DeviceTensor::SUPPORTED_DT
            .into_iter()
            .flat_map(|dt| DiagGather.kernel_name(dt).into_iter()),
    );

    functions.extend(
        tract_gpu::tensor::DeviceTensor::SUPPORTED_DT
            .into_iter()
            .flat_map(|dt| Gather.kernel_name(dt).into_iter()),
    );

    functions.extend(
        tract_gpu::tensor::DeviceTensor::SUPPORTED_DT
            .into_iter()
            .flat_map(|dt| gather_nd::kernel_name(dt).ok()),
    );

    functions.extend(
        [tract_core::prelude::DatumType::F32, tract_core::prelude::DatumType::F16]
            .into_iter()
            .flat_map(|dt| resize::kernel_name(dt).ok()),
    );

    functions.extend(
        tract_gpu::tensor::DeviceTensor::SUPPORTED_DT
            .into_iter()
            .flat_map(|dt| cumsum::kernel_name(dt).ok()),
    );

    functions.into_iter().collect()
}
