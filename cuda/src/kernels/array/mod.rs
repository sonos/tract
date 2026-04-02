mod broadcast;
mod cast;
mod concat;
mod copy;
mod pad;
mod permute_axes;
mod rotate_half;

pub use broadcast::{MultiBroadcast, cuda_broadcast_dispatch};
pub use cast::Cast;
pub use concat::{Concat, cuda_concat_dispatch};
pub use copy::{Memcpy, cuda_memcpy_dispatch};
pub use pad::Pad;
pub use permute_axes::{PermuteAxes, cuda_permute_dispatch};
pub use rotate_half::RotateHalf;

pub fn all_functions() -> Vec<String> {
    use std::collections::HashSet;
    let mut functions = HashSet::<String>::new();

    functions.extend(
        tract_gpu::tensor::DeviceTensor::SUPPORTED_DT
            .into_iter()
            .flat_map(|dt| crate::kernels::BroadcastKind::ALL.into_iter().map(move |b| (dt, b)))
            .flat_map(|(dt, b)| MultiBroadcast.kernel_name(dt, b).into_iter()),
    );

    functions.extend(
        tract_gpu::tensor::DeviceTensor::SUPPORTED_DT
            .into_iter()
            .flat_map(|dt1| {
                tract_gpu::tensor::DeviceTensor::SUPPORTED_DT.into_iter().map(move |dt2| (dt1, dt2))
            })
            .flat_map(|(dt1, dt2)| Cast.kernel_name(dt1, dt2).into_iter()),
    );

    functions.extend(
        tract_gpu::tensor::DeviceTensor::SUPPORTED_DT.into_iter().flat_map(Pad::kernel_name),
    );
    functions.into_iter().collect()
}
