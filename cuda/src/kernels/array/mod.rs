mod cast;
mod copy;
mod dispatch;
mod rotate_half;

pub use cast::Cast;
pub use copy::Memcpy;
pub use dispatch::cuda_copy_nd_dispatch;
pub use rotate_half::RotateHalf;

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

    functions.into_iter().collect()
}
