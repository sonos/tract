mod broadcast;
mod cast;
mod concat;
mod copy;
mod permute_axes;

pub use broadcast::MultiBroadcast;
pub use cast::Cast;
pub use concat::Concat;
pub use copy::Memcpy;
pub use permute_axes::PermuteAxes;

pub fn all_functions() -> Vec<String> {
    use std::collections::HashSet;
    let mut functions = HashSet::<String>::new();

    functions.extend(
        crate::MetalTensor::SUPPORTED_DT
            .into_iter()
            .map(move |dt| dt)
            .flat_map(|dt| crate::kernels::BroadcastKind::ALL.into_iter().map(move |b| (dt, b)))
            .flat_map(|(dt, b)| MultiBroadcast.kernel_name(dt, b).into_iter()),
    );

    functions.extend(
        crate::MetalTensor::SUPPORTED_DT
            .into_iter()
            .map(move |dt| dt)
            .flat_map(|dt| Memcpy.kernel_name(dt).into_iter()),
    );

    functions.extend(
        crate::MetalTensor::SUPPORTED_DT
            .into_iter()
            .map(move |dt| dt)
            .flat_map(|dt1| crate::MetalTensor::SUPPORTED_DT.into_iter().map(move |dt2| (dt1, dt2)))
            .flat_map(|(dt1, dt2)| Cast.kernel_name(dt1, dt2).into_iter()),
    );

    functions.into_iter().collect()
}
