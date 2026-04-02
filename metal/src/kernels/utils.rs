use metal::{ComputePipelineState, MTLSize};
use tract_core::internal::TractResult;

pub fn build_metal_size_for_shape(shape: &[usize]) -> MTLSize {
    match shape.len() {
        0 => panic!("Unexpected empty shape while build grid size"),
        1 => MTLSize { width: shape[0] as _, height: 1, depth: 1 },
        2 => MTLSize { width: shape[1] as _, height: shape[0] as _, depth: 1 },
        3.. => MTLSize {
            width: shape[shape.len() - 1] as _,
            height: shape[shape.len() - 2] as _,
            depth: (shape[..shape.len() - 2].iter().product::<usize>()) as _,
        },
    }
}

pub fn build_metal_grid_and_groups_for_el_wise_op(
    shape: &[usize],
    max_thread: usize,
) -> (MTLSize, MTLSize) {
    let grid_size = match shape.len() {
        0 => panic!("Unexpected empty shape while build grid size"),
        1 => MTLSize { width: 1, height: 1, depth: 1 },
        2 => MTLSize { width: shape[0] as _, height: 1, depth: 1 },
        3 => MTLSize { width: shape[1] as _, height: shape[0] as _, depth: 1 },
        4.. => MTLSize {
            width: shape[shape.len() - 2] as _,
            height: shape[shape.len() - 3] as _,
            depth: (shape[..shape.len() - 3].iter().product::<usize>()) as _,
        },
    };

    (grid_size, MTLSize { width: shape[shape.len() - 1].min(max_thread) as _, height: 1, depth: 1 })
}

pub fn build_metal_size_with_ones() -> MTLSize {
    MTLSize { width: 1, height: 1, depth: 1 }
}

pub use tract_gpu::utils::{compute_broadcast_strides, reshape_to_rank_2, reshape_to_rank_3};
