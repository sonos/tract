use metal::{ComputeCommandEncoderRef, ComputePipelineState, MTLSize, NSUInteger};
use tract_core::internal::TractResult;

/// Dispatch a flat, one-thread-per-element kernel that indexes its work by
/// `thread_position_in_grid`, packing the `n` threads into full-width
/// threadgroups.
///
/// The element-wise / cast / copy kernels historically dispatched `n`
/// threadgroups of a *single* thread (`dispatch_thread_groups(grid = n,
/// group = 1)`), which leaves 31 of every 32 SIMD lanes idle on Apple GPUs
/// (each threadgroup owns its own SIMD-group). `dispatch_threads` lets Metal
/// pack the same `n` threads into threadgroups of up to the pipeline maximum
/// (non-uniform threadgroups cover the tail), with no kernel change since the
/// global `thread_position_in_grid` each thread sees is unchanged.
pub fn dispatch_threads_1d(
    encoder: &ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    n: usize,
) {
    let grid = MTLSize { width: n as NSUInteger, height: 1, depth: 1 };
    let group_width = n.clamp(1, pipeline.max_total_threads_per_threadgroup() as usize);
    let group = MTLSize { width: group_width as NSUInteger, height: 1, depth: 1 };
    encoder.dispatch_threads(grid, group);
}

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
