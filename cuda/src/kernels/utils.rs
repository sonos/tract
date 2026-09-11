use cudarc::driver::LaunchConfig;

/// Threads per block for a copy. A block of the 1024 a launch may ask for
/// leaves a third of an SM's thread slots unused -- they come in 1536 -- and a
/// copy is bandwidth-bound, so it wants every slot it can fill.
const COPY_THREADS: usize = 256;

pub use tract_gpu::utils::{compute_broadcast_strides, reshape_to_rank_2, reshape_to_rank_3};

/// The grid a `copy_ndN` kernel runs on: a block covers whole rows of the
/// innermost axis, `x` the rows a single block does not hold, and `y` every
/// axis beyond the innermost two. The rows take `x` because a tensor has far
/// more of them, and `y` and `z` hold 65535 where `x` holds 2^31.
///
/// `block_dim` stays a whole multiple of the innermost extent -- the kernel
/// maps a thread to a row by dividing by it -- and only a row wider than a
/// block breaks that, in which case one block takes one row and strides
/// through it.
pub fn cuda_launch_cfg_for_cpy(shape: &[usize]) -> LaunchConfig {
    let rank = shape.len();
    assert!((1..=6).contains(&rank), "Unsupported rank {rank} for cuda copy launch config");
    if rank == 1 {
        let block = shape[0].clamp(1, COPY_THREADS);
        return LaunchConfig {
            grid_dim: (shape[0].div_ceil(block) as _, 1, 1),
            block_dim: (block as _, 1, 1),
            shared_mem_bytes: 0,
        };
    }
    let inner = shape[rank - 1];
    let prev = shape[rank - 2];
    let outer: usize = shape[..rank - 2].iter().product();
    let (block, rows) = if inner > COPY_THREADS {
        (COPY_THREADS, 0)
    } else {
        let rows = (COPY_THREADS / inner).min(prev);
        (inner * rows, rows)
    };
    LaunchConfig {
        grid_dim: (if rows == 0 { prev as _ } else { prev.div_ceil(rows) as _ }, outer as _, 1),
        block_dim: (block as _, 1, 1),
        shared_mem_bytes: 0,
    }
}
