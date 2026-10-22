use cudarc::driver::LaunchConfig;

use tract_gpu::utils::COPY_THREADS;
use tract_gpu::utils::copy_launch;
pub use tract_gpu::utils::{compute_broadcast_strides, reshape_to_rank_2, reshape_to_rank_3};

/// [`tract_gpu::utils::copy_launch`] at [`COPY_THREADS`] threads a block.
pub fn cuda_launch_cfg_for_cpy(shape: &[usize]) -> LaunchConfig {
    let launch = copy_launch(shape, COPY_THREADS);
    LaunchConfig {
        grid_dim: (launch.groups[0] as _, launch.groups[1] as _, launch.groups[2] as _),
        block_dim: (launch.threads as _, 1, 1),
        shared_mem_bytes: 0,
    }
}
