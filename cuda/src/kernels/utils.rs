use cudarc::driver::LaunchConfig;
use num_traits::{AsPrimitive, Zero};
use tract_core::internal::*;
use tract_core::tract_linalg::block_quant::{BlockQuantFact, Q4_0};
use tract_gpu::tensor::DeviceTensor;

use crate::kernels::MAX_THREADS;
use crate::tensor::CudaTensor;

pub fn compute_broadcast_strides<T: Zero + Copy + 'static>(
    shape: &[usize],
    strides: &[isize],
) -> TractResult<TVec<T>>
where
    isize: AsPrimitive<T>,
{
    ensure!(
        shape.len() == strides.len(),
        "Mistmach between shape and strides length while computing broadcast strides"
    );
    Ok(strides
        .iter()
        .zip(shape)
        .map(|(s, dim)| if *dim == 1 { T::zero() } else { s.as_() })
        .collect::<TVec<T>>())
}

pub fn reshape_to_rank_2(shape: &[usize], axis: usize) -> TVec<usize> {
    let dim_axis_0 = shape[0..axis].iter().product::<usize>();
    let dim_axis_2 = shape[axis..].iter().product::<usize>();
    tvec![dim_axis_0, dim_axis_2]
}

pub fn reshape_to_rank_3(shape: &[usize], axis: usize) -> TVec<usize> {
    let dim_axis_0 = shape[0..axis].iter().product::<usize>();
    let dim_axis_1 = shape[axis];
    let dim_axis_2 = shape[axis + 1..].iter().product::<usize>();
    tvec![dim_axis_0, dim_axis_1, dim_axis_2]
}

pub fn cuda_launch_cfg_for_cpy(shape: &[usize]) -> LaunchConfig {
    // Grid layout: z=dim0, y=dim1, x=product(middle dims), threads=innermost
    // nd1: x=1, threads=d0
    // nd2: x=d0, threads=d1
    // nd3: x=d0*d1, threads=d2
    // nd4: z=d0, x=d1*d2, threads=d3
    // nd5: z=d0, y=d1, x=d2*d3, threads=d4
    // nd6: z=d0, y=d1, x=d2*d3*d4, threads=d5
    let rank = shape.len();
    let grid_dim = match rank {
        0 => panic!("Unexpected empty shape while build grid size"),
        1 => (1, 1, 1),
        2 => (shape[0] as _, 1, 1),
        3 => (shape[1] as _, shape[0] as _, 1),
        4 => (shape[2] as _, shape[1] as _, shape[0] as _),
        5 => (shape[2] as u32 * shape[3] as u32, shape[1] as _, shape[0] as _),
        6 => (shape[2] as u32 * shape[3] as u32 * shape[4] as u32, shape[1] as _, shape[0] as _),
        _ => panic!("Unsupported rank {rank} for cuda copy launch config"),
    };
    LaunchConfig {
        grid_dim,
        block_dim: (shape[rank - 1].min(MAX_THREADS) as _, 1, 1),
        shared_mem_bytes: 0,
    }
}
