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
    let grid_dim = match shape.len() {
        0 => panic!("Unexpected empty shape while build grid size"),
        1 => (1, 1, 1),
        2 => (shape[0] as _, 1, 1),
        3 => (shape[1] as _, shape[0] as _, 1),
        4.. => (
            shape[shape.len() - 2] as _,
            shape[shape.len() - 3] as _,
            (shape[..shape.len() - 3].iter().product::<usize>()) as _,
        ),
    };
    LaunchConfig {
        grid_dim,
        block_dim: (shape[shape.len() - 1].min(MAX_THREADS) as _, 1, 1),
        shared_mem_bytes: 0,
    }
}
