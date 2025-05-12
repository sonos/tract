use metal::{ComputePipelineState, MTLSize};
use num_traits::{AsPrimitive, Zero};
use tract_core::internal::{ensure, tvec, TVec, TractResult};

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
