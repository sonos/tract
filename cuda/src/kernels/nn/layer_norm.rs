use crate::context::cuda_context;
use crate::kernels::launch_args::TractLaunchArgs;
use crate::kernels::{LibraryName, MAX_THREADS, WARP_SIZE, get_cuda_view, utils};
use cudarc::driver::{LaunchConfig, PushKernelArg};
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

const ROWS_PER_BLOCK: usize = 4;
const MAX_VECTORS_PER_LANE: usize = 8;

pub fn is_supported_dt(dt: DatumType) -> bool {
    matches!(dt, DatumType::F32 | DatumType::F16)
}

pub fn kernel_names() -> Vec<String> {
    let mut names = vec![];
    for t in ["f32", "f16"] {
        for affine in ["", "_affine"] {
            for nv in 1..=MAX_VECTORS_PER_LANE {
                names.push(format!("layer_norm_rows{nv}{affine}_{t}"));
            }
            names.push(format!("layer_norm{affine}_small_{t}"));
            names.push(format!("layer_norm{affine}_{t}"));
        }
    }
    names
}

/// Vectors per lane for the warp-per-row kernel, when the rows are contiguous, short enough to
/// sit in registers, and every row starts on a 16-byte boundary.
fn vectors_per_lane(
    input: &DeviceTensor,
    output: &DeviceTensor,
    shape_nd3: &[usize],
) -> Option<usize> {
    let elem = input.datum_type().size_of();
    let vec = 16 / elem;
    let cols = shape_nd3[1];
    let aligned = |t: &DeviceTensor| t.buffer_offset::<usize>().is_multiple_of(16);
    let nv = cols.div_ceil(WARP_SIZE * vec);
    (shape_nd3[2] == 1
        && cols.is_multiple_of(vec)
        && nv <= MAX_VECTORS_PER_LANE
        && aligned(input)
        && aligned(output))
    .then_some(nv)
}

pub fn cuda_layer_norm_dispatch(
    input: &DeviceTensor,
    affine: Option<(&DeviceTensor, &DeviceTensor)>,
    axis: usize,
    eps: f32,
    output: &DeviceTensor,
) -> TractResult<()> {
    ensure!(is_supported_dt(input.datum_type()));
    ensure!(output.shape() == input.shape() && output.datum_type() == input.datum_type());
    if let Some((scale, bias)) = affine {
        for t in [scale, bias] {
            ensure!(t.datum_type() == DatumType::F32 && t.len() == input.shape()[axis]);
        }
    }
    crate::with_cuda_stream(|stream| {
        let tname = DeviceTensor::tname(input.datum_type())?;
        let aname = if affine.is_some() { "_affine" } else { "" };
        let shape_nd3 = utils::reshape_to_rank_3(input.shape(), axis);
        let i_view = get_cuda_view(input);
        let o_view = get_cuda_view(output);
        let s_view = affine.map(|(s, _)| get_cuda_view(s)).unwrap_or_else(|| get_cuda_view(output));
        let b_view = affine.map(|(_, b)| get_cuda_view(b)).unwrap_or_else(|| get_cuda_view(output));
        if let Some(nv) = vectors_per_lane(input, output, &shape_nd3) {
            let func = cuda_context()
                .load_pipeline(LibraryName::NN, format!("layer_norm_rows{nv}{aname}_{tname}"))?;
            let mut args = TractLaunchArgs::new(stream, &func);
            args.push_view(&i_view);
            args.push_view(&s_view);
            args.push_view(&b_view);
            args.push_view(&o_view);
            args.push_i32(shape_nd3[0]);
            args.push_i32(shape_nd3[1]);
            args.push::<f32>(eps);
            args.launch(LaunchConfig {
                grid_dim: (shape_nd3[0].div_ceil(ROWS_PER_BLOCK) as _, 1, 1),
                block_dim: ((ROWS_PER_BLOCK * WARP_SIZE) as _, 1, 1),
                shared_mem_bytes: 0,
            })
        } else {
            let small = shape_nd3[1] < MAX_THREADS;
            let bname = if small { "small_" } else { "" };
            let func = cuda_context()
                .load_pipeline(LibraryName::NN, format!("layer_norm{aname}_{bname}{tname}"))?;
            let mut args = TractLaunchArgs::new(stream, &func);
            args.push_view(&i_view);
            args.push_view(&s_view);
            args.push_view(&b_view);
            args.push_view(&o_view);
            args.push_slice_i32(&shape_nd3);
            args.push_slice_i32(&Tensor::natural_strides(&shape_nd3));
            args.push::<f32>(eps);
            args.launch(LaunchConfig {
                grid_dim: ((shape_nd3[0] * shape_nd3[2]) as _, 1, 1),
                block_dim: (if small { WARP_SIZE } else { MAX_THREADS } as _, 1, 1),
                shared_mem_bytes: 0,
            })
        }
    })
}

crate::register_cuda_op!(tract_core::ops::nn::LayerNorm, |source, node, op| {
    let input_facts = source.node_input_facts(node.id)?;
    rule_if!(is_supported_dt(input_facts[0].datum_type));
    rule_if!(input_facts[1..].iter().all(|f| f.datum_type == DatumType::F32));
    Ok(Some(Box::new(tract_gpu::ops::layer_norm::GpuLayerNorm {
        axis: op.axis,
        eps: op.eps.clone(),
        backend_name: "Cuda",
        dispatch: cuda_layer_norm_dispatch,
    })))
});
