use crate::context::TractCudaStream;
use crate::context::cuda_context;
use crate::kernels::LibraryName;
use crate::kernels::get_cuda_view;
use crate::kernels::launch_args::TractLaunchArgs;
use cudarc::driver::LaunchConfig;
use tract_core::internal::*;
use tract_gpu::ops::mamba2_recurrent::GpuMamba2Recurrent;
use tract_gpu::tensor::DeviceTensor;
use tract_transformers::ops::mamba2_recurrent::Mamba2Recurrent;

const MAX_STATE: usize = 256;
const ROWS_PER_BLOCK: usize = 8;

fn is_supported_dt(dt: DatumType) -> bool {
    matches!(dt, DatumType::F32 | DatumType::F16)
}

pub fn kernel_name(x: DatumType, log_decay: DatumType, state: DatumType) -> TractResult<String> {
    ensure!([x, log_decay, state].into_iter().all(is_supported_dt));
    Ok(format!(
        "mamba2_recurrent_{}_{}_{}",
        DeviceTensor::tname(x)?,
        DeviceTensor::tname(log_decay)?,
        DeviceTensor::tname(state)?
    ))
}

pub fn kernel_names() -> Vec<String> {
    let dts = [DatumType::F32, DatumType::F16];
    let mut names = vec![];
    for x in dts {
        for d in dts {
            for s in dts {
                names.extend(kernel_name(x, d, s).ok());
            }
        }
    }
    names
}

#[allow(clippy::too_many_arguments)]
fn dispatch_eval(
    stream: &TractCudaStream,
    x: &DeviceTensor,
    b: &DeviceTensor,
    c: &DeviceTensor,
    log_decay: &DeviceTensor,
    initial_state: &DeviceTensor,
    output: &DeviceTensor,
    final_state: &DeviceTensor,
) -> TractResult<()> {
    let &[batch, steps, heads, head_dim] = x.shape() else { bail!("Mamba2 x must be rank 4") };
    let &[_, _, groups, d_state] = b.shape() else { bail!("Mamba2 b must be rank 4") };
    ensure!(b.datum_type() == x.datum_type() && c.datum_type() == x.datum_type());
    ensure!(b.shape() == c.shape() && groups > 0 && heads % groups == 0);
    ensure!(d_state <= MAX_STATE, "CUDA Mamba2 takes a state of at most {MAX_STATE}");
    ensure!(log_decay.len() == batch * steps * heads);
    ensure!(initial_state.shape() == [batch, heads, head_dim, d_state]);
    let rows = batch * heads * head_dim;
    if rows == 0 {
        return Ok(());
    }
    let name = kernel_name(x.datum_type(), log_decay.datum_type(), initial_state.datum_type())?;
    let func = cuda_context().load_pipeline(LibraryName::NN, name)?;
    let views = [x, b, c, log_decay, initial_state, output, final_state].map(get_cuda_view);
    let mut args = TractLaunchArgs::new(stream, &func);
    for view in &views {
        args.push_view(view);
    }
    for dim in [batch, steps, heads, head_dim, groups, d_state] {
        args.push_i32(dim);
    }
    args.launch(LaunchConfig {
        grid_dim: (rows.div_ceil(ROWS_PER_BLOCK) as _, 1, 1),
        block_dim: ((ROWS_PER_BLOCK * 32) as _, 1, 1),
        shared_mem_bytes: 0,
    })
}

pub fn cuda_mamba2_recurrent_dispatch(
    x: &DeviceTensor,
    b: &DeviceTensor,
    c: &DeviceTensor,
    log_decay: &DeviceTensor,
    initial_state: &DeviceTensor,
    output: &DeviceTensor,
    final_state: &DeviceTensor,
) -> TractResult<()> {
    crate::with_cuda_stream(|stream| {
        dispatch_eval(stream, x, b, c, log_decay, initial_state, output, final_state)
    })
}

crate::register_cuda_op!(Mamba2Recurrent, |source, node, _op| {
    let facts = source.node_input_facts(node.id)?;
    rule_if!(facts.iter().all(|f| is_supported_dt(f.datum_type)));
    rule_if!(facts[1].datum_type == facts[0].datum_type);
    rule_if!(facts[2].datum_type == facts[0].datum_type);
    rule_if_some!(d_state = facts[1].shape.last().and_then(|d| d.as_i64()));
    rule_if!(d_state as usize <= MAX_STATE);
    Ok(Some(Box::new(GpuMamba2Recurrent {
        backend_name: "Cuda",
        dispatch: cuda_mamba2_recurrent_dispatch,
    })))
});
