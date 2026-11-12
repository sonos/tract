use crate::encoder::EncoderExt;
use crate::{LibraryName, MetalStream};
use metal::{MTLResourceUsage, MTLSize, NSUInteger};
use tract_core::internal::*;
use tract_gpu::ops::mamba2_recurrent::GpuMamba2Recurrent;
use tract_gpu::tensor::DeviceTensor;
use tract_transformers::ops::mamba2_recurrent::Mamba2Recurrent;

const MAX_STATE: usize = 256;
const ROWS_PER_GROUP: usize = 8;

fn is_supported_dt(dt: DatumType) -> bool {
    matches!(dt, DatumType::F32 | DatumType::F16)
}

pub fn kernel_name(x: DatumType, log_decay: DatumType, state: DatumType) -> TractResult<String> {
    ensure!([x, log_decay, state].into_iter().all(is_supported_dt));
    Ok(format!(
        "nn_ops::mamba2_recurrent_{}_{}_{}",
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
    stream: &MetalStream,
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
    ensure!(d_state <= MAX_STATE, "Metal Mamba2 takes a state of at most {MAX_STATE}");
    ensure!(log_decay.len() == batch * steps * heads);
    ensure!(initial_state.shape() == [batch, heads, head_dim, d_state]);
    let rows = batch * heads * head_dim;
    if rows == 0 {
        return Ok(());
    }
    let inputs = [x, b, c, log_decay, initial_state];
    let outputs = [output, final_state];
    for t in inputs.iter().chain(&outputs) {
        stream.retain_tensor(t);
    }
    let name = kernel_name(x.datum_type(), log_decay.datum_type(), initial_state.datum_type())?;
    let pipeline = stream.load_pipeline(LibraryName::NNOps, &name)?;
    let params = [batch, steps, heads, head_dim, groups, d_state].map(|d| d as i32);
    let command_buffer = stream.command_buffer();
    command_buffer.encode(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        for (ix, t) in inputs.iter().enumerate() {
            encoder.set_metal_tensor(ix as u64, t, MTLResourceUsage::Read);
        }
        encoder.set_metal_tensor(5, output, MTLResourceUsage::Write);
        encoder.set_metal_tensor(6, final_state, MTLResourceUsage::Write);
        encoder.set_slice(7, &params);
        let grid_size =
            MTLSize { width: rows.div_ceil(ROWS_PER_GROUP) as NSUInteger, height: 1, depth: 1 };
        let group_size =
            MTLSize { width: (ROWS_PER_GROUP * 32) as NSUInteger, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(grid_size, group_size);
    });
    Ok(())
}

pub fn metal_mamba2_recurrent_dispatch(
    x: &DeviceTensor,
    b: &DeviceTensor,
    c: &DeviceTensor,
    log_decay: &DeviceTensor,
    initial_state: &DeviceTensor,
    output: &DeviceTensor,
    final_state: &DeviceTensor,
) -> TractResult<()> {
    crate::with_metal_stream(|stream| {
        dispatch_eval(stream, x, b, c, log_decay, initial_state, output, final_state)
    })
}

crate::register_metal_op!(Mamba2Recurrent, |source, node, _op| {
    let facts = source.node_input_facts(node.id)?;
    rule_if!(facts.iter().all(|f| is_supported_dt(f.datum_type)));
    rule_if!(facts[1].datum_type == facts[0].datum_type);
    rule_if!(facts[2].datum_type == facts[0].datum_type);
    rule_if_some!(d_state = facts[1].shape.last().and_then(|d| d.as_i64()));
    rule_if!(d_state as usize <= MAX_STATE);
    Ok(Some(Box::new(GpuMamba2Recurrent {
        backend_name: "Metal",
        dispatch: metal_mamba2_recurrent_dispatch,
    })))
});
