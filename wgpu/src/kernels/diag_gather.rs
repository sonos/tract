use tract_core::internal::*;
use tract_gpu::ops::diag_gather::GpuDiagGather;
use tract_gpu::tensor::DeviceTensor;
use tract_transformers::ops::diag_gather::DiagGather;

use crate::kernels::shaders::{
    EntryPoint, GRID_LIMIT, LayoutKind, ShaderDtype, WORKGROUP, pack_u32s, preamble,
};
use crate::utils::{element_offset, get_wgpu_buffer};
use crate::{register_wgpu_op, with_wgpu_queue};

/// `output[…, i, k] = input[…, i, offset + k − i]`, zero outside the input row. The input's
/// leading axes must collapse into one batch axis; `output` is contiguous.
pub fn wgpu_diag_gather_dispatch(
    input: &DeviceTensor,
    offset: i64,
    out_len: usize,
    output: &DeviceTensor,
) -> TractResult<()> {
    with_wgpu_queue(|q| {
        q.retain_tensor(input);
        q.retain_tensor(output);
        let rank = input.rank();
        ensure!(rank >= 2 && output.rank() == rank);
        ensure!(output.datum_type() == input.datum_type());
        ensure!(output.shape()[..rank - 1] == input.shape()[..rank - 1]);
        ensure!(output.shape()[rank - 1] == out_len);
        let (shape, strides) = (input.shape(), input.strides());
        ensure!(strides.iter().all(|s| *s >= 0), "negative stride in {strides:?}");
        for axis in 0..rank.saturating_sub(3) {
            ensure!(
                shape[axis + 1] == 1
                    || strides[axis] == strides[axis + 1] * shape[axis + 1] as isize,
                "DiagGather input leading axes do not collapse: {shape:?} {strides:?}"
            );
        }
        let in_stride_b = if rank > 2 { strides[rank - 3] } else { 0 };
        let offset: i32 = offset.try_into().context("DiagGather offset overflows i32")?;
        let dt = ShaderDtype::from_datum(input.datum_type())?;
        let layout = LayoutKind::Chain(2);
        let pipeline = q.context().chain_pipeline(
            ("diag_gather", dt),
            layout,
            EntryPoint::typed("diag_gather", dt),
            || diag_gather_module(dt),
        )?;
        let bg = q.context().bind_group(
            layout,
            &[get_wgpu_buffer(input), get_wgpu_buffer(output)],
            q.uniform(),
        )?;
        let params = [
            element_offset(input, 0) as u32,
            element_offset(output, 0) as u32,
            output.len() as u32,
            out_len as u32,
            shape[rank - 2] as u32,
            shape[rank - 1] as u32,
            offset as u32,
            in_stride_b as u32,
            strides[rank - 2] as u32,
            strides[rank - 1] as u32,
        ];
        let dyn_off = q.alloc_uniform(&pack_u32s(&params))?;
        q.dispatch("diag_gather", &pipeline, &bg, dyn_off, output.len() as u64)
    })
}

register_wgpu_op!(DiagGather, |source, node, op| {
    let dt = source.node_input_facts(node.id)?[0].datum_type;
    rule_if!(matches!(dt, DatumType::F32 | DatumType::F16));
    Ok(Some(Box::new(GpuDiagGather::new(
        op.offset.clone(),
        op.out_len.clone(),
        "Wgpu",
        wgpu_diag_gather_dispatch,
    ))))
});

pub fn diag_gather_module(dt: ShaderDtype) -> String {
    let t = dt.wgsl();
    let suf = dt.suffix();
    let mut s = preamble(dt);
    s.push_str(&format!(
        r#"
struct Params {{
    off_in: u32,
    off_out: u32,
    len: u32,
    out_len: u32,
    t_q: u32,
    r_in: u32,
    offset: i32,
    stride_b: u32,
    stride_i: u32,
    stride_r: u32,
}}

@group(0) @binding(0) var<storage, read> inp: array<{t}>;
@group(0) @binding(1) var<storage, read_write> outp: array<{t}>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size({WORKGROUP})
fn diag_gather_{suf}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let o = gid.y * {GRID_LIMIT}u * {WORKGROUP}u + gid.x;
    if (o >= params.len) {{ return; }}
    let k = o % params.out_len;
    let row = o / params.out_len;
    let i = row % params.t_q;
    let b = row / params.t_q;
    let r = params.offset + i32(k) - i32(i);
    if (r >= 0 && r < i32(params.r_in)) {{
        outp[params.off_out + o] = inp[params.off_in + b * params.stride_b + i * params.stride_i + u32(r) * params.stride_r];
    }} else {{
        outp[params.off_out + o] = {t}(0);
    }}
}}
"#
    ));
    s
}
