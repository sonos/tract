use tract_core::internal::*;
use tract_gpu::ops::causal_conv1d_update::GpuCausalConv1dUpdate;
use tract_gpu::tensor::DeviceTensor;
use tract_transformers::ops::causal_conv1d_update::CausalConv1dUpdate;

use crate::kernels::shaders::{EntryPoint, GRID_LIMIT, LayoutKind, WORKGROUP, pack_u32s};
use crate::utils::{element_offset, get_wgpu_buffer};
use crate::{register_wgpu_op, with_wgpu_queue};

const MAX_TAPS: usize = 8;

pub fn wgpu_causal_conv1d_update_dispatch(
    input: &DeviceTensor,
    weight: &DeviceTensor,
    state: &DeviceTensor,
    bias: Option<&DeviceTensor>,
    output: &DeviceTensor,
    final_state: &DeviceTensor,
) -> TractResult<()> {
    let tensors = [input, weight, state, output, final_state];
    ensure!(tensors.iter().chain(&bias).all(|t| t.datum_type() == DatumType::F16));
    let &[batch, channels, s_len] = input.shape() else { bail!("conv input must be [b, C, S]") };
    let taps = *weight.shape().last().context("conv weight must have a kernel axis")?;
    ensure!(taps <= MAX_TAPS, "the wgpu causal conv takes at most {MAX_TAPS} taps");
    ensure!(weight.len() == channels * taps);
    ensure!(state.shape() == [batch, channels, taps]);
    ensure!(output.shape() == input.shape() && final_state.shape() == state.shape());
    if let Some(bias) = bias {
        ensure!(bias.len() == channels);
    }
    let lanes = batch * channels;
    if lanes == 0 {
        return Ok(());
    }
    with_wgpu_queue(|q| {
        for t in tensors.iter().chain(&bias) {
            q.retain_tensor(t);
        }
        let layout = LayoutKind::TailWrites { n: 6, writes: 2 };
        let pipeline = q.context().chain_pipeline(
            "causal_conv1d_update",
            layout,
            EntryPoint::plain("causal_conv1d_update"),
            causal_conv1d_update_module,
        )?;
        let bias_or_dummy = bias.unwrap_or(weight);
        let buffers =
            [input, weight, state, bias_or_dummy, output, final_state].map(get_wgpu_buffer);
        let bg = q.context().bind_group(layout, &buffers, q.uniform())?;
        let mut params: Vec<u32> = [input, weight, state, bias_or_dummy, output, final_state]
            .iter()
            .map(|t| element_offset(t, 0) as u32)
            .collect();
        params.extend([channels, taps, s_len, lanes, bias.is_some() as usize].map(|d| d as u32));
        params.push(0);
        let dyn_off = q.alloc_uniform(&pack_u32s(&params))?;
        q.dispatch("causal_conv1d_update", &pipeline, &bg, dyn_off, lanes as u64)
    })
}

register_wgpu_op!(CausalConv1dUpdate, |source, node, _op| {
    let facts = source.node_input_facts(node.id)?;
    rule_if!(facts.iter().all(|f| f.datum_type == DatumType::F16));
    rule_if!(facts[0].rank() == 3);
    rule_if_some!(taps = facts[1].shape.last().and_then(|d| d.as_i64()));
    rule_if!(taps as usize <= MAX_TAPS);
    Ok(Some(Box::new(GpuCausalConv1dUpdate {
        backend_name: "Wgpu",
        dispatch: wgpu_causal_conv1d_update_dispatch,
    })))
});

/// One thread per (batch, channel): the window over `concat(state, input)` slides along
/// S in registers, each output is SiLU of the bias plus the window dotted with the weights.
fn causal_conv1d_update_module() -> String {
    format!(
        r#"enable f16;

struct Params {{
    off_input: u32,
    off_weight: u32,
    off_state: u32,
    off_bias: u32,
    off_out: u32,
    off_final: u32,
    channels: u32,
    taps: u32,
    s_len: u32,
    lanes: u32,
    has_bias: u32,
    _pad: u32,
}}

@group(0) @binding(0) var<storage, read> input: array<f16>;
@group(0) @binding(1) var<storage, read> weight: array<f16>;
@group(0) @binding(2) var<storage, read> initial_state: array<f16>;
@group(0) @binding(3) var<storage, read> bias: array<f16>;
@group(0) @binding(4) var<storage, read_write> outp: array<f16>;
@group(0) @binding(5) var<storage, read_write> final_state: array<f16>;
@group(0) @binding(6) var<uniform> params: Params;

@compute @workgroup_size({WORKGROUP})
fn causal_conv1d_update(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let lane = gid.y * {GRID_LIMIT}u * {WORKGROUP}u + gid.x;
    if (lane >= params.lanes) {{ return; }}
    let channel = lane % params.channels;
    let k = params.taps;
    let state_base = params.off_state + lane * k;
    let input_base = params.off_input + lane * params.s_len;
    let weight_base = params.off_weight + channel * k;
    var window: array<f32, {MAX_TAPS}>;
    for (var tap = 0u; tap < k; tap++) {{
        window[tap] = f32(initial_state[state_base + tap]);
    }}
    var b = 0.0;
    if (params.has_bias != 0u) {{ b = f32(bias[params.off_bias + channel]); }}
    for (var t = 0u; t < params.s_len; t++) {{
        for (var tap = 0u; tap + 1u < k; tap++) {{
            window[tap] = window[tap + 1u];
        }}
        window[k - 1u] = f32(input[input_base + t]);
        var sum = b;
        for (var tap = 0u; tap < k; tap++) {{
            sum += window[tap] * f32(weight[weight_base + tap]);
        }}
        outp[params.off_out + lane * params.s_len + t] = f16(sum / (1.0 + exp(-sum)));
    }}
    for (var tap = 0u; tap < k; tap++) {{
        final_state[params.off_final + lane * k + tap] = f16(window[tap]);
    }}
}}
"#
    )
}
