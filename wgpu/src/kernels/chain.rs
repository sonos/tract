//! Dispatch for a fused elementwise chain.

use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

use crate::kernels::shaders::{
    AT8, ChainStep, EntryPoint, GRID_LIMIT, LayoutKind, ShaderDtype, WORKGROUP,
    binary_ops_vec4_wgsl, binary_ops_wgsl, broadcast_strides, pack_u32s, pad8_u32, preamble,
    unary_ops_vec4_wgsl, unary_ops_wgsl,
};
use crate::utils::{element_offset, get_wgpu_buffer};
use crate::with_wgpu_queue;

/// Extra operands a chain can read besides its head, bounded by what the
/// 256-byte uniform slot holds.
pub const MAX_EXTRA_INPUTS: usize = 3;

pub fn wgpu_chain_dispatch(
    steps: &[ChainStep],
    inputs: &[&DeviceTensor],
    output: &DeviceTensor,
) -> TractResult<()> {
    with_wgpu_queue(|q| {
        ensure!(inputs.len() <= MAX_EXTRA_INPUTS + 1, "chain has too many inputs");
        for t in inputs {
            q.retain_tensor(t);
        }
        q.retain_tensor(output);
        let dt = ShaderDtype::from_datum(output.datum_type())?;
        let layout = LayoutKind::Chain(inputs.len() as u8 + 1);
        let out_shape = output.shape();
        let natural = Tensor::natural_strides(out_shape);
        let contiguous: Vec<bool> =
            inputs.iter().map(|t| t.shape() == out_shape && t.strides() == &natural[..]).collect();
        // Four values a thread: the kernel is bandwidth-bound, so a wider load
        // is the whole win. The four sit next to each other along the last
        // axis, so an operand that broadcasts over that axis is one value for
        // all of them and can be splatted; anything else still has to gather,
        // and then a wider thread buys nothing.
        let last = *out_shape.last().unwrap_or(&1);
        let kinds: Option<Vec<ChainOperand>> = (output.len().is_multiple_of(4)
            && last.is_multiple_of(4))
        .then(|| {
            inputs
                .iter()
                .zip(contiguous.iter())
                .map(|(t, c)| {
                    if *c {
                        Some(ChainOperand::Contiguous)
                    } else if t.rank() == out_shape.len()
                        && t.shape().last() == Some(&1)
                        && out_shape.last() != Some(&1)
                    {
                        Some(ChainOperand::Splat)
                    } else {
                        None
                    }
                })
                .collect::<Option<Vec<_>>>()
        })
        .flatten();
        let key = ("chain", dt, steps, &contiguous, kinds.as_deref());
        let pipeline =
            q.context().chain_pipeline(key, layout, EntryPoint::plain("chain"), || {
                chain_wgsl(dt, steps, &contiguous, kinds.as_deref())
            })?;

        let mut buffers: Vec<&crate::context::WgpuBuffer> =
            inputs.iter().map(|t| get_wgpu_buffer(t)).collect();
        buffers.push(get_wgpu_buffer(output));
        let bg = q.context().bind_group(layout, &buffers, q.uniform())?;

        let mut offsets = [0u32; 4];
        for (i, t) in inputs.iter().enumerate() {
            offsets[i] = element_offset(t, 0) as u32;
        }
        let mut vals =
            vec![element_offset(output, 0) as u32, output.len() as u32, out_shape.len() as u32, 0];
        vals.extend_from_slice(&offsets);
        vals.extend_from_slice(&pad8_u32(out_shape));
        for t in inputs {
            vals.extend_from_slice(&broadcast_strides(t.shape(), t.strides(), out_shape));
        }
        let dyn_off = q.alloc_uniform(&pack_u32s(&vals))?;
        let threads = if kinds.is_some() { output.len() / 4 } else { output.len() };
        q.dispatch("chain", &pipeline, &bg, dyn_off, threads as u64)
    })
}

/// Names the generated program, and so keys its pipeline. `contiguous` says
/// which inputs share the output's layout and can skip index arithmetic.
/// How a chain operand is read: element for element with the output, one value
/// splatted across the four a thread handles, or gathered through its strides.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChainOperand {
    Contiguous,
    Splat,
    Gather,
}

/// The chain over `vec4`s: a thread takes four values at a time, which is
/// where the win is on a bandwidth-bound kernel. The four are adjacent along
/// the last axis, so an operand broadcasting over that axis is read once and
/// splatted.
fn chain_vec4_wgsl(dt: ShaderDtype, steps: &[ChainStep], kinds: &[ChainOperand]) -> String {
    let t = dt.wgsl();
    let inputs = kinds.len();
    let mut s = preamble(dt);
    s.push_str("struct Params {\n    off_out: u32,\n    len: u32,\n    rank: u32,\n    _p: u32,\n    off_in: vec4<u32>,\n    out_sh0: vec4<u32>,\n    out_sh1: vec4<u32>,\n");
    for i in 0..inputs {
        s.push_str(&format!("    s{i}_0: vec4<u32>,\n    s{i}_1: vec4<u32>,\n"));
    }
    s.push_str("}\n\n");
    for (i, kind) in kinds.iter().enumerate() {
        let ty = match kind {
            ChainOperand::Contiguous => format!("array<vec4<{t}>>"),
            _ => format!("array<{t}>"),
        };
        s.push_str(&format!("@group(0) @binding({i}) var<storage, read> in{i}: {ty};\n"));
    }
    s.push_str(&format!(
        "@group(0) @binding({inputs}) var<storage, read_write> outp: array<vec4<{t}>>;\n"
    ));
    s.push_str(&format!("@group(0) @binding({}) var<uniform> params: Params;\n\n", inputs + 1));
    s.push_str(AT8);
    s.push_str("\n\n");
    s.push_str(&unary_ops_wgsl(t));
    s.push_str(&binary_ops_wgsl(t));
    s.push_str(&unary_ops_vec4_wgsl(t));
    s.push_str(&binary_ops_vec4_wgsl(t));
    for (i, kind) in kinds.iter().enumerate() {
        if *kind == ChainOperand::Contiguous {
            continue;
        }
        s.push_str(&format!(
            r#"
fn gather{i}(linear: u32) -> u32 {{
    var rest = linear;
    var idx = 0u;
    for (var k = 0u; k < params.rank; k++) {{
        let axis = params.rank - 1u - k;
        let dim = max(at8(params.out_sh0, params.out_sh1, axis), 1u);
        let c = rest % dim;
        rest = rest / dim;
        idx += c * at8(params.s{i}_0, params.s{i}_1, axis);
    }}
    return idx;
}}
"#
        ));
    }
    s.push_str(&format!(
        r#"
@compute @workgroup_size({WORKGROUP})
fn chain(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.y * {GRID_LIMIT}u * {WORKGROUP}u + gid.x;
    let base = i * 4u;
    if (base >= params.len) {{ return; }}
    var v = in0[params.off_in[0u] / 4u + i];
"#
    ));
    for step in steps {
        match step {
            ChainStep::Unary(op) => s.push_str(&format!("    v = op_{op}_v(v);\n")),
            ChainStep::Binary { op, rhs, swapped } => {
                match kinds[*rhs] {
                    ChainOperand::Contiguous => s.push_str(&format!(
                        "    let b{rhs} = in{rhs}[params.off_in[{rhs}u] / 4u + i];\n"
                    )),
                    _ => s.push_str(&format!(
                        "    let b{rhs} = vec4<{t}>(in{rhs}[params.off_in[{rhs}u] + gather{rhs}(base)]);\n"
                    )),
                }
                if *swapped {
                    s.push_str(&format!("    v = op_{op}_v(b{rhs}, v);\n"));
                } else {
                    s.push_str(&format!("    v = op_{op}_v(v, b{rhs});\n"));
                }
            }
        }
    }
    s.push_str("    outp[params.off_out / 4u + i] = v;\n}\n");
    s
}

/// A whole elementwise chain as one kernel: the running value stays in
/// registers, so only the chain's own inputs and its final output touch memory.
/// Extra operands broadcast against the output shape; the head and the output
/// share it.
pub fn chain_wgsl(
    dt: ShaderDtype,
    steps: &[ChainStep],
    contiguous: &[bool],
    kinds: Option<&[ChainOperand]>,
) -> String {
    if let Some(kinds) = kinds {
        return chain_vec4_wgsl(dt, steps, kinds);
    }
    let t = dt.wgsl();
    let inputs = contiguous.len();
    let mut s = preamble(dt);
    s.push_str("struct Params {\n    off_out: u32,\n    len: u32,\n    rank: u32,\n    _p: u32,\n    off_in: vec4<u32>,\n    out_sh0: vec4<u32>,\n    out_sh1: vec4<u32>,\n");
    for i in 0..inputs {
        s.push_str(&format!("    s{i}_0: vec4<u32>,\n    s{i}_1: vec4<u32>,\n"));
    }
    s.push_str("}\n\n");
    for i in 0..inputs {
        s.push_str(&format!("@group(0) @binding({i}) var<storage, read> in{i}: array<{t}>;\n"));
    }
    s.push_str(&format!(
        "@group(0) @binding({inputs}) var<storage, read_write> outp: array<{t}>;\n"
    ));
    s.push_str(&format!("@group(0) @binding({}) var<uniform> params: Params;\n\n", inputs + 1));
    s.push_str(AT8);
    s.push_str("\n\n");
    s.push_str(&unary_ops_wgsl(t));
    s.push_str(&binary_ops_wgsl(t));
    for (i, contig) in contiguous.iter().enumerate() {
        if *contig {
            s.push_str(&format!("\nfn gather{i}(linear: u32) -> u32 {{ return linear; }}\n"));
            continue;
        }
        s.push_str(&format!(
            r#"
fn gather{i}(linear: u32) -> u32 {{
    var rest = linear;
    var idx = 0u;
    for (var k = 0u; k < params.rank; k++) {{
        let axis = params.rank - 1u - k;
        let dim = max(at8(params.out_sh0, params.out_sh1, axis), 1u);
        let c = rest % dim;
        rest = rest / dim;
        idx += c * at8(params.s{i}_0, params.s{i}_1, axis);
    }}
    return idx;
}}
"#
        ));
    }
    s.push_str(&format!(
        r#"
@compute @workgroup_size({WORKGROUP})
fn chain(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.y * {GRID_LIMIT}u * {WORKGROUP}u + gid.x;
    if (i >= params.len) {{ return; }}
    var v = in0[params.off_in[0] + gather0(i)];
"#
    ));
    for step in steps {
        match step {
            ChainStep::Unary(op) => s.push_str(&format!("    v = op_{op}(v);\n")),
            ChainStep::Binary { op, rhs, swapped } => {
                s.push_str(&format!(
                    "    let b{rhs} = in{rhs}[params.off_in[{rhs}] + gather{rhs}(i)];\n"
                ));
                if *swapped {
                    s.push_str(&format!("    v = op_{op}(b{rhs}, v);\n"));
                } else {
                    s.push_str(&format!("    v = op_{op}(v, b{rhs});\n"));
                }
            }
        }
    }
    s.push_str("    outp[params.off_out + i] = v;\n}\n");
    s
}
