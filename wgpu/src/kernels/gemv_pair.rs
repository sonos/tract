//! Dispatch for two chained matrix-vector products.

use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

use crate::kernels::matmul::epilogue_mode;
use crate::kernels::shaders::{
    ChainStep, EntryPoint, LayoutKind, ShaderDtype, binary_ops_wgsl, epilogue_body, pack_u32s,
    preamble, unary_ops_wgsl,
};
use crate::utils::{element_offset, get_wgpu_buffer};
use crate::with_wgpu_queue;

/// The widest hidden vector the kernel will take: it is held in workgroup
/// memory, one slot per value.
pub const MAX_WIDTH: usize = GEMV_PAIR_WG as usize;

/// Operands the activation, the trailing steps and the input scale can carry
/// between them, bounded by the one `vec4` of offsets the uniform holds.
pub const MAX_ACT_OPERANDS: usize = 4;

/// `x` is one row of `k` values, `w1` is `k` by `r` and `w2` is `r` by `n`.
/// Strides are in elements, row first.
#[derive(Debug, Clone, Copy)]
pub struct GemvPairShape {
    pub k: usize,
    pub r: usize,
    pub n: usize,
    pub x_s: usize,
    pub w1: (usize, usize),
    pub w2: (usize, usize),
    pub out_s: usize,
}

/// `extras` holds the activation's operands first (`act_extras` of them) and
/// the trailing steps' after; `scale`, when given, multiplies `x` on load.
#[allow(clippy::too_many_arguments)]
pub fn wgpu_gemv_pair_dispatch(
    shape: GemvPairShape,
    act: &[ChainStep],
    act_extras: usize,
    post: &[ChainStep],
    x: &DeviceTensor,
    w1: &DeviceTensor,
    w2: &DeviceTensor,
    extras: &[&DeviceTensor],
    scale: Option<&DeviceTensor>,
    output: &DeviceTensor,
) -> TractResult<()> {
    with_wgpu_queue(|q| {
        for t in [x, w1, w2, output] {
            q.retain_tensor(t);
        }
        for t in extras.iter().copied().chain(scale) {
            q.retain_tensor(t);
        }
        let dt = ShaderDtype::from_datum(x.datum_type())?;
        let bound = extras.len() + scale.is_some() as usize;
        ensure!(bound <= MAX_ACT_OPERANDS, "gemv_pair carries at most {MAX_ACT_OPERANDS} operands");
        let layout = LayoutKind::Chain(4 + bound as u8);
        let key = ("gemv_pair", dt, bound, act, post, scale.is_some());
        let pipeline =
            q.context().chain_pipeline(key, layout, EntryPoint::typed("gemv_pair", dt), || {
                gemv_pair_module(dt, act, post, bound, scale.is_some())
            })?;
        let mut buffers = vec![get_wgpu_buffer(x), get_wgpu_buffer(w1), get_wgpu_buffer(w2)];
        buffers.extend(extras.iter().chain(scale.iter()).map(|t| get_wgpu_buffer(t)));
        buffers.push(get_wgpu_buffer(output));
        let bg = q.context().bind_group(layout, &buffers, q.uniform())?;
        let mut vals = vec![
            element_offset(x, 0) as u32,
            element_offset(w1, 0) as u32,
            element_offset(w2, 0) as u32,
            element_offset(output, 0) as u32,
            shape.k as u32,
            shape.r as u32,
            shape.n as u32,
            shape.x_s as u32,
            shape.w1.0 as u32,
            shape.w1.1 as u32,
            shape.w2.0 as u32,
            shape.w2.1 as u32,
            shape.out_s as u32,
            0,
            0,
            0,
        ];
        let mut off_extra = [0u32; 4];
        let mut mode_extra = [0u32; 4];
        for (i, t) in extras.iter().enumerate() {
            off_extra[i] = element_offset(t, 0) as u32;
            mode_extra[i] = epilogue_mode(t, if i < act_extras { shape.r } else { shape.n })?;
        }
        if let Some(t) = scale {
            ensure!(t.len() == 1, "gemv_pair's input scale must be a single value");
            off_extra[extras.len()] = element_offset(t, 0) as u32;
        }
        vals.extend_from_slice(&off_extra);
        vals.extend_from_slice(&mode_extra);
        let dyn_off = q.alloc_uniform(&pack_u32s(&vals))?;
        q.dispatch_grid("gemv_pair", &pipeline, &bg, dyn_off, [1, 1, 1])
    })
}

/// Steps of each of the gemv pair's reductions taken per iteration.
pub const GEMV_PAIR_UNROLL: usize = 10;

/// The fused squeeze-excitation gate runs as one workgroup of this width.
pub const GEMV_PAIR_WG: u32 = 256;

/// A single-channel tensor into a texture the caller can sample: the mask
/// leaves the graph as a GPU resource rather than as bytes on the host.
/// A matmul whose thread computes a four-by-four tile of the output.
///
/// Both operands are read as `vec4`: four rows of `a` are contiguous because the
/// activations are channels-first, and four columns of `b` are contiguous
/// because the weights were transposed once at build time. Eight `k` per pass,
/// with a four-step tail, keeps the loop overhead off what is otherwise a
/// two-load, sixteen-multiply step and gives each accumulator eight
/// independent products to overlap. Requires `m`, `n` and `k` all multiples
/// of four, a unit prefix, and both matmul axes unit-strided on their fastest
/// side.
/// Two matrix-vector products with an activation between them, as one kernel.
///
/// One workgroup: it computes the hidden vector cooperatively, holds it in
/// workgroup memory across a barrier, and reads it back for the second product.
/// Both vectors are narrow, so the whole gate fits in a single launch, which is
/// the point — three dispatches this small are all launch and no work. The
/// activation's own operands, a bias and a clamp, ride in `extras`.
/// `extras` counts every bound operand: the activation's, then `post`'s, then
/// the scale, which is the last one when `scaled`.
///
/// Both reductions are unrolled by [`GEMV_PAIR_UNROLL`] with a scalar tail, the
/// second into that many accumulators: the vectors are short enough that only a
/// few dozen of the workgroup's threads have a row at all, so what is left to
/// win is instruction-level parallelism inside each of them rather than more
/// threads.
pub fn gemv_pair_module(
    dt: ShaderDtype,
    act: &[ChainStep],
    post: &[ChainStep],
    extras: usize,
    scaled: bool,
) -> String {
    let t = dt.wgsl();
    let suf = dt.suffix();
    let out_binding = 3 + extras;
    let uniform_binding = 4 + extras;
    let extra_bindings = (0..extras)
        .map(|i| {
            format!("@group(0) @binding({}) var<storage, read> extra{i}: array<{t}>;\n", 3 + i)
        })
        .collect::<String>();
    let lib = format!("{}{}", unary_ops_wgsl(t), binary_ops_wgsl(t));
    let act_body = epilogue_body(act, "j");
    let post_body = epilogue_body(post, "c");
    let unroll = GEMV_PAIR_UNROLL;
    let unroll_last = unroll - 1;
    let first_unrolled: String = (0..unroll)
        .map(|i| format!("            sum += x_at(kk + {i}u) * x_scale * w1_at(kk + {i}u, j);\n"))
        .collect();
    let acc_decls: String = (0..unroll).map(|i| format!("        var acc{i} = 0.0;\n")).collect();
    let second_unrolled: String = (0..unroll)
        .map(|i| format!("            acc{i} += hidden[j + {i}u] * w2_at(j + {i}u, c);\n"))
        .collect();
    let acc_sum = (0..unroll).map(|i| format!("acc{i}")).collect::<Vec<_>>().join(" + ");
    let x_scale = if scaled {
        let i = extras - 1;
        format!("f32(extra{i}[params.off_extra[{i}u]])")
    } else {
        "1.0".to_string()
    };
    let mut s = preamble(dt);
    s.push_str(&format!(
        r#"
struct Params {{
    off_x: u32,
    off_w1: u32,
    off_w2: u32,
    off_out: u32,
    k: u32,
    r: u32,
    n: u32,
    x_s: u32,
    w1_row_s: u32,
    w1_col_s: u32,
    w2_row_s: u32,
    w2_col_s: u32,
    out_s: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
    off_extra: vec4<u32>,
    mode_extra: vec4<u32>,
}}

@group(0) @binding(0) var<storage, read> x: array<{t}>;
@group(0) @binding(1) var<storage, read> w1: array<{t}>;
@group(0) @binding(2) var<storage, read> w2: array<{t}>;
{extra_bindings}@group(0) @binding({out_binding}) var<storage, read_write> outp: array<{t}>;
@group(0) @binding({uniform_binding}) var<uniform> params: Params;
{lib}

fn x_at(kk: u32) -> f32 {{ return f32(x[params.off_x + kk * params.x_s]); }}
fn w1_at(kk: u32, j: u32) -> f32 {{
    return f32(w1[params.off_w1 + kk * params.w1_row_s + j * params.w1_col_s]);
}}
fn w2_at(j: u32, c: u32) -> f32 {{
    return f32(w2[params.off_w2 + j * params.w2_row_s + c * params.w2_col_s]);
}}

var<workgroup> hidden: array<f32, {GEMV_PAIR_WG}>;

@compute @workgroup_size({GEMV_PAIR_WG})
fn gemv_pair_{suf}(@builtin(local_invocation_id) lid: vec3<u32>) {{
    let tid = lid.x;
    let x_scale = {x_scale};
    for (var j = tid; j < params.r; j += {GEMV_PAIR_WG}u) {{
        var sum = 0.0;
        var kk = 0u;
        while (kk + {unroll_last}u < params.k) {{
{first_unrolled}            kk += {unroll}u;
        }}
        for (; kk < params.k; kk++) {{
            sum += x_at(kk) * x_scale * w1_at(kk, j);
        }}
        var v = {t}(sum);
{act_body}
        hidden[j] = f32(v);
    }}
    workgroupBarrier();
    for (var c = tid; c < params.n; c += {GEMV_PAIR_WG}u) {{
{acc_decls}        var j = 0u;
        while (j + {unroll_last}u < params.r) {{
{second_unrolled}            j += {unroll}u;
        }}
        for (; j < params.r; j++) {{
            acc0 += hidden[j] * w2_at(j, c);
        }}
        var v = {t}({acc_sum});
{post_body}
        outp[params.off_out + c * params.out_s] = v;
    }}
}}
"#
    ));
    s
}
