use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;
use tract_transformers::ops::sdpa::Sdpa;

use crate::kernels::shaders::{EntryPoint, LayoutKind, ShaderDtype, pack_u32s, preamble};
use crate::utils::{element_offset, get_wgpu_buffer};
use crate::{register_wgpu_op, with_wgpu_queue};

/// Whether the flash kernel takes this `Sdpa`: unmasked and not causal, f16 or
/// f32 throughout, rank 3 or 4 Q, K and V with the same leading dims, and
/// concrete head dims within [`FLASH_SDPA_MAX_HEAD_DIM`].
pub fn flash_sdpa_supported(facts: &[&TypedFact], op: &Sdpa) -> bool {
    if facts.len() != 3 || op.is_causal {
        return false;
    }
    let [qf, kf, vf] = [facts[0], facts[1], facts[2]];
    let rank = qf.rank();
    if !matches!(rank, 3 | 4)
        || kf.rank() != rank
        || vf.rank() != rank
        || !matches!(qf.datum_type, DatumType::F16 | DatumType::F32)
        || kf.datum_type != qf.datum_type
        || vf.datum_type != qf.datum_type
        || qf.shape[..rank - 2] != kf.shape[..rank - 2]
        || kf.shape[..rank - 1] != vf.shape[..rank - 1]
    {
        return false;
    }
    let dim = |f: &TypedFact| f.shape[rank - 1].to_usize().ok();
    match (dim(qf), dim(kf), dim(vf)) {
        (Some(d), Some(dk), Some(dv)) => {
            d == dk && d <= FLASH_SDPA_MAX_HEAD_DIM && dv <= FLASH_SDPA_MAX_HEAD_DIM
        }
        _ => false,
    }
}

/// (batch, head, position, channel) strides of a rank 3 or 4 attention operand.
fn bhsd_strides(t: &DeviceTensor) -> [u32; 4] {
    let s = t.strides();
    if s.len() == 3 {
        [s[0] as u32, 0, s[1] as u32, s[2] as u32]
    } else {
        [s[0] as u32, s[1] as u32, s[2] as u32, s[3] as u32]
    }
}

pub fn wgpu_flash_sdpa_dispatch(
    scale: f32,
    q: &DeviceTensor,
    k: &DeviceTensor,
    v: &DeviceTensor,
    output: &DeviceTensor,
) -> TractResult<()> {
    with_wgpu_queue(|queue| {
        for t in [q, k, v, output] {
            queue.retain_tensor(t);
        }
        let dt = ShaderDtype::from_datum(q.datum_type())?;
        let rank = q.rank();
        let (sq, d) = (q.shape()[rank - 2], q.shape()[rank - 1]);
        let (sk, dv) = (k.shape()[rank - 2], v.shape()[rank - 1]);
        let (batch, heads) =
            if rank == 4 { (q.shape()[0], q.shape()[1]) } else { (q.shape()[0], 1) };
        let layout = LayoutKind::Chain(4);
        let pipeline = queue.context().chain_pipeline(
            ("flash_sdpa", dt, d, dv),
            layout,
            EntryPoint::typed("flash_sdpa", dt),
            || flash_sdpa_module(dt, d, dv),
        )?;
        let buffers =
            [get_wgpu_buffer(q), get_wgpu_buffer(k), get_wgpu_buffer(v), get_wgpu_buffer(output)];
        let bg = queue.context().bind_group(layout, &buffers, queue.uniform())?;
        let mut vals = vec![
            element_offset(q, 0) as u32,
            element_offset(k, 0) as u32,
            element_offset(v, 0) as u32,
            element_offset(output, 0) as u32,
            sq as u32,
            sk as u32,
            scale.to_bits(),
            0,
        ];
        for t in [q, k, v, output] {
            vals.extend_from_slice(&bhsd_strides(t));
        }
        let dyn_off = queue.alloc_uniform(&pack_u32s(&vals))?;
        let grid = [sq.div_ceil(FLASH_SDPA_ROWS) as u32, heads as u32, batch as u32];
        queue.dispatch_grid("flash_sdpa", &pipeline, &bg, dyn_off, grid)
    })
}

register_wgpu_op!(Sdpa, |source, node, op| {
    rule_if!(flash_sdpa_supported(&source.node_input_facts(node.id)?, op));
    let d = source.node_input_facts(node.id)?[0].shape.last().unwrap().to_usize()?;
    let scale = match &op.scale {
        Some(s) => s.cast_to_scalar::<f32>()?,
        None => 1.0 / (d as f32).sqrt(),
    };
    Ok(Some(Box::new(crate::ops::sdpa::WgpuFlashSdpa { scale })))
});

/// Query rows of a flash-attention workgroup.
pub const FLASH_SDPA_ROWS: usize = 32;
/// Keys a flash-attention workgroup stages per step.
pub const FLASH_SDPA_KEYS: usize = 64;
/// Largest head dim, for queries, keys and values alike, the flash-attention
/// kernel takes: its tiles then fit WebGPU's default 16 KiB of workgroup memory.
pub const FLASH_SDPA_MAX_HEAD_DIM: usize = 48;

/// Reads four consecutive channels of one position of `src` from `base`, zero
/// past `len`, with every lane named rather than indexed.
fn flash_sdpa_loader(t: &str, src: &str, len: usize) -> String {
    let lanes: Vec<String> = (0..4)
        .map(|j| format!("select({t}(0.0), {src}[base + min(c4 * 4u + {j}u, {last}u) * stride], c4 * 4u + {j}u < {len}u)", last = len - 1))
        .collect();
    format!(
        "fn quad_{src}(base: u32, c4: u32, stride: u32) -> vec4<{t}> {{\n    return vec4<{t}>({});\n}}\n",
        lanes.join(", ")
    )
}

/// Unmasked attention, softmax(scale Q Kᵀ) V, without the score matrix: one
/// workgroup per [`FLASH_SDPA_ROWS`] queries of one (batch, head) walks the
/// keys [`FLASH_SDPA_KEYS`] at a time, keeping a running maximum and sum per
/// query and rescaling its output accumulators as the maximum grows. Each of the
/// 128 threads holds a four-by-four block of the scores and four rows by four
/// value columns of the output, all in f32; the query tile, the key or value
/// block and the probabilities are staged in workgroup memory in the tensor
/// type. `d` and `dv` are the query/key and value head dims, at most
/// [`FLASH_SDPA_MAX_HEAD_DIM`]. Q, K, V and the output are read through
/// (batch, head, position, channel) strides. Workgroup `x` tiles the queries,
/// `y` is the head and `z` the batch.
pub fn flash_sdpa_module(dt: ShaderDtype, d: usize, dv: usize) -> String {
    let t = dt.wgsl();
    let suf = dt.suffix();
    let rows = FLASH_SDPA_ROWS;
    let keys = FLASH_SDPA_KEYS;
    let d4 = d.div_ceil(4);
    let dv4 = dv.div_ceil(4);
    let kv_len = keys * d4.max(dv4);
    let mut s = preamble(dt);
    s.push_str(&format!(
        r#"
struct Params {{
    off_q: u32,
    off_k: u32,
    off_v: u32,
    off_o: u32,
    sq: u32,
    sk: u32,
    scale: f32,
    _p: u32,
    q_s: vec4<u32>,
    k_s: vec4<u32>,
    v_s: vec4<u32>,
    o_s: vec4<u32>,
}}

@group(0) @binding(0) var<storage, read> q: array<{t}>;
@group(0) @binding(1) var<storage, read> k: array<{t}>;
@group(0) @binding(2) var<storage, read> v: array<{t}>;
@group(0) @binding(3) var<storage, read_write> outp: array<{t}>;
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> q_tile: array<vec4<{t}>, {q_len}>;
var<workgroup> kv_tile: array<vec4<{t}>, {kv_len}>;
var<workgroup> p_tile: array<vec4<{t}>, {p_len}>;
var<workgroup> red: array<f32, {red_len}>;

{loaders}

@compute @workgroup_size(128)
fn flash_sdpa_{suf}(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>) {{
    let row0 = wid.x * {rows}u;
    let h = wid.y;
    let b = wid.z;
    let tx = lid % 16u;
    let ty = lid / 16u;
    let q_base = params.off_q + b * params.q_s.x + h * params.q_s.y;
    let k_base = params.off_k + b * params.k_s.x + h * params.k_s.y;
    let v_base = params.off_v + b * params.v_s.x + h * params.v_s.y;
    let o_base = params.off_o + b * params.o_s.x + h * params.o_s.y;

    for (var i = lid; i < {q_len}u; i += 128u) {{
        let r = i / {d4}u;
        var val = vec4<{t}>(0.0);
        if (row0 + r < params.sq) {{
            val = quad_q(q_base + (row0 + r) * params.q_s.z, i % {d4}u, params.q_s.w);
        }}
        q_tile[i] = val;
    }}

    var m = array<f32, 4>(-3.0e38, -3.0e38, -3.0e38, -3.0e38);
    var l = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
    var o = array<vec4<f32>, 4>(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));

    for (var kb = 0u; kb < params.sk; kb += {keys}u) {{
        for (var i = lid; i < {kd_len}u; i += 128u) {{
            let c = i / {d4}u;
            var val = vec4<{t}>(0.0);
            if (kb + c < params.sk) {{
                val = quad_k(k_base + (kb + c) * params.k_s.z, i % {d4}u, params.k_s.w);
            }}
            kv_tile[i] = val;
        }}
        workgroupBarrier();
"#,
        q_len = rows * d4,
        p_len = rows * keys / 4,
        loaders = [("q", d), ("k", d), ("v", dv)]
            .iter()
            .map(|(src, len)| flash_sdpa_loader(t, src, *len))
            .collect::<String>(),
        red_len = rows * 16,
        kd_len = keys * d4,
    ));
    for r in 0..4 {
        for c in 0..4 {
            s += &format!("        var s{r}_{c} = 0.0;\n");
        }
    }
    s += &format!("        for (var e = 0u; e < {d4}u; e++) {{\n");
    for r in 0..4 {
        s += &format!("            let a{r} = vec4<f32>(q_tile[(ty * 4u + {r}u) * {d4}u + e]);\n");
    }
    for c in 0..4 {
        s += &format!("            let b{c} = vec4<f32>(kv_tile[(tx * 4u + {c}u) * {d4}u + e]);\n");
    }
    for r in 0..4 {
        for c in 0..4 {
            s += &format!("            s{r}_{c} += dot(a{r}, b{c});\n");
        }
    }
    s += "        }\n";
    for r in 0..4 {
        s += &format!(
            "        let sr{r} = select(vec4<f32>(-3.0e38), vec4<f32>(s{r}_0, s{r}_1, s{r}_2, s{r}_3) * params.scale, vec4<u32>(kb + tx * 4u) + vec4<u32>(0u, 1u, 2u, 3u) < vec4<u32>(params.sk));\n"
        );
        s += &format!(
            "        red[(ty * 4u + {r}u) * 16u + tx] = max(max(sr{r}.x, sr{r}.y), max(sr{r}.z, sr{r}.w));\n"
        );
    }
    s += &format!(
        r#"        workgroupBarrier();
        for (var i = lid; i < {vd_len}u; i += 128u) {{
            let j = i / {dv4}u;
            var val = vec4<{t}>(0.0);
            if (kb + j < params.sk) {{
                val = quad_v(v_base + (kb + j) * params.v_s.z, i % {dv4}u, params.v_s.w);
            }}
            kv_tile[i] = val;
        }}
"#,
        vd_len = keys * dv4,
    );
    for r in 0..4 {
        s += &format!(
            r#"        {{
            var mx = m[{r}];
            for (var j = 0u; j < 16u; j++) {{ mx = max(mx, red[(ty * 4u + {r}u) * 16u + j]); }}
            let alpha = exp(m[{r}] - mx);
            m[{r}] = mx;
            let p = exp(sr{r} - vec4<f32>(mx));
            l[{r}] = l[{r}] * alpha + p.x + p.y + p.z + p.w;
            o[{r}] *= alpha;
            p_tile[(ty * 4u + {r}u) * 16u + tx] = vec4<{t}>(p);
        }}
"#
        );
    }
    let pv: String = ["x", "y", "z", "w"]
        .iter()
        .enumerate()
        .map(|(jj, c)| {
            format!(
                "                let v{jj} = vec4<f32>(kv_tile[(j4 * 4u + {jj}u) * {dv4}u + tx]);
                o[0] += p0.{c} * v{jj};
                o[1] += p1.{c} * v{jj};
                o[2] += p2.{c} * v{jj};
                o[3] += p3.{c} * v{jj};
"
            )
        })
        .collect();
    s += &format!(
        r#"        workgroupBarrier();
        if (tx < {dv4}u) {{
            for (var j4 = 0u; j4 < 16u; j4++) {{
                let p0 = vec4<f32>(p_tile[(ty * 4u) * 16u + j4]);
                let p1 = vec4<f32>(p_tile[(ty * 4u + 1u) * 16u + j4]);
                let p2 = vec4<f32>(p_tile[(ty * 4u + 2u) * 16u + j4]);
                let p3 = vec4<f32>(p_tile[(ty * 4u + 3u) * 16u + j4]);
{pv}
            }}
        }}
        workgroupBarrier();
    }}

    for (var r = 0u; r < 4u; r++) {{
        red[(ty * 4u + r) * 16u + tx] = l[r];
    }}
    workgroupBarrier();
    if (tx < {dv4}u) {{
        for (var r = 0u; r < 4u; r++) {{
            let row = row0 + ty * 4u + r;
            if (row >= params.sq) {{ continue; }}
            var sum = 0.0;
            for (var j = 0u; j < 16u; j++) {{ sum += red[(ty * 4u + r) * 16u + j]; }}
            let val = o[r] / sum;
            for (var j = 0u; j < 4u; j++) {{
                let c = tx * 4u + j;
                if (c < {dv}u) {{
                    outp[o_base + row * params.o_s.z + c * params.o_s.w] = {t}(val[j]);
                }}
            }}
        }}
    }}
}}
"#
    );
    s
}
