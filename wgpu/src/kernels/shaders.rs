//! WGSL sources for Phase 1 kernels.
//!
//! Flattened 1-D indexing: `dispatch_workgroups` is 3-D, tract tensors are
//! routinely 4-D+, so shaders take a linear id and apply strides from uniforms.
//! Byte offsets live in uniforms so arena views work without 256-byte-aligned
//! `GPUBufferBinding.offset` (the whole storage buffer is bound at 0).
//!
//! Structure follows tfjs-backend-webgpu unary/binary (Apache-2.0): one entry
//! point per op, tightly packed storage arrays. Not a transcription of their
//! TypeScript generators.

use tract_core::internal::*;

pub const WORKGROUP: u32 = 64;

/// Reads one of the eight values a uniform packs as two `vec4<u32>`.
const AT8: &str = "fn at8(a: vec4<u32>, b: vec4<u32>, i: u32) -> u32 {
    if (i < 4u) { return a[i]; }
    return b[i - 4u];
}";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LayoutKind {
    Unary,
    Binary,
    Resize,
    Ingest,
    /// Writes a tensor into a storage texture the caller owns.
    Export,
    /// A fused elementwise chain: `n` storage buffers, the last one the output.
    Chain(u8),
}

impl LayoutKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::Unary => "tract-wgpu-unary-layout",
            Self::Binary => "tract-wgpu-binary-layout",
            Self::Resize => "tract-wgpu-resize-layout",
            Self::Ingest => "tract-wgpu-ingest-layout",
            Self::Export => "tract-wgpu-export-layout",
            Self::Chain(_) => "tract-wgpu-chain-layout",
        }
    }

    pub fn storage_count(self) -> u32 {
        match self {
            Self::Unary => 2,
            Self::Binary => 3,
            Self::Resize => 4,
            Self::Ingest => 1,
            Self::Export => 1,
            Self::Chain(n) => n as u32,
        }
    }

    pub fn bind_group_layout_entries(self) -> Vec<wgpu::BindGroupLayoutEntry> {
        if self == Self::Export {
            return vec![
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: std::num::NonZeroU64::new(256),
                    },
                    count: None,
                },
            ];
        }
        if self == Self::Ingest {
            return vec![
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: std::num::NonZeroU64::new(256),
                    },
                    count: None,
                },
            ];
        }
        let mut entries = Vec::new();
        let n = self.storage_count();
        for i in 0..n {
            let read_only = i + 1 < n;
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: i,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        }
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: n,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: true,
                min_binding_size: std::num::NonZeroU64::new(256),
            },
            count: None,
        });
        entries
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShaderDtype {
    F32,
    F16,
}

impl ShaderDtype {
    pub fn from_datum(dt: DatumType) -> TractResult<Self> {
        match dt {
            DatumType::F32 => Ok(Self::F32),
            DatumType::F16 => Ok(Self::F16),
            _ => bail!("tract-wgpu Phase 1 kernel has no WGSL path for {dt:?}"),
        }
    }

    pub fn wgsl(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
        }
    }

    /// Entry points are suffixed with the WGSL type they were generated for.
    pub fn suffix(self) -> &'static str {
        self.wgsl()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModuleKind {
    ElementWise,
    Binary,
    Copy,
    Reduce,
    Pool,
    Cast,
    Conv,
    Deconv,
    Resize,
    Softmax,
    Ingest,
    Export,
    MatMul,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ModuleKey {
    pub kind: ModuleKind,
    pub dtype: ShaderDtype,
}

impl ModuleKey {
    /// Names the module in a debugger and in wgpu's own diagnostics. Only a
    /// cache miss builds one.
    pub fn label(self) -> String {
        format!("{:?}_{}", self.kind, self.dtype.suffix()).to_lowercase()
    }

    pub fn layout(self) -> LayoutKind {
        match self.kind {
            ModuleKind::ElementWise
            | ModuleKind::Copy
            | ModuleKind::Reduce
            | ModuleKind::Pool
            | ModuleKind::Cast
            | ModuleKind::Softmax => LayoutKind::Unary,
            ModuleKind::Binary | ModuleKind::Conv | ModuleKind::Deconv | ModuleKind::MatMul => {
                LayoutKind::Binary
            }
            ModuleKind::Resize => LayoutKind::Resize,
            ModuleKind::Ingest => LayoutKind::Ingest,
            ModuleKind::Export => LayoutKind::Export,
        }
    }

    pub fn wgsl(self) -> String {
        match self.kind {
            ModuleKind::ElementWise => element_wise_wgsl(self.dtype),
            ModuleKind::Binary => binary_wgsl(self.dtype),
            ModuleKind::Copy => copy_wgsl(self.dtype),
            ModuleKind::Reduce => reduce_wgsl(self.dtype),
            ModuleKind::Pool => pool_wgsl(self.dtype),
            ModuleKind::Cast => cast_wgsl(self.dtype),
            ModuleKind::Conv => conv_wgsl(self.dtype),
            ModuleKind::Deconv => deconv_wgsl(self.dtype),
            ModuleKind::Resize => resize_wgsl(self.dtype),
            ModuleKind::Softmax => softmax_wgsl(self.dtype),
            ModuleKind::Ingest => ingest_wgsl(self.dtype),
            ModuleKind::Export => export_wgsl(self.dtype),
            ModuleKind::MatMul => matmul_wgsl(self.dtype),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PipelineKey {
    pub module: ModuleKey,
    pub entry: EntryPoint,
}

/// How many values of the last axis one thread of a kernel handles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Width {
    Scalar,
    Vec4,
    Vec4Splat,
}

impl Width {
    fn infix(self) -> &'static str {
        match self {
            Self::Scalar => "",
            Self::Vec4 => "_v4",
            Self::Vec4Splat => "_v4s",
        }
    }
}

/// A kernel's WGSL entry point, held in the pieces the generator spells it
/// from so that naming one costs no allocation on the dispatch path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EntryPoint {
    stem: &'static str,
    infix: &'static str,
    suffix: &'static str,
}

impl EntryPoint {
    /// One of a module's per-dtype entry points, `<stem>_<dtype>`.
    pub fn typed(stem: &'static str, dtype: ShaderDtype) -> Self {
        Self { stem, infix: "", suffix: dtype.suffix() }
    }

    /// A per-dtype entry point in its `width`-value-per-thread form.
    pub fn wide(stem: &'static str, width: Width, dtype: ShaderDtype) -> Self {
        Self { stem, infix: width.infix(), suffix: dtype.suffix() }
    }

    /// An entry point that names the dtypes it works on itself.
    pub fn plain(name: &'static str) -> Self {
        Self { stem: name, infix: "", suffix: "" }
    }

    /// Spelled out, as the generated WGSL declares it. Only a pipeline cache
    /// miss needs this.
    pub fn name(&self) -> String {
        let sep = if self.suffix.is_empty() { "" } else { "_" };
        format!("{}{}{sep}{}", self.stem, self.infix, self.suffix)
    }
}

/// The dtypes this device can run a kernel for.
pub fn dtypes(shader_f16: bool) -> &'static [ShaderDtype] {
    if shader_f16 { &[ShaderDtype::F32, ShaderDtype::F16] } else { &[ShaderDtype::F32] }
}

/// Every pipeline a module serves: each of `stems` at each dtype the device
/// can run.
pub fn keys_for(kind: ModuleKind, stems: &[&'static str], shader_f16: bool) -> Vec<PipelineKey> {
    dtypes(shader_f16)
        .iter()
        .flat_map(|dt| {
            stems.iter().map(|stem| PipelineKey {
                module: ModuleKey { kind, dtype: *dt },
                entry: EntryPoint::typed(stem, *dt),
            })
        })
        .collect()
}

/// The set's own `&'static str` for `name`, matched the way an op reaches us
/// from tract: case-insensitively, and without allocating to lowercase it.
pub fn op_in_set(set: &'static [&'static str], name: &str) -> Option<&'static str> {
    set.iter().copied().find(|op| op.eq_ignore_ascii_case(name))
}

fn preamble(dt: ShaderDtype) -> String {
    let enable = match dt {
        ShaderDtype::F16 => "enable f16;\n",
        ShaderDtype::F32 => "",
    };
    enable.to_string()
}

/// One link of a fused elementwise chain. `Binary` reads its second operand
/// from an extra input; `swapped` puts the running value on the right.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ChainStep {
    Unary(String),
    Binary { op: String, rhs: usize, swapped: bool },
}

/// Names the generated program, and so keys its pipeline. `contiguous` says
/// which inputs share the output's layout and can skip index arithmetic.
/// How a chain operand is read: element for element with the output, one value
/// splatted across the four a thread handles, or gathered through its strides.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChainOperand {
    Contiguous,
    Splat,
    Gather,
}

pub fn chain_key(
    dt: ShaderDtype,
    steps: &[ChainStep],
    contiguous: &[bool],
    kinds: Option<&[ChainOperand]>,
) -> String {
    let mut key = format!("chain_{}", dt.suffix());
    match kinds {
        Some(kinds) => {
            key.push_str("_v4");
            for k in kinds {
                key.push(match k {
                    ChainOperand::Contiguous => 'c',
                    ChainOperand::Splat => 's',
                    ChainOperand::Gather => 'g',
                });
            }
        }
        None => {
            for c in contiguous {
                key.push(if *c { 'c' } else { 'b' });
            }
        }
    }
    for step in steps {
        match step {
            ChainStep::Unary(op) => key.push_str(&format!("_{op}")),
            ChainStep::Binary { op, rhs, swapped } => {
                key.push_str(&format!("_{op}{rhs}{}", if *swapped { "r" } else { "" }))
            }
        }
    }
    key
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
    let i = gid.x;
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
    let i = gid.x;
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

/// A four-wide wrapper for each op, so a kernel that moves whole `vec4`s can
/// still call them. The bodies stay scalar: the win here is the wider load and
/// store, not the arithmetic.
fn unary_ops_vec4_wgsl(t: &str) -> String {
    let mut s = String::new();
    for name in ELEMENT_WISE_OPS {
        s.push_str(&format!(
            "fn op_{name}_v(x: vec4<{t}>) -> vec4<{t}> {{ return vec4<{t}>(op_{name}(x.x), op_{name}(x.y), op_{name}(x.z), op_{name}(x.w)); }}\n"
        ));
    }
    s
}

fn binary_ops_vec4_wgsl(t: &str) -> String {
    let mut s = String::new();
    for name in BINARY_OPS {
        s.push_str(&format!(
            "fn op_{name}_v(a: vec4<{t}>, b: vec4<{t}>) -> vec4<{t}> {{ return vec4<{t}>(op_{name}(a.x, b.x), op_{name}(a.y, b.y), op_{name}(a.z, b.z), op_{name}(a.w, b.w)); }}\n"
        ));
    }
    s
}

/// Every unary `op_*` a kernel may call, plus the erf polynomial they share.
fn unary_ops_wgsl(t: &str) -> String {
    format!(
        r#"fn approx_erf(x: f32) -> f32 {{
    let a1 = 0.0705230784;
    let a2 = 0.0422820123;
    let a3 = 0.0092705272;
    let a4 = 0.0001520143;
    let a5 = 0.0002765672;
    let a6 = 0.0000430638;
    let ax = abs(x);
    var y = a6 * ax;
    y = (a5 + y) * ax;
    y = (a4 + y) * ax;
    y = (a3 + y) * ax;
    y = (a2 + y) * ax;
    y = (a1 + y) * ax;
    y = 1.0 - (1.0 / pow(y + 1.0, 16.0));
    return sign(x) * y;
}}

fn op_abs(x: {t}) -> {t} {{ return abs(x); }}
fn op_exp(x: {t}) -> {t} {{ return exp(x); }}
fn op_ln(x: {t}) -> {t} {{ return log(x); }}
fn op_sqrt(x: {t}) -> {t} {{ return sqrt(x); }}
fn op_rsqrt(x: {t}) -> {t} {{ return inverseSqrt(x); }}
fn op_sigmoid(x: {t}) -> {t} {{
    let y = {t}(1.0) / ({t}(1.0) + exp(-abs(x)));
    if (x < {t}(0.0)) {{ return {t}(1.0) - y; }}
    return y;
}}
fn op_square(x: {t}) -> {t} {{ return x * x; }}
fn op_recip(x: {t}) -> {t} {{ return {t}(1.0) / x; }}
fn op_ceil(x: {t}) -> {t} {{ return ceil(x); }}
fn op_floor(x: {t}) -> {t} {{ return floor(x); }}
fn op_round(x: {t}) -> {t} {{ return sign(x) * floor(abs(x) + {t}(0.5)); }}
fn op_roundhalftoeven(x: {t}) -> {t} {{ return round(x); }}
fn op_cos(x: {t}) -> {t} {{ return cos(x); }}
fn op_sin(x: {t}) -> {t} {{ return sin(x); }}
fn op_tan(x: {t}) -> {t} {{ return tan(x); }}
fn op_acos(x: {t}) -> {t} {{ return acos(x); }}
fn op_asin(x: {t}) -> {t} {{ return asin(x); }}
fn op_atan(x: {t}) -> {t} {{ return atan(x); }}
fn op_cosh(x: {t}) -> {t} {{ return cosh(x); }}
fn op_sinh(x: {t}) -> {t} {{ return sinh(x); }}
fn op_tanh(x: {t}) -> {t} {{ return tanh(x); }}
fn op_acosh(x: {t}) -> {t} {{ return acosh(x); }}
fn op_asinh(x: {t}) -> {t} {{ return asinh(x); }}
fn op_atanh(x: {t}) -> {t} {{ return atanh(x); }}
fn op_erf(x: {t}) -> {t} {{ return {t}(approx_erf(f32(x))); }}
fn op_neg(x: {t}) -> {t} {{ return -x; }}
fn op_sign(x: {t}) -> {t} {{ return sign(x); }}
fn op_hardswish(x: {t}) -> {t} {{
    let h = max({t}(0.0), min({t}(1.0), x / {t}(6.0) + {t}(0.5)));
    return x * h;
}}
fn op_silu(x: {t}) -> {t} {{ return x / ({t}(1.0) + exp(-x)); }}
"#
    )
}

/// Every binary `op_*` a kernel may call.
fn binary_ops_wgsl(t: &str) -> String {
    format!(
        r#"fn op_add(a: {t}, b: {t}) -> {t} {{ return a + b; }}
fn op_sub(a: {t}, b: {t}) -> {t} {{ return a - b; }}
fn op_mul(a: {t}, b: {t}) -> {t} {{ return a * b; }}
fn op_div(a: {t}, b: {t}) -> {t} {{ return a / b; }}
fn op_min(a: {t}, b: {t}) -> {t} {{ return min(a, b); }}
fn op_max(a: {t}, b: {t}) -> {t} {{ return max(a, b); }}
fn op_pow(a: {t}, b: {t}) -> {t} {{ return pow(a, b); }}
"#
    )
}

fn element_wise_wgsl(dt: ShaderDtype) -> String {
    let t = dt.wgsl();
    let suf = dt.suffix();
    let lib = unary_ops_wgsl(t);
    let mut s = preamble(dt);
    s.push_str(&format!(
        r#"
struct Params {{
    off_in: u32,
    off_out: u32,
    len: u32,
    _pad: u32,
}}

@group(0) @binding(0) var<storage, read> inp: array<{t}>;
@group(0) @binding(1) var<storage, read_write> outp: array<{t}>;
@group(0) @binding(2) var<uniform> params: Params;

{lib}
"#
    ));
    for name in ELEMENT_WISE_OPS {
        s.push_str(&format!(
            r#"
@compute @workgroup_size({WORKGROUP})
fn {name}_{suf}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= params.len) {{ return; }}
    outp[params.off_out + i] = op_{name}(inp[params.off_in + i]);
}}
"#
        ));
    }
    s
}

pub const ELEMENT_WISE_OPS: &[&str] = &[
    "abs",
    "exp",
    "ln",
    "sigmoid",
    "square",
    "sqrt",
    "rsqrt",
    "recip",
    "ceil",
    "floor",
    "round",
    "roundhalftoeven",
    "cos",
    "acos",
    "acosh",
    "cosh",
    "sin",
    "asin",
    "asinh",
    "sinh",
    "tan",
    "atan",
    "atanh",
    "tanh",
    "erf",
    "neg",
    "sign",
    "hardswish",
    "silu",
];

fn binary_wgsl(dt: ShaderDtype) -> String {
    let t = dt.wgsl();
    let suf = dt.suffix();
    let lib = binary_ops_wgsl(t);
    let mut s = preamble(dt);
    s.push_str(&format!(
        r#"
struct Params {{
    off_lhs: u32,
    off_rhs: u32,
    off_out: u32,
    rank: u32,
    len: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
    lhs_s0: vec4<u32>,
    lhs_s1: vec4<u32>,
    rhs_s0: vec4<u32>,
    rhs_s1: vec4<u32>,
    out_sh0: vec4<u32>,
    out_sh1: vec4<u32>,
}}

@group(0) @binding(0) var<storage, read> lhs: array<{t}>;
@group(0) @binding(1) var<storage, read> rhs: array<{t}>;
@group(0) @binding(2) var<storage, read_write> outp: array<{t}>;
@group(0) @binding(3) var<uniform> params: Params;

{AT8}

fn gather_idx(linear: u32, is_lhs: bool) -> u32 {{
    var rest = linear;
    var idx = 0u;
    let r = params.rank;
    for (var k = 0u; k < r; k++) {{
        let axis = r - 1u - k;
        let dim = max(at8(params.out_sh0, params.out_sh1, axis), 1u);
        let c = rest % dim;
        rest = rest / dim;
        let stride = select(
            at8(params.rhs_s0, params.rhs_s1, axis),
            at8(params.lhs_s0, params.lhs_s1, axis),
            is_lhs,
        );
        idx += c * stride;
    }}
    return idx;
}}

{lib}"#
    ));
    for name in BINARY_OPS {
        s.push_str(&format!(
            r#"
@compute @workgroup_size({WORKGROUP})
fn {name}_{suf}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= params.len) {{ return; }}
    let li = params.off_lhs + gather_idx(i, true);
    let ri = params.off_rhs + gather_idx(i, false);
    outp[params.off_out + i] = op_{name}(lhs[li], rhs[ri]);
}}
"#
        ));
    }
    s.push_str(&binary_vec4_entries(t, suf));
    s
}

/// The four-wide binary entries. `_v4` is both operands element for element
/// with the output; `_v4s` is a right operand that broadcasts over the last
/// axis, so one value covers the four a thread handles. Anything else gathers
/// per element and gains nothing from a wider thread.
fn binary_vec4_entries(t: &str, suf: &str) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        r#"
fn lhs4(i: u32) -> vec4<{t}> {{
    let b = params.off_lhs + i * 4u;
    return vec4<{t}>(lhs[b], lhs[b + 1u], lhs[b + 2u], lhs[b + 3u]);
}}

fn rhs4(i: u32) -> vec4<{t}> {{
    let b = params.off_rhs + i * 4u;
    return vec4<{t}>(rhs[b], rhs[b + 1u], rhs[b + 2u], rhs[b + 3u]);
}}

fn store4(i: u32, v: vec4<{t}>) {{
    let b = params.off_out + i * 4u;
    outp[b] = v.x;
    outp[b + 1u] = v.y;
    outp[b + 2u] = v.z;
    outp[b + 3u] = v.w;
}}
"#
    ));
    s.push_str(&binary_ops_vec4_wgsl(t));
    for name in BINARY_OPS {
        s.push_str(&format!(
            r#"
@compute @workgroup_size({WORKGROUP})
fn {name}_v4_{suf}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i * 4u >= params.len) {{ return; }}
    store4(i, op_{name}_v(lhs4(i), rhs4(i)));
}}

@compute @workgroup_size({WORKGROUP})
fn {name}_v4s_{suf}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i * 4u >= params.len) {{ return; }}
    let r = vec4<{t}>(rhs[params.off_rhs + gather_idx(i * 4u, false)]);
    store4(i, op_{name}_v(lhs4(i), r));
}}
"#
        ));
    }
    s
}

pub const BINARY_OPS: &[&str] = &["mul", "add", "div", "sub", "pow", "min", "max"];

fn copy_wgsl(dt: ShaderDtype) -> String {
    let t = dt.wgsl();
    let suf = dt.suffix();
    let mut s = preamble(dt);
    s.push_str(&format!(
        r#"
struct Params {{
    off_in: u32,
    off_out: u32,
    rank: u32,
    len: u32,
    in_s0: vec4<u32>,
    in_s1: vec4<u32>,
    out_s0: vec4<u32>,
    out_s1: vec4<u32>,
    out_sh0: vec4<u32>,
    out_sh1: vec4<u32>,
}}

@group(0) @binding(0) var<storage, read> inp: array<{t}>;
@group(0) @binding(1) var<storage, read_write> outp: array<{t}>;
@group(0) @binding(2) var<uniform> params: Params;

{AT8}

@compute @workgroup_size({WORKGROUP})
fn copy_{suf}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= params.len) {{ return; }}
    var rest = i;
    var in_i = params.off_in;
    var out_i = params.off_out;
    let r = params.rank;
    for (var k = 0u; k < r; k++) {{
        let axis = r - 1u - k;
        let dim = max(at8(params.out_sh0, params.out_sh1, axis), 1u);
        let c = rest % dim;
        rest = rest / dim;
        in_i += c * at8(params.in_s0, params.in_s1, axis);
        out_i += c * at8(params.out_s0, params.out_s1, axis);
    }}
    outp[out_i] = inp[in_i];
}}
"#
    ));
    s
}

fn reduce_wgsl(dt: ShaderDtype) -> String {
    let t = dt.wgsl();
    let suf = dt.suffix();
    let mut s = preamble(dt);
    s.push_str(&format!(
        r#"
struct Params {{
    off_in: u32,
    off_out: u32,
    outer: u32,
    k: u32,
    inner: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}}

@group(0) @binding(0) var<storage, read> inp: array<{t}>;
@group(0) @binding(1) var<storage, read_write> outp: array<{t}>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size({WORKGROUP})
fn reduce_sum_{suf}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    let n = params.outer * params.inner;
    if (i >= n) {{ return; }}
    let o = i / params.inner;
    let r = i % params.inner;
    var acc = {t}(0.0);
    for (var k = 0u; k < params.k; k++) {{
        acc += inp[params.off_in + (o * params.k + k) * params.inner + r];
    }}
    outp[params.off_out + i] = acc;
}}

@compute @workgroup_size({WORKGROUP})
fn reduce_prod_{suf}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    let n = params.outer * params.inner;
    if (i >= n) {{ return; }}
    let o = i / params.inner;
    let r = i % params.inner;
    var acc = {t}(1.0);
    for (var k = 0u; k < params.k; k++) {{
        acc *= inp[params.off_in + (o * params.k + k) * params.inner + r];
    }}
    outp[params.off_out + i] = acc;
}}

@compute @workgroup_size({WORKGROUP})
fn reduce_max_{suf}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    let n = params.outer * params.inner;
    if (i >= n) {{ return; }}
    let o = i / params.inner;
    let r = i % params.inner;
    var acc = inp[params.off_in + (o * params.k) * params.inner + r];
    for (var k = 1u; k < params.k; k++) {{
        acc = max(acc, inp[params.off_in + (o * params.k + k) * params.inner + r]);
    }}
    outp[params.off_out + i] = acc;
}}

@compute @workgroup_size({WORKGROUP})
fn reduce_min_{suf}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    let n = params.outer * params.inner;
    if (i >= n) {{ return; }}
    let o = i / params.inner;
    let r = i % params.inner;
    var acc = inp[params.off_in + (o * params.k) * params.inner + r];
    for (var k = 1u; k < params.k; k++) {{
        acc = min(acc, inp[params.off_in + (o * params.k + k) * params.inner + r]);
    }}
    outp[params.off_out + i] = acc;
}}

@compute @workgroup_size({WORKGROUP})
fn reduce_mean_of_squares_{suf}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    let n = params.outer * params.inner;
    if (i >= n) {{ return; }}
    let o = i / params.inner;
    let r = i % params.inner;
    var acc = {t}(0.0);
    for (var k = 0u; k < params.k; k++) {{
        let v = inp[params.off_in + (o * params.k + k) * params.inner + r];
        acc += v * v;
    }}
    outp[params.off_out + i] = acc / {t}(f32(params.k));
}}
"#
    ));
    s
}

fn pool_wgsl(dt: ShaderDtype) -> String {
    let t = dt.wgsl();
    let suf = dt.suffix();
    let neg_inf = match dt {
        ShaderDtype::F32 => "f32(-3.402823e+38)",
        ShaderDtype::F16 => "f16(-65504.0)",
    };
    let mut s = preamble(dt);
    s.push_str(&format!(
        r#"
struct Params {{
    off_in: u32,
    off_out: u32,
    n: i32,
    ih: i32,
    iw: i32,
    c: i32,
    oh: i32,
    ow: i32,
    kh: i32,
    kw: i32,
    stride_h: i32,
    stride_w: i32,
    pad_h: i32,
    pad_w: i32,
    dil_h: i32,
    dil_w: i32,
    count_include_pad: i32,
    normalize: i32,
    channels_last: i32,
    _p0: i32,
}}

@group(0) @binding(0) var<storage, read> inp: array<{t}>;
@group(0) @binding(1) var<storage, read_write> outp: array<{t}>;
@group(0) @binding(2) var<uniform> params: Params;

fn in_idx_nhwc(n: i32, h: i32, w: i32, c: i32) -> u32 {{
    return u32(((n * params.ih + h) * params.iw + w) * params.c + c);
}}
fn in_idx_nchw(n: i32, c: i32, h: i32, w: i32) -> u32 {{
    return u32(((n * params.c + c) * params.ih + h) * params.iw + w);
}}
fn out_idx_nhwc(n: i32, h: i32, w: i32, c: i32) -> u32 {{
    return u32(((n * params.oh + h) * params.ow + w) * params.c + c);
}}
fn out_idx_nchw(n: i32, c: i32, h: i32, w: i32) -> u32 {{
    return u32(((n * params.c + c) * params.oh + h) * params.ow + w);
}}

@compute @workgroup_size({WORKGROUP})
fn max_pool_2d_{suf}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = i32(gid.x);
    let nout = params.n * params.oh * params.ow * params.c;
    if (i >= nout) {{ return; }}
    var rest = i;
    let c = rest % params.c; rest = rest / params.c;
    let ow = rest % params.ow; rest = rest / params.ow;
    let oh = rest % params.oh;
    let n = rest / params.oh;
    let h0 = oh * params.stride_h - params.pad_h;
    let w0 = ow * params.stride_w - params.pad_w;
    var best = {neg_inf};
    for (var kh = 0; kh < params.kh; kh++) {{
        let ih = h0 + kh * params.dil_h;
        if (ih < 0 || ih >= params.ih) {{ continue; }}
        for (var kw = 0; kw < params.kw; kw++) {{
            let iw = w0 + kw * params.dil_w;
            if (iw < 0 || iw >= params.iw) {{ continue; }}
            var idx: u32;
            if (params.channels_last != 0) {{
                idx = in_idx_nhwc(n, ih, iw, c);
            }} else {{
                idx = in_idx_nchw(n, c, ih, iw);
            }}
            best = max(best, inp[params.off_in + idx]);
        }}
    }}
    var oidx: u32;
    if (params.channels_last != 0) {{
        oidx = out_idx_nhwc(n, oh, ow, c);
    }} else {{
        oidx = out_idx_nchw(n, c, oh, ow);
    }}
    outp[params.off_out + oidx] = best;
}}

@compute @workgroup_size({WORKGROUP})
fn sum_pool_2d_{suf}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = i32(gid.x);
    let nout = params.n * params.oh * params.ow * params.c;
    if (i >= nout) {{ return; }}
    var rest = i;
    let c = rest % params.c; rest = rest / params.c;
    let ow = rest % params.ow; rest = rest / params.ow;
    let oh = rest % params.oh;
    let n = rest / params.oh;
    let h0 = oh * params.stride_h - params.pad_h;
    let w0 = ow * params.stride_w - params.pad_w;
    var acc = {t}(0.0);
    var count = 0;
    for (var kh = 0; kh < params.kh; kh++) {{
        let ih = h0 + kh * params.dil_h;
        if (ih < 0 || ih >= params.ih) {{ continue; }}
        for (var kw = 0; kw < params.kw; kw++) {{
            let iw = w0 + kw * params.dil_w;
            if (iw < 0 || iw >= params.iw) {{ continue; }}
            var idx: u32;
            if (params.channels_last != 0) {{
                idx = in_idx_nhwc(n, ih, iw, c);
            }} else {{
                idx = in_idx_nchw(n, c, ih, iw);
            }}
            acc += inp[params.off_in + idx];
            count += 1;
        }}
    }}
    if (params.normalize != 0) {{
        var denom = count;
        if (params.count_include_pad != 0) {{
            denom = params.kh * params.kw;
        }}
        if (denom > 0) {{
            acc = acc / {t}(f32(denom));
        }}
    }}
    var oidx: u32;
    if (params.channels_last != 0) {{
        oidx = out_idx_nhwc(n, oh, ow, c);
    }} else {{
        oidx = out_idx_nchw(n, c, oh, ow);
    }}
    outp[params.off_out + oidx] = acc;
}}
"#
    ));
    s
}

fn cast_wgsl(dt: ShaderDtype) -> String {
    match dt {
        ShaderDtype::F32 => {
            // f32 -> f16
            format!(
                r#"
enable f16;
struct Params {{ off_in: u32, off_out: u32, len: u32, _p: u32 }}
@group(0) @binding(0) var<storage, read> inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> outp: array<f16>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size({WORKGROUP})
fn cast_f32_f16(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= params.len) {{ return; }}
    outp[params.off_out + i] = f16(inp[params.off_in + i]);
}}
"#
            )
        }
        ShaderDtype::F16 => {
            format!(
                r#"
enable f16;
struct Params {{ off_in: u32, off_out: u32, len: u32, _p: u32 }}
@group(0) @binding(0) var<storage, read> inp: array<f16>;
@group(0) @binding(1) var<storage, read_write> outp: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size({WORKGROUP})
fn cast_f16_f32(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= params.len) {{ return; }}
    outp[params.off_out + i] = f32(inp[params.off_in + i]);
}}
"#
            )
        }
    }
}

/// Workgroup edge of the depthwise tile: 16x16 output values per workgroup.
pub const DW_WG: u32 = 16;

/// Shape of one depthwise convolution, baked into its program so the filter
/// loops unroll and the halo tile is sized exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DepthwiseShape {
    pub kh: u32,
    pub kw: u32,
    pub stride_h: u32,
    pub stride_w: u32,
    pub dil_h: u32,
    pub dil_w: u32,
}

impl DepthwiseShape {
    fn tile_h(&self) -> u32 {
        (DW_WG - 1) * self.stride_h + (self.kh - 1) * self.dil_h + 1
    }

    fn tile_w(&self) -> u32 {
        (DW_WG - 1) * self.stride_w + (self.kw - 1) * self.dil_w + 1
    }

    pub fn key(&self) -> String {
        format!(
            "{}x{}s{}x{}d{}x{}",
            self.kh, self.kw, self.stride_h, self.stride_w, self.dil_h, self.dil_w
        )
    }
}

/// Depthwise convolution, ported from tfjs-backend-webgpu
/// `DepthwiseConv2DNCHWSharedProgram`
/// (tfjs-backend-webgpu/src/depthwise_conv2d_nchw_shared_webgpu.ts, Apache-2.0,
/// Copyright 2021 Google LLC): a workgroup owns one 16x16 output tile of one
/// channel, staging that tile's input halo and the filter in workgroup memory,
/// so each input value is read once instead of once per filter tap. Strides and
/// dilations size the halo here, where tfjs assumed one; indices come from
/// tract's strides, and the epilogue replaces their bias/activation snippet.
pub fn conv_depthwise_module(
    dt: ShaderDtype,
    shape: DepthwiseShape,
    epilogue: &[ChainStep],
    extras: usize,
) -> String {
    let t = dt.wgsl();
    let suf = dt.suffix();
    let DepthwiseShape { kh, kw, stride_h, stride_w, dil_h, dil_w } = shape;
    let (tile_h, tile_w) = (shape.tile_h(), shape.tile_w());
    let wg = DW_WG;
    let out_binding = 2 + extras;
    let uniform_binding = 3 + extras;
    let extra_bindings = (0..extras)
        .map(|i| {
            format!("@group(0) @binding({}) var<storage, read> extra{i}: array<{t}>;\n", 2 + i)
        })
        .collect::<String>();
    let lib = if epilogue.is_empty() {
        String::new()
    } else {
        format!("{}{}", unary_ops_wgsl(t), binary_ops_wgsl(t))
    };
    let epilogue = epilogue_body(epilogue, "co");
    let mut s = preamble(dt);
    s.push_str(&format!(
        r#"
struct Params {{
    off_in: u32,
    off_w: u32,
    off_out: u32,
    n: u32,
    co: u32,
    ih: u32,
    iw: u32,
    oh: u32,
    ow: u32,
    pad_h: i32,
    pad_w: i32,
    _p0: u32,
    in_sn: i32,
    in_sc: i32,
    in_sh: i32,
    in_sw: i32,
    w_so: i32,
    w_sh: i32,
    w_sw: i32,
    _p1: u32,
    out_sn: i32,
    out_sc: i32,
    out_sh: i32,
    out_sw: i32,
    off_extra: vec4<u32>,
    mode_extra: vec4<u32>,
}}

@group(0) @binding(0) var<storage, read> inp: array<{t}>;
@group(0) @binding(1) var<storage, read> wgt: array<{t}>;
{extra_bindings}@group(0) @binding({out_binding}) var<storage, read_write> outp: array<{t}>;
@group(0) @binding({uniform_binding}) var<uniform> params: Params;
{lib}
var<workgroup> x_tile: array<array<f32, {tile_w}>, {tile_h}>;
var<workgroup> w_tile: array<array<f32, {kw}>, {kh}>;

@compute @workgroup_size({wg}, {wg}, 1)
fn conv_dw_{suf}(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) lidx: u32,
) {{
    let co = wid.z % params.co;
    let n = wid.z / params.co;
    let oh0 = wid.y * {wg}u;
    let ow0 = wid.x * {wg}u;
    let ih0 = i32(oh0) * {stride_h} - params.pad_h;
    let iw0 = i32(ow0) * {stride_w} - params.pad_w;

    for (var r = lid.y; r < {tile_h}u; r = r + {wg}u) {{
        for (var c = lid.x; c < {tile_w}u; c = c + {wg}u) {{
            let ih = ih0 + i32(r);
            let iw = iw0 + i32(c);
            var v = 0.0;
            if (ih >= 0 && ih < i32(params.ih) && iw >= 0 && iw < i32(params.iw)) {{
                let idx = params.off_in + u32(
                    i32(n) * params.in_sn
                    + i32(co) * params.in_sc
                    + ih * params.in_sh
                    + iw * params.in_sw
                );
                v = f32(inp[idx]);
            }}
            x_tile[r][c] = v;
        }}
    }}
    if (lidx < {kh}u * {kw}u) {{
        let wr = lidx / {kw}u;
        let wc = lidx % {kw}u;
        let idx = params.off_w + u32(
            i32(co) * params.w_so + i32(wr) * params.w_sh + i32(wc) * params.w_sw
        );
        w_tile[wr][wc] = f32(wgt[idx]);
    }}
    workgroupBarrier();

    let oh = oh0 + lid.y;
    let ow = ow0 + lid.x;
    if (oh >= params.oh || ow >= params.ow) {{ return; }}
    var acc = 0.0;
    for (var wr = 0u; wr < {kh}u; wr++) {{
        for (var wc = 0u; wc < {kw}u; wc++) {{
            acc = fma(
                x_tile[lid.y * {stride_h}u + wr * {dil_h}u][lid.x * {stride_w}u + wc * {dil_w}u],
                w_tile[wr][wc],
                acc
            );
        }}
    }}
    var v = {t}(acc);
{epilogue}
    let out_i = params.off_out + u32(
        i32(n) * params.out_sn
        + i32(co) * params.out_sc
        + i32(oh) * params.out_sh
        + i32(ow) * params.out_sw
    );
    outp[out_i] = v;
}}
"#
    ));
    s
}

fn conv_wgsl(dt: ShaderDtype) -> String {
    conv_module(dt, &[], 0)
}

/// `extras` epilogue operands bind between the weights and the output.
pub fn conv_module(dt: ShaderDtype, epilogue: &[ChainStep], extras: usize) -> String {
    let t = dt.wgsl();
    let suf = dt.suffix();
    let out_binding = 2 + extras;
    let uniform_binding = 3 + extras;
    let extra_bindings = (0..extras)
        .map(|i| {
            format!("@group(0) @binding({}) var<storage, read> extra{i}: array<{t}>;\n", 2 + i)
        })
        .collect::<String>();
    let lib = if epilogue.is_empty() {
        String::new()
    } else {
        format!("{}{}", unary_ops_wgsl(t), binary_ops_wgsl(t))
    };
    let epilogue = epilogue_body(epilogue, "co");
    let mut s = preamble(dt);
    // Packed params: see kernels/conv.rs. 2D NCHW/NHWC, OIHW weights.
    s.push_str(&format!(
        r#"
struct Params {{
    off_in: u32,
    off_w: u32,
    off_out: u32,
    n: u32,
    ci: u32,
    co: u32,
    ih: u32,
    iw: u32,
    oh: u32,
    ow: u32,
    kh: u32,
    kw: u32,
    groups: u32,
    ci_pg: u32,
    co_pg: u32,
    channels_last: u32,
    pad_h: i32,
    pad_w: i32,
    stride_h: i32,
    stride_w: i32,
    dil_h: i32,
    dil_w: i32,
    in_sn: i32,
    in_sc: i32,
    in_sh: i32,
    in_sw: i32,
    w_so: i32,
    w_si: i32,
    w_sh: i32,
    w_sw: i32,
    out_sn: i32,
    out_sc: i32,
    out_sh: i32,
    out_sw: i32,
    _pad: vec2<u32>,
    off_extra: vec4<u32>,
    mode_extra: vec4<u32>,
}}

@group(0) @binding(0) var<storage, read> inp: array<{t}>;
@group(0) @binding(1) var<storage, read> wgt: array<{t}>;
{extra_bindings}@group(0) @binding({out_binding}) var<storage, read_write> outp: array<{t}>;
@group(0) @binding({uniform_binding}) var<uniform> params: Params;
{lib}
@compute @workgroup_size({WORKGROUP})
fn conv2d_{suf}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    let nout = params.n * params.co * params.oh * params.ow;
    if (i >= nout) {{ return; }}
    var rest = i;
    let ow = rest % params.ow; rest = rest / params.ow;
    let oh = rest % params.oh; rest = rest / params.oh;
    let co = rest % params.co;
    let n = rest / params.co;
    let g = co / params.co_pg;
    let ci0 = g * params.ci_pg;
    var acc = 0.0;
    // Where a tap reads does not depend on the channel, so the channels run
    // innermost and the window is walked once rather than once per channel.
    for (var kh = 0u; kh < params.kh; kh++) {{
        let ih = i32(oh) * params.stride_h + i32(kh) * params.dil_h - params.pad_h;
        if (ih < 0 || ih >= i32(params.ih)) {{ continue; }}
        for (var kw = 0u; kw < params.kw; kw++) {{
            let iw = i32(ow) * params.stride_w + i32(kw) * params.dil_w - params.pad_w;
            if (iw < 0 || iw >= i32(params.iw)) {{ continue; }}
            let in_base = i32(n) * params.in_sn
                + i32(ci0) * params.in_sc
                + ih * params.in_sh
                + iw * params.in_sw;
            let w_base = i32(co) * params.w_so
                + i32(kh) * params.w_sh
                + i32(kw) * params.w_sw;
            for (var ci = 0u; ci < params.ci_pg; ci++) {{
                let in_i = params.off_in + u32(in_base + i32(ci) * params.in_sc);
                let w_i = params.off_w + u32(w_base + i32(ci) * params.w_si);
                acc += f32(inp[in_i]) * f32(wgt[w_i]);
            }}
        }}
    }}
    let out_i = params.off_out + u32(
        i32(n) * params.out_sn
        + i32(co) * params.out_sc
        + i32(oh) * params.out_sh
        + i32(ow) * params.out_sw
    );
    var v = {t}(acc);
{epilogue}
    outp[out_i] = v;
}}
"#
    ));
    s
}

fn deconv_wgsl(dt: ShaderDtype) -> String {
    let t = dt.wgsl();
    let suf = dt.suffix();
    let mut s = preamble(dt);
    // Gather-based conv_transpose (ORT style). WGSL has no float atomics.
    s.push_str(&format!(
        r#"
struct Params {{
    off_in: u32,
    off_w: u32,
    off_out: u32,
    n: u32,
    ci: u32,
    co: u32,
    ih: u32,
    iw: u32,
    oh: u32,
    ow: u32,
    kh: u32,
    kw: u32,
    groups: u32,
    ci_pg: u32,
    co_pg: u32,
    _p0: u32,
    pad_h: i32,
    pad_w: i32,
    stride_h: i32,
    stride_w: i32,
    dil_h: i32,
    dil_w: i32,
    in_sn: i32,
    in_sc: i32,
    in_sh: i32,
    in_sw: i32,
    w_so: i32,
    w_si: i32,
    w_sh: i32,
    w_sw: i32,
    out_sn: i32,
    out_sc: i32,
    out_sh: i32,
    out_sw: i32,
}}

@group(0) @binding(0) var<storage, read> inp: array<{t}>;
@group(0) @binding(1) var<storage, read> wgt: array<{t}>;
@group(0) @binding(2) var<storage, read_write> outp: array<{t}>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size({WORKGROUP})
fn conv_transpose2d_{suf}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    let nout = params.n * params.co * params.oh * params.ow;
    if (i >= nout) {{ return; }}
    var rest = i;
    let ow = rest % params.ow; rest = rest / params.ow;
    let oh = rest % params.oh; rest = rest / params.oh;
    let co = rest % params.co;
    let n = rest / params.co;
    let g = co / params.co_pg;
    let ci0 = g * params.ci_pg;
    var acc = 0.0;
    // Which taps land on an input pixel does not depend on the channel, and
    // finding out costs two integer divisions, so the channels run innermost.
    for (var kh = 0u; kh < params.kh; kh++) {{
        let ih_num = i32(oh) + params.pad_h - i32(kh) * params.dil_h;
        if (params.stride_h == 0 || ih_num % params.stride_h != 0) {{ continue; }}
        let ih = ih_num / params.stride_h;
        if (ih < 0 || ih >= i32(params.ih)) {{ continue; }}
        for (var kw = 0u; kw < params.kw; kw++) {{
            let iw_num = i32(ow) + params.pad_w - i32(kw) * params.dil_w;
            if (params.stride_w == 0 || iw_num % params.stride_w != 0) {{ continue; }}
            let iw = iw_num / params.stride_w;
            if (iw < 0 || iw >= i32(params.iw)) {{ continue; }}
            let in_base = i32(n) * params.in_sn
                + i32(ci0) * params.in_sc
                + ih * params.in_sh
                + iw * params.in_sw;
            let w_base = i32(co) * params.w_so
                + i32(kh) * params.w_sh
                + i32(kw) * params.w_sw;
            for (var ci = 0u; ci < params.ci_pg; ci++) {{
                let in_i = params.off_in + u32(in_base + i32(ci) * params.in_sc);
                let w_i = params.off_w + u32(w_base + i32(ci) * params.w_si);
                acc += f32(inp[in_i]) * f32(wgt[w_i]);
            }}
        }}
    }}
    let out_i = params.off_out + u32(
        i32(n) * params.out_sn
        + i32(co) * params.out_sc
        + i32(oh) * params.out_sh
        + i32(ow) * params.out_sw
    );
    outp[out_i] = {t}(acc);
}}
"#
    ));
    s
}

fn resize_wgsl(dt: ShaderDtype) -> String {
    let t = dt.wgsl();
    let suf = dt.suffix();
    let mut s = preamble(dt);
    s.push_str(&format!(
        r#"
struct Params {{
    off_in: u32,
    off_idx: u32,
    off_w: u32,
    off_out: u32,
    outer: u32,
    len_in: u32,
    len_out: u32,
    inner: u32,
    window: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}}

@group(0) @binding(0) var<storage, read> inp: array<{t}>;
@group(0) @binding(1) var<storage, read> indices: array<i32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> outp: array<{t}>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size({WORKGROUP})
fn resize_axis_{suf}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    let n = params.outer * params.len_out * params.inner;
    if (i >= n) {{ return; }}
    let inner_i = i % params.inner;
    let t = i / params.inner;
    let xo = t % params.len_out;
    let outer_i = t / params.len_out;
    var acc = 0.0;
    for (var w = 0u; w < params.window; w++) {{
        let ix = indices[params.off_idx + xo * params.window + w];
        let wt = weights[params.off_w + xo * params.window + w];
        let in_i = params.off_in + (outer_i * params.len_in + u32(ix)) * params.inner + inner_i;
        acc += f32(inp[in_i]) * wt;
    }}
    outp[params.off_out + (outer_i * params.len_out) * params.inner + xo * params.inner + inner_i] = {t}(acc);
}}
"#
    ));
    s
}

fn softmax_wgsl(dt: ShaderDtype) -> String {
    let t = dt.wgsl();
    let suf = dt.suffix();
    let mut s = preamble(dt);
    s.push_str(&format!(
        r#"
struct Params {{
    off_in: u32,
    off_out: u32,
    outer: u32,
    k: u32,
    inner: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}}

@group(0) @binding(0) var<storage, read> inp: array<{t}>;
@group(0) @binding(1) var<storage, read_write> outp: array<{t}>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size({WORKGROUP})
fn softmax_{suf}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    let n = params.outer * params.inner;
    if (i >= n) {{ return; }}
    let o = i / params.inner;
    let r = i % params.inner;
    var m = inp[params.off_in + (o * params.k) * params.inner + r];
    for (var k = 1u; k < params.k; k++) {{
        m = max(m, inp[params.off_in + (o * params.k + k) * params.inner + r]);
    }}
    var sum = {t}(0.0);
    for (var k = 0u; k < params.k; k++) {{
        let e = exp(inp[params.off_in + (o * params.k + k) * params.inner + r] - m);
        outp[params.off_out + (o * params.k + k) * params.inner + r] = e;
        sum += e;
    }}
    let inv = {t}(1.0) / sum;
    for (var k = 0u; k < params.k; k++) {{
        let idx = params.off_out + (o * params.k + k) * params.inner + r;
        outp[idx] = outp[idx] * inv;
    }}
}}
"#
    ));
    s
}

/// Applies the fused steps to `v`, reading each extra operand at `index`.
fn epilogue_body(steps: &[ChainStep], index: &str) -> String {
    let mut s = String::new();
    for step in steps {
        match step {
            ChainStep::Unary(op) => s.push_str(&format!("    v = op_{op}(v);\n")),
            ChainStep::Binary { op, rhs, swapped } => {
                let i = rhs - 1;
                s.push_str(&format!(
                    "    let e{i} = extra{i}[params.off_extra[{i}u] + select(0u, {index}, params.mode_extra[{i}u] != 0u)];\n"
                ));
                if *swapped {
                    s.push_str(&format!("    v = op_{op}(e{i}, v);\n"));
                } else {
                    s.push_str(&format!("    v = op_{op}(v, e{i});\n"));
                }
            }
        }
    }
    s
}

/// Names a generated program by its kernel and fused epilogue.
pub fn program_key(kind: &str, dt: ShaderDtype, steps: &[ChainStep], extras: usize) -> String {
    let mut key = format!("{kind}_{}_{extras}", dt.suffix());
    for step in steps {
        match step {
            ChainStep::Unary(op) => key.push_str(&format!("_{op}")),
            ChainStep::Binary { op, rhs, swapped } => {
                key.push_str(&format!("_{op}{rhs}{}", if *swapped { "r" } else { "" }))
            }
        }
    }
    key
}

fn matmul_wgsl(dt: ShaderDtype) -> String {
    matmul_module(dt, &[], 0)
}

/// `extras` epilogue operands bind between B and the output.
pub fn matmul_module(dt: ShaderDtype, epilogue: &[ChainStep], extras: usize) -> String {
    let t = dt.wgsl();
    let suf = dt.suffix();
    let out_binding = 2 + extras;
    let uniform_binding = 3 + extras;
    let extra_bindings = (0..extras)
        .map(|i| {
            format!("@group(0) @binding({}) var<storage, read> extra{i}: array<{t}>;\n", 2 + i)
        })
        .collect::<String>();
    let lib = if epilogue.is_empty() {
        String::new()
    } else {
        format!("{}{}", unary_ops_wgsl(t), binary_ops_wgsl(t))
    };
    let epilogue = epilogue_body(epilogue, "col");
    let mut s = preamble(dt);
    s.push_str(&format!(
        r#"
struct Params {{
    off_a: u32,
    off_b: u32,
    off_out: u32,
    prefix: u32,
    m: u32,
    k: u32,
    n: u32,
    ta: u32,
    tb: u32,
    tc: u32,
    _p0: u32,
    _p1: u32,
    a_s0: vec4<u32>,
    a_s1: vec4<u32>,
    b_s0: vec4<u32>,
    b_s1: vec4<u32>,
    out_s0: vec4<u32>,
    out_s1: vec4<u32>,
    out_sh0: vec4<u32>,
    out_sh1: vec4<u32>,
    off_extra: vec4<u32>,
    mode_extra: vec4<u32>,
}}

@group(0) @binding(0) var<storage, read> a: array<{t}>;
@group(0) @binding(1) var<storage, read> b: array<{t}>;
{extra_bindings}@group(0) @binding({out_binding}) var<storage, read_write> outp: array<{t}>;
@group(0) @binding({uniform_binding}) var<uniform> params: Params;
{lib}
{AT8}

@compute @workgroup_size({WORKGROUP})
fn matmul_{suf}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    let mn = params.m * params.n;
    let nout = params.prefix * mn;
    if (i >= nout) {{ return; }}
    let pref = i / mn;
    let rest = i % mn;
    var row: u32;
    var col: u32;
    if (params.tc != 0u) {{
        row = rest % params.m;
        col = rest / params.m;
    }} else {{
        col = rest % params.n;
        row = rest / params.n;
    }}
    // Shapes and strides are right-aligned in the 8 slots: the two matmul axes
    // land in slots 6 and 7, prefix axes in 0..5, padded with dim 1 stride 0.
    // `pref` is row-major over the prefix, so decode from the fastest axis up.
    var a_i = params.off_a;
    var b_i = params.off_b;
    var o_i = params.off_out;
    var restp = pref;
    for (var j = 0u; j < 6u; j++) {{
        let ax = 5u - j;
        let dim = max(at8(params.out_sh0, params.out_sh1, ax), 1u);
        let c = restp % dim;
        restp = restp / dim;
        a_i += c * at8(params.a_s0, params.a_s1, ax);
        b_i += c * at8(params.b_s0, params.b_s1, ax);
        o_i += c * at8(params.out_s0, params.out_s1, ax);
    }}
    let a_row_s = select(at8(params.a_s0, params.a_s1, 6u), at8(params.a_s0, params.a_s1, 7u), params.ta != 0u);
    let a_col_s = select(at8(params.a_s0, params.a_s1, 7u), at8(params.a_s0, params.a_s1, 6u), params.ta != 0u);
    let b_row_s = select(at8(params.b_s0, params.b_s1, 6u), at8(params.b_s0, params.b_s1, 7u), params.tb != 0u);
    let b_col_s = select(at8(params.b_s0, params.b_s1, 7u), at8(params.b_s0, params.b_s1, 6u), params.tb != 0u);
    var acc = 0.0;
    for (var kk = 0u; kk < params.k; kk++) {{
        let av = f32(a[a_i + row * a_row_s + kk * a_col_s]);
        let bv = f32(b[b_i + kk * b_row_s + col * b_col_s]);
        acc += av * bv;
    }}
    let o_row_s = select(at8(params.out_s0, params.out_s1, 6u), at8(params.out_s0, params.out_s1, 7u), params.tc != 0u);
    let o_col_s = select(at8(params.out_s0, params.out_s1, 7u), at8(params.out_s0, params.out_s1, 6u), params.tc != 0u);
    var v = {t}(acc);
{epilogue}
    outp[o_i + row * o_row_s + col * o_col_s] = v;
}}
"#
    ));
    s
}

/// A single-channel tensor into a texture the caller can sample: the mask
/// leaves the graph as a GPU resource rather than as bytes on the host.
fn export_wgsl(dt: ShaderDtype) -> String {
    let t = dt.wgsl();
    let suf = dt.suffix();
    let mut s = preamble(dt);
    s.push_str(&format!(
        r#"
struct Params {{
    off_in: u32,
    width: u32,
    height: u32,
    _p: u32,
}}

@group(0) @binding(0) var<storage, read> inp: array<{t}>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size({WORKGROUP})
fn export_{suf}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    let n = params.width * params.height;
    if (i >= n) {{ return; }}
    let x = i % params.width;
    let y = i / params.width;
    let v = clamp(f32(inp[params.off_in + i]), 0.0, 1.0);
    textureStore(dst, vec2<i32>(i32(x), i32(y)), vec4<f32>(v, v, v, 1.0));
}}
"#
    ));
    s
}

fn ingest_wgsl(dt: ShaderDtype) -> String {
    let _ = dt;
    format!(
        r#"
struct Params {{
    off_out: u32,
    width: u32,
    height: u32,
    _p: u32,
}}

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> outp: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size({WORKGROUP})
fn rgba_to_nchw_f32(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    let n = params.width * params.height;
    if (i >= n) {{ return; }}
    let x = i % params.width;
    let y = i / params.width;
    let px = textureLoad(src, vec2<i32>(i32(x), i32(y)), 0);
    let hw = params.width * params.height;
    let o = params.off_out + i;
    outp[o] = px.r;
    outp[o + hw] = px.g;
    outp[o + 2u * hw] = px.b;
}}
"#
    )
}

pub fn pack_u32s(vals: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 4);
    for v in vals {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

pub fn pad8_u32(xs: &[usize]) -> [u32; 8] {
    let mut out = [1u32; 8];
    for (i, v) in xs.iter().take(8).enumerate() {
        out[i] = *v as u32;
    }
    out
}

pub fn pad8_stride(xs: &[isize]) -> [u32; 8] {
    let mut out = [0u32; 8];
    for (i, s) in xs.iter().take(8).enumerate() {
        out[i] = (*s).max(0) as u32;
    }
    out
}

/// Right-aligned into the 8 slots: axis `i` of a rank-`r` tensor lands in slot
/// `8 - r + i`, so a matmul's two trailing axes always sit in slots 6 and 7 and
/// the prefix in 0..6. Absent axes read as dim 1.
pub fn rpad8_dims(shape: &[usize]) -> [u32; 8] {
    let mut out = [1u32; 8];
    let base = 8 - shape.len().min(8);
    for (i, d) in shape.iter().take(8).enumerate() {
        out[base + i] = *d as u32;
    }
    out
}

/// [`rpad8_dims`] for strides: absent axes read as stride 0, and so do axes this
/// tensor broadcasts over (`shape[i] == 1 && out_shape[i] != 1`).
pub fn rpad8_strides(shape: &[usize], strides: &[isize], out_shape: &[usize]) -> [u32; 8] {
    let mut out = [0u32; 8];
    let base = 8 - shape.len().min(8);
    for i in 0..shape.len().min(8) {
        if shape[i] == 1 && out_shape.get(i).is_some_and(|d| *d != 1) {
            continue;
        }
        out[base + i] = strides[i].max(0) as u32;
    }
    out
}

/// Strides in elements, zeroed on broadcast axes (`shape[i] == 1 && out[i] != 1`).
pub fn broadcast_strides(shape: &[usize], strides: &[isize], out_shape: &[usize]) -> [u32; 8] {
    let mut out = [0u32; 8];
    for i in 0..out_shape.len().min(8) {
        if i < shape.len() && shape[i] == 1 && out_shape[i] != 1 {
            out[i] = 0;
        } else if i < strides.len() {
            out[i] = strides[i].max(0) as u32;
        }
    }
    out
}
