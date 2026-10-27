//! WGSL sources for Phase 1 kernels.
//!
//! Flattened 1-D indexing: `dispatch_workgroups` is 3-D, tract tensors are
//! routinely 4-D+, so shaders take a linear id and apply strides from uniforms.
//! Byte offsets live in uniforms so device views work without 256-byte-aligned
//! `GPUBufferBinding.offset` (the whole storage buffer is bound at 0).
//!
//! Structure follows tfjs-backend-webgpu unary/binary (Apache-2.0): one entry
//! point per op, tightly packed storage arrays. Not a transcription of their
//! TypeScript generators.

use tract_core::internal::*;

pub const WORKGROUP: u32 = 64;
/// Threads per workgroup of the blocked matmul.
pub const MATMUL_BLOCKED_WG: u32 = 192;

/// Threads in a cooperative sum: each output gets one workgroup this wide.
pub const SUM_RUN_WG: u32 = 256;
/// Steps of each of the gemv pair's reductions taken per iteration.
pub const GEMV_PAIR_UNROLL: usize = 10;

/// The fused squeeze-excitation gate runs as one workgroup of this width.
pub const GEMV_PAIR_WG: u32 = 256;

/// WebGPU's cap on workgroups per dispatch dimension. A kernel that walks a
/// tensor linearly takes the overflow on y and folds it back into its index.
pub const GRID_LIMIT: u32 = 65535;

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
    Transpose,
    Reduce,
    Pool,
    Cast,
    Conv,
    Deconv,
    Resize,
    Softmax,
    Ingest,
    IngestBuffer,
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
            | ModuleKind::Transpose
            | ModuleKind::Reduce
            | ModuleKind::Pool
            | ModuleKind::Cast
            | ModuleKind::Softmax => LayoutKind::Unary,
            ModuleKind::Binary | ModuleKind::Conv | ModuleKind::Deconv | ModuleKind::MatMul => {
                LayoutKind::Binary
            }
            ModuleKind::Resize => LayoutKind::Resize,
            ModuleKind::Ingest => LayoutKind::Ingest,
            ModuleKind::IngestBuffer => LayoutKind::Unary,
            ModuleKind::Export => LayoutKind::Export,
        }
    }

    pub fn wgsl(self) -> String {
        match self.kind {
            ModuleKind::ElementWise => element_wise_wgsl(self.dtype),
            ModuleKind::Binary => binary_wgsl(self.dtype),
            ModuleKind::Copy => copy_wgsl(self.dtype),
            ModuleKind::Transpose => transpose_wgsl(self.dtype),
            ModuleKind::Reduce => reduce_wgsl(self.dtype),
            ModuleKind::Pool => pool_wgsl(self.dtype),
            ModuleKind::Cast => cast_wgsl(self.dtype),
            ModuleKind::Conv => conv_wgsl(self.dtype),
            ModuleKind::Deconv => deconv_wgsl(self.dtype),
            ModuleKind::Resize => resize_wgsl(self.dtype),
            ModuleKind::Softmax => softmax_wgsl(self.dtype),
            ModuleKind::Ingest => ingest_wgsl(self.dtype),
            ModuleKind::IngestBuffer => include_str!("ingest_buffer.wgsl").to_string(),
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
fn op_exp(x: {t}) -> {t} {{ return {t}(exp(f32(x))); }}
fn op_ln(x: {t}) -> {t} {{ return {t}(log(f32(x))); }}
fn op_sqrt(x: {t}) -> {t} {{ return {t}(sqrt(f32(x))); }}
fn op_rsqrt(x: {t}) -> {t} {{ return {t}(inverseSqrt(f32(x))); }}
fn op_sigmoid(x: {t}) -> {t} {{
    let e = exp(-abs(f32(x)));
    return {t}(select(1.0, e, x < {t}(0.0)) / (1.0 + e));
}}
fn op_square(x: {t}) -> {t} {{ return x * x; }}
fn op_recip(x: {t}) -> {t} {{ return {t}(1.0) / x; }}
fn op_ceil(x: {t}) -> {t} {{ return ceil(x); }}
fn op_floor(x: {t}) -> {t} {{ return floor(x); }}
fn op_round(x: {t}) -> {t} {{ return sign(x) * floor(abs(x) + {t}(0.5)); }}
fn op_roundhalftoeven(x: {t}) -> {t} {{ return round(x); }}
fn op_cos(x: {t}) -> {t} {{ return {t}(cos(f32(x))); }}
fn op_sin(x: {t}) -> {t} {{ return {t}(sin(f32(x))); }}
fn op_tan(x: {t}) -> {t} {{ return {t}(tan(f32(x))); }}
fn op_acos(x: {t}) -> {t} {{ return {t}(acos(f32(x))); }}
fn op_asin(x: {t}) -> {t} {{ return {t}(asin(f32(x))); }}
fn op_atan(x: {t}) -> {t} {{ return {t}(atan(f32(x))); }}
fn op_cosh(x: {t}) -> {t} {{ return {t}(cosh(f32(x))); }}
fn op_sinh(x: {t}) -> {t} {{ return {t}(sinh(f32(x))); }}
fn op_tanh(x: {t}) -> {t} {{ return {t}(tanh(clamp(f32(x), -15.0, 15.0))); }}
fn op_acosh(x: {t}) -> {t} {{ return {t}(acosh(f32(x))); }}
fn op_asinh(x: {t}) -> {t} {{ return {t}(asinh(f32(x))); }}
fn op_atanh(x: {t}) -> {t} {{ return {t}(atanh(f32(x))); }}
fn op_erf(x: {t}) -> {t} {{ return {t}(approx_erf(f32(x))); }}
fn op_neg(x: {t}) -> {t} {{ return -x; }}
fn op_sign(x: {t}) -> {t} {{ return sign(x); }}
fn op_hardswish(x: {t}) -> {t} {{
    let h = max({t}(0.0), min({t}(1.0), x / {t}(6.0) + {t}(0.5)));
    return x * h;
}}
fn op_silu(x: {t}) -> {t} {{
    let xf = f32(x);
    return {t}(xf / (1.0 + exp(-xf)));
}}
fn op_geluexact(x: {t}) -> {t} {{
    let xf = f32(x);
    return {t}(0.5 * xf * (1.0 + approx_erf(xf * 0.70710678)));
}}
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
fn op_pow(a: {t}, b: {t}) -> {t} {{ return {t}(pow(f32(a), f32(b))); }}
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
    let i = gid.y * {GRID_LIMIT}u * {WORKGROUP}u + gid.x;
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
    "geluexact",
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
    let i = gid.y * {GRID_LIMIT}u * {WORKGROUP}u + gid.x;
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
    let i = gid.y * {GRID_LIMIT}u * {WORKGROUP}u + gid.x;
    if (i * 4u >= params.len) {{ return; }}
    store4(i, op_{name}_v(lhs4(i), rhs4(i)));
}}

@compute @workgroup_size({WORKGROUP})
fn {name}_v4s_{suf}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.y * {GRID_LIMIT}u * {WORKGROUP}u + gid.x;
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
    let i = gid.y * {GRID_LIMIT}u * {WORKGROUP}u + gid.x;
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

/// A batch of 2-D transposes as `tract_gpu::utils::TransposeLaunch` describes
/// them: each workgroup stages one `TRANSPOSE_TILE` square through workgroup
/// memory, reading it along the input's unit-stride rows and writing it along
/// the output's unit-stride columns. Tiles are numbered columns first, then
/// rows, then batch, and the numbering folds onto the grid's x and y.
fn transpose_wgsl(dt: ShaderDtype) -> String {
    use tract_gpu::utils::{TRANSPOSE_ROWS, TRANSPOSE_TILE};
    let t = dt.wgsl();
    let suf = dt.suffix();
    let mut s = preamble(dt);
    s.push_str(&format!(
        r#"
struct Params {{
    off_in: u32,
    off_out: u32,
    rows: u32,
    cols: u32,
    in_batch_stride: u32,
    in_col_stride: u32,
    out_batch_stride: u32,
    out_row_stride: u32,
    col_tiles: u32,
    row_tiles: u32,
    tiles: u32,
}}

@group(0) @binding(0) var<storage, read> inp: array<{t}>;
@group(0) @binding(1) var<storage, read_write> outp: array<{t}>;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> tile: array<array<{t}, {TRANSPOSE_TILE_PADDED}>, {TRANSPOSE_TILE}>;

@compute @workgroup_size({TRANSPOSE_TILE}, {TRANSPOSE_ROWS})
fn copy_transpose_{suf}(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{
    let t = wid.y * {GRID_LIMIT}u + wid.x;
    if (t >= params.tiles) {{ return; }}
    let c0 = (t % params.col_tiles) * {TRANSPOSE_TILE}u;
    let r0 = (t / params.col_tiles % params.row_tiles) * {TRANSPOSE_TILE}u;
    let b = t / params.col_tiles / params.row_tiles;
    let base_in = params.off_in + b * params.in_batch_stride;
    let base_out = params.off_out + b * params.out_batch_stride;
    for (var j = lid.y; j < {TRANSPOSE_TILE}u; j += {TRANSPOSE_ROWS}u) {{
        let r = r0 + lid.x;
        let c = c0 + j;
        if (r < params.rows && c < params.cols) {{
            tile[j][lid.x] = inp[base_in + r + c * params.in_col_stride];
        }}
    }}
    workgroupBarrier();
    for (var j = lid.y; j < {TRANSPOSE_TILE}u; j += {TRANSPOSE_ROWS}u) {{
        let r = r0 + j;
        let c = c0 + lid.x;
        if (r < params.rows && c < params.cols) {{
            outp[base_out + r * params.out_row_stride + c] = tile[lid.x][j];
        }}
    }}
}}
"#,
        TRANSPOSE_TILE_PADDED = TRANSPOSE_TILE + 1,
    ));
    s
}

/// A sum over the trailing values of each output: one workgroup per output,
/// threads striding over its `k` values, then a tree over workgroup memory.
pub fn sum_run_module(dt: ShaderDtype) -> String {
    let t = dt.wgsl();
    let suf = dt.suffix();
    let mut s = preamble(dt);
    s.push_str(&format!(
        r#"
struct Params {{
    off_in: u32,
    off_out: u32,
    k: u32,
    _p0: u32,
}}

@group(0) @binding(0) var<storage, read> inp: array<{t}>;
@group(0) @binding(1) var<storage, read_write> outp: array<{t}>;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> partial: array<f32, {SUM_RUN_WG}>;

@compute @workgroup_size({SUM_RUN_WG})
fn sum_run_{suf}(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{
    let base = params.off_in + wid.x * params.k;
    var acc = 0.0;
    for (var i = lid.x; i < params.k; i += {SUM_RUN_WG}u) {{
        acc += f32(inp[base + i]);
    }}
    partial[lid.x] = acc;
    for (var stride = {SUM_RUN_WG}u / 2u; stride > 0u; stride /= 2u) {{
        workgroupBarrier();
        if (lid.x < stride) {{
            partial[lid.x] += partial[lid.x + stride];
        }}
    }}
    if (lid.x == 0u) {{
        outp[params.off_out + wid.x] = {t}(partial[0]);
    }}
}}
"#
    ));
    s
}

/// Threads of each group norm statistics and finalize workgroup.
pub const GROUP_NORM_WG: u32 = 256;

/// Chan's merge of `(count, mean, m2)` triples, and a tree of it over
/// workgroup memory whose result every thread reads.
fn moments_lib() -> String {
    format!(
        r#"
struct Moments {{
    n: f32,
    mean: f32,
    m2: f32,
}}

fn merge_moments(a: Moments, b: Moments) -> Moments {{
    let n = a.n + b.n;
    if (n == 0.0) {{ return a; }}
    let d = b.mean - a.mean;
    let f = b.n / n;
    return Moments(n, a.mean + d * f, a.m2 + b.m2 + d * d * a.n * f);
}}

fn shifted_moments(n: f32, shift: f32, s1: f32, s2: f32) -> Moments {{
    if (n == 0.0) {{ return Moments(0.0, 0.0, 0.0); }}
    return Moments(n, shift + s1 / n, s2 - s1 * s1 / n);
}}

var<workgroup> wg_moments: array<Moments, {GROUP_NORM_WG}>;

fn workgroup_merge_moments(lid: u32, m: Moments) -> Moments {{
    wg_moments[lid] = m;
    for (var stride = {GROUP_NORM_WG}u / 2u; stride > 0u; stride /= 2u) {{
        workgroupBarrier();
        if (lid < stride) {{
            wg_moments[lid] = merge_moments(wg_moments[lid], wg_moments[lid + stride]);
        }}
    }}
    workgroupBarrier();
    return wg_moments[0];
}}
"#
    )
}

/// Partial `(count, mean, m2)` statistics of a group norm input, see
/// `tract_gpu::ops::group_norm::GroupNormParams`. Channel-first, workgroup `x`
/// is a group and `y` a chunk of it; channel-last, each thread walks one
/// channel over a chunk of the positions.
pub fn group_norm_stats_module(dt: ShaderDtype) -> String {
    let t = dt.wgsl();
    let suf = dt.suffix();
    let mut s = preamble(dt);
    s.push_str(&moments_lib());
    s.push_str(&format!(
        r#"
struct Params {{
    off_in: u32,
    off_partials: u32,
    pre: u32,
    channels: u32,
    post: u32,
    groups: u32,
    splits: u32,
    _p0: u32,
}}

@group(0) @binding(0) var<storage, read> inp: array<{t}>;
@group(0) @binding(1) var<storage, read_write> partials: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

fn store_moments(at: u32, m: Moments) {{
    partials[params.off_partials + at * 3u] = m.n;
    partials[params.off_partials + at * 3u + 1u] = m.mean;
    partials[params.off_partials + at * 3u + 2u] = m.m2;
}}

@compute @workgroup_size({GROUP_NORM_WG})
fn group_norm_stats_{suf}(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{
    let o = wid.x / params.groups;
    let g = wid.x % params.groups;
    let cg = params.channels / params.groups;
    let len = params.pre * cg * params.post;
    let chunk = (len + params.splits - 1u) / params.splits;
    let start = wid.y * chunk;
    let end = min(len, start + chunk);
    var shift = 0.0;
    var s1 = 0.0;
    var s2 = 0.0;
    var n = 0.0;
    for (var j = start + lid.x; j < end; j += {GROUP_NORM_WG}u) {{
        let q = j % params.post;
        let t = j / params.post;
        let v = f32(inp[params.off_in + ((o * params.pre + t / cg) * params.channels + g * cg + t % cg) * params.post + q]);
        if (n == 0.0) {{ shift = v; }}
        s1 += v - shift;
        s2 += (v - shift) * (v - shift);
        n += 1.0;
    }}
    let m = workgroup_merge_moments(lid.x, shifted_moments(n, shift, s1, s2));
    if (lid.x == 0u) {{
        store_moments(wid.x * params.splits + wid.y, m);
    }}
}}

@compute @workgroup_size({GROUP_NORM_WG})
fn group_norm_stats_channel_last_{suf}(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{
    let channel_blocks = (params.channels + {GROUP_NORM_WG}u - 1u) / {GROUP_NORM_WG}u;
    let o = wid.x / channel_blocks;
    let c = (wid.x % channel_blocks) * {GROUP_NORM_WG}u + lid.x;
    if (c >= params.channels) {{ return; }}
    let chunk = (params.pre + params.splits - 1u) / params.splits;
    let start = wid.y * chunk;
    let end = min(params.pre, start + chunk);
    var shift = 0.0;
    var s1 = 0.0;
    var s2 = 0.0;
    var n = 0.0;
    for (var p = start; p < end; p++) {{
        let v = f32(inp[params.off_in + (o * params.pre + p) * params.channels + c]);
        if (n == 0.0) {{ shift = v; }}
        s1 += v - shift;
        s2 += (v - shift) * (v - shift);
        n += 1.0;
    }}
    store_moments((o * params.channels + c) * params.splits + wid.y, shifted_moments(n, shift, s1, s2));
}}
"#
    ));
    s
}

/// Merges each group's partial statistics into its mean and inverse standard
/// deviation, one workgroup per group.
pub fn group_norm_finalize_module() -> String {
    let mut s = moments_lib();
    s.push_str(
        r#"
struct Params {
    off_partials: u32,
    off_stats: u32,
    entries: u32,
    eps: f32,
}

@group(0) @binding(0) var<storage, read> partials: array<f32>;
@group(0) @binding(1) var<storage, read_write> stats: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

"#,
    );
    s.push_str(&format!(
        r#"
@compute @workgroup_size({GROUP_NORM_WG})
fn group_norm_finalize(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{
    let base = params.off_partials + wid.x * params.entries * 3u;
    var m = Moments(0.0, 0.0, 0.0);
    for (var e = lid.x; e < params.entries; e += {GROUP_NORM_WG}u) {{
        m = merge_moments(m, Moments(partials[base + 3u * e], partials[base + 3u * e + 1u], partials[base + 3u * e + 2u]));
    }}
    m = workgroup_merge_moments(lid.x, m);
    if (lid.x == 0u) {{
        stats[params.off_stats + wid.x * 2u] = m.mean;
        stats[params.off_stats + wid.x * 2u + 1u] = inverseSqrt(m.m2 / m.n + params.eps);
    }}
}}
"#
    ));
    s
}

/// Normalizes each element with its group's `(mean, inverse std)`.
pub fn group_norm_apply_module(dt: ShaderDtype) -> String {
    let t = dt.wgsl();
    let suf = dt.suffix();
    let mut s = preamble(dt);
    s.push_str(&format!(
        r#"
struct Params {{
    off_in: u32,
    off_stats: u32,
    off_out: u32,
    len: u32,
    pre: u32,
    channels: u32,
    post: u32,
    groups: u32,
}}

@group(0) @binding(0) var<storage, read> inp: array<{t}>;
@group(0) @binding(1) var<storage, read> stats: array<f32>;
@group(0) @binding(2) var<storage, read_write> outp: array<{t}>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size({WORKGROUP})
fn group_norm_apply_{suf}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.y * {GRID_LIMIT}u * {WORKGROUP}u + gid.x;
    if (i >= params.len) {{ return; }}
    let c = (i / params.post) % params.channels;
    let o = i / (params.pre * params.channels * params.post);
    let at = params.off_stats + (o * params.groups + c / (params.channels / params.groups)) * 2u;
    outp[params.off_out + i] = {t}((f32(inp[params.off_in + i]) - stats[at]) * stats[at + 1u]);
}}
"#
    ));
    s
}

/// Threads of each layer norm workgroup.
pub const LAYER_NORM_WG: u32 = 256;

/// Layer norm over the middle axis of an `(outer, k, inner)` view, one
/// workgroup per normalized row. `affine` multiplies by an F32 scale and adds
/// an F32 bias along that axis.
pub fn layer_norm_module(dt: ShaderDtype, affine: bool) -> String {
    let t = dt.wgsl();
    let suf = dt.suffix();
    let (bindings, apply) = if affine {
        (
            format!(
                "@group(0) @binding(0) var<storage, read> inp: array<{t}>;
@group(0) @binding(1) var<storage, read> scale: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> outp: array<{t}>;
@group(0) @binding(4) var<uniform> params: Params;"
            ),
            "y = y * scale[params.off_scale + i] + bias[params.off_bias + i];",
        )
    } else {
        (
            format!(
                "@group(0) @binding(0) var<storage, read> inp: array<{t}>;
@group(0) @binding(1) var<storage, read_write> outp: array<{t}>;
@group(0) @binding(2) var<uniform> params: Params;"
            ),
            "",
        )
    };
    let mut s = preamble(dt);
    s.push_str(&format!(
        r#"
struct Params {{
    off_in: u32,
    off_out: u32,
    off_scale: u32,
    off_bias: u32,
    k: u32,
    inner: u32,
    rows: u32,
    eps: f32,
}}

{bindings}

var<workgroup> partial: array<f32, {LAYER_NORM_WG}>;

fn workgroup_sum(lid: u32, v: f32) -> f32 {{
    partial[lid] = v;
    for (var stride = {LAYER_NORM_WG}u / 2u; stride > 0u; stride /= 2u) {{
        workgroupBarrier();
        if (lid < stride) {{
            partial[lid] += partial[lid + stride];
        }}
    }}
    workgroupBarrier();
    let total = partial[0];
    workgroupBarrier();
    return total;
}}

@compute @workgroup_size({LAYER_NORM_WG})
fn layer_norm_{suf}(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{
    let row = wid.y * {GRID_LIMIT}u + wid.x;
    if (row >= params.rows) {{ return; }}
    let base = (row / params.inner) * params.k * params.inner + row % params.inner;
    var sum = 0.0;
    for (var i = lid.x; i < params.k; i += {LAYER_NORM_WG}u) {{
        sum += f32(inp[params.off_in + base + i * params.inner]);
    }}
    let mean = workgroup_sum(lid.x, sum) / f32(params.k);
    var sq = 0.0;
    for (var i = lid.x; i < params.k; i += {LAYER_NORM_WG}u) {{
        let d = f32(inp[params.off_in + base + i * params.inner]) - mean;
        sq += d * d;
    }}
    let inv = inverseSqrt(workgroup_sum(lid.x, sq) / f32(params.k) + params.eps);
    for (var i = lid.x; i < params.k; i += {LAYER_NORM_WG}u) {{
        var y = (f32(inp[params.off_in + base + i * params.inner]) - mean) * inv;
        {apply}
        outp[params.off_out + base + i * params.inner] = {t}(y);
    }}
}}
"#
    ));
    s
}

/// RMS norm over the middle axis of an `(outer, k, inner)` view, one workgroup
/// per normalized row, reading `input` and writing `output` dtypes. `scaled`
/// multiplies by an F32 scale along that axis; `round_f16` first rounds the
/// normalized value to f16, as a graph materializing it in f16 would.
pub fn rms_norm_module(
    input: ShaderDtype,
    output: ShaderDtype,
    scaled: bool,
    round_f16: bool,
) -> String {
    let (ti, to) = (input.wgsl(), output.wgsl());
    let bindings = if scaled {
        format!(
            "@group(0) @binding(0) var<storage, read> inp: array<{ti}>;
@group(0) @binding(1) var<storage, read> scale: array<f32>;
@group(0) @binding(2) var<storage, read_write> outp: array<{to}>;
@group(0) @binding(3) var<uniform> params: Params;"
        )
    } else {
        format!(
            "@group(0) @binding(0) var<storage, read> inp: array<{ti}>;
@group(0) @binding(1) var<storage, read_write> outp: array<{to}>;
@group(0) @binding(2) var<uniform> params: Params;"
        )
    };
    let round =
        if round_f16 { "y = unpack2x16float(pack2x16float(vec2<f32>(y, 0.0))).x;" } else { "" };
    let apply = if scaled { "y = y * scale[params.off_scale + i];" } else { "" };
    let enable =
        if input == ShaderDtype::F16 || output == ShaderDtype::F16 { "enable f16;\n" } else { "" };
    format!(
        r#"{enable}
struct Params {{
    off_in: u32,
    off_out: u32,
    off_scale: u32,
    k: u32,
    inner: u32,
    rows: u32,
    eps: f32,
}}

{bindings}

var<workgroup> partial: array<f32, {LAYER_NORM_WG}>;

@compute @workgroup_size({LAYER_NORM_WG})
fn rms_norm(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{
    let row = wid.y * {GRID_LIMIT}u + wid.x;
    if (row >= params.rows) {{ return; }}
    let base = (row / params.inner) * params.k * params.inner + row % params.inner;
    var sq = 0.0;
    for (var i = lid.x; i < params.k; i += {LAYER_NORM_WG}u) {{
        let x = f32(inp[params.off_in + base + i * params.inner]);
        sq += x * x;
    }}
    partial[lid.x] = sq;
    for (var stride = {LAYER_NORM_WG}u / 2u; stride > 0u; stride /= 2u) {{
        workgroupBarrier();
        if (lid.x < stride) {{
            partial[lid.x] += partial[lid.x + stride];
        }}
    }}
    workgroupBarrier();
    let inv = inverseSqrt(partial[0] / f32(params.k) + params.eps);
    for (var i = lid.x; i < params.k; i += {LAYER_NORM_WG}u) {{
        var y = f32(inp[params.off_in + base + i * params.inner]) * inv;
        {round}
        {apply}
        outp[params.off_out + base + i * params.inner] = {to}(y);
    }}
}}
"#
    )
}

/// `softmax(x * scale + mask)` over the innermost axis of a rank-5 view, one
/// workgroup per row. The input and mask are read through their strides (the
/// mask's zero along the axes it broadcasts); the output is contiguous.
pub fn scaled_masked_softmax_module(dt: ShaderDtype) -> String {
    let t = dt.wgsl();
    let suf = dt.suffix();
    let mut s = preamble(dt);
    s.push_str(&format!(
        r#"
struct Params {{
    off_in: u32,
    off_mask: u32,
    off_out: u32,
    d1: u32,
    d2: u32,
    d3: u32,
    k: u32,
    rows: u32,
    scale: f32,
    x_inner: u32,
    mask_inner: u32,
    _pad: u32,
    x_strides: vec4<u32>,
    mask_strides: vec4<u32>,
}}

@group(0) @binding(0) var<storage, read> inp: array<{t}>;
@group(0) @binding(1) var<storage, read> mask: array<{t}>;
@group(0) @binding(2) var<storage, read_write> outp: array<{t}>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> partial: array<f32, {LAYER_NORM_WG}>;

fn logit(x: u32, m: u32, i: u32) -> f32 {{
    return f32(inp[x + i * params.x_inner]) * params.scale + f32(mask[m + i * params.mask_inner]);
}}

@compute @workgroup_size({LAYER_NORM_WG})
fn scaled_masked_softmax_{suf}(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{
    let row = wid.y * {GRID_LIMIT}u + wid.x;
    if (row >= params.rows) {{ return; }}
    let at = vec4<u32>(
        row / (params.d1 * params.d2 * params.d3),
        (row / (params.d2 * params.d3)) % params.d1,
        (row / params.d3) % params.d2,
        row % params.d3,
    );
    let x = params.off_in + dot(at, params.x_strides);
    let m = params.off_mask + dot(at, params.mask_strides);
    let o = params.off_out + row * params.k;
    var hi = -3.4028235e38;
    for (var i = lid.x; i < params.k; i += {LAYER_NORM_WG}u) {{
        hi = max(hi, logit(x, m, i));
    }}
    partial[lid.x] = hi;
    for (var stride = {LAYER_NORM_WG}u / 2u; stride > 0u; stride /= 2u) {{
        workgroupBarrier();
        if (lid.x < stride) {{
            partial[lid.x] = max(partial[lid.x], partial[lid.x + stride]);
        }}
    }}
    workgroupBarrier();
    hi = partial[0];
    workgroupBarrier();
    var sum = 0.0;
    for (var i = lid.x; i < params.k; i += {LAYER_NORM_WG}u) {{
        sum += exp(logit(x, m, i) - hi);
    }}
    partial[lid.x] = sum;
    for (var stride = {LAYER_NORM_WG}u / 2u; stride > 0u; stride /= 2u) {{
        workgroupBarrier();
        if (lid.x < stride) {{
            partial[lid.x] += partial[lid.x + stride];
        }}
    }}
    workgroupBarrier();
    let inv = 1.0 / partial[0];
    for (var i = lid.x; i < params.k; i += {LAYER_NORM_WG}u) {{
        outp[o + i] = {t}(exp(logit(x, m, i) - hi) * inv);
    }}
}}
"#
    ));
    s
}

/// `x` when `x >= 0`, otherwise `alpha * x`. `alpha` is an f32 uniform, cast to the
/// storage type so an f16 run uses the same slope as the f16 CPU kernel.
pub fn leaky_relu_module(dt: ShaderDtype) -> String {
    let t = dt.wgsl();
    let suf = dt.suffix();
    let mut s = preamble(dt);
    s.push_str(&format!(
        r#"
struct Params {{
    off_in: u32,
    off_out: u32,
    len: u32,
    alpha: f32,
}}

@group(0) @binding(0) var<storage, read> inp: array<{t}>;
@group(0) @binding(1) var<storage, read_write> outp: array<{t}>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size({WORKGROUP})
fn leaky_relu_{suf}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.y * {GRID_LIMIT}u * {WORKGROUP}u + gid.x;
    if (i >= params.len) {{ return; }}
    let x = inp[params.off_in + i];
    outp[params.off_out + i] = select(x, x * {t}(params.alpha), x < {t}(0.0));
}}
"#
    ));
    s
}

/// Replaces each NaN of the input with `value`. NaN is told by its bits, on the value widened
/// to f32 (which keeps a NaN a NaN): WGSL lets a compiler assume no NaN and fold `x != x`.
pub fn nan_replace_module(dt: ShaderDtype) -> String {
    let t = dt.wgsl();
    let suf = dt.suffix();
    let mut s = preamble(dt);
    s.push_str(&format!(
        r#"
struct Params {{
    off_in: u32,
    off_out: u32,
    len: u32,
    value: f32,
}}

@group(0) @binding(0) var<storage, read> inp: array<{t}>;
@group(0) @binding(1) var<storage, read_write> outp: array<{t}>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size({WORKGROUP})
fn nan_replace_{suf}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.y * {GRID_LIMIT}u * {WORKGROUP}u + gid.x;
    if (i >= params.len) {{ return; }}
    let x = inp[params.off_in + i];
    let bits = bitcast<u32>(f32(x));
    outp[params.off_out + i] = select(x, {t}(params.value), (bits & 0x7fffffffu) > 0x7f800000u);
}}
"#
    ));
    s
}

pub fn gather_nd_module(dt: ShaderDtype) -> String {
    let t = dt.wgsl();
    let suf = dt.suffix();
    let mut s = preamble(dt);
    s.push_str(&format!(
        r#"
struct Params {{
    off_data: u32,
    off_idx: u32,
    off_out: u32,
    len: u32,
    inner: u32,
    rows: u32,
    n: u32,
    batch_stride: u32,
    dims: vec4<u32>,
    strides: vec4<u32>,
}}

@group(0) @binding(0) var<storage, read> data: array<{t}>;
@group(0) @binding(1) var<storage, read> idx: array<f32>;
@group(0) @binding(2) var<storage, read_write> outp: array<{t}>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size({WORKGROUP})
fn gather_nd_{suf}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.y * {GRID_LIMIT}u * {WORKGROUP}u + gid.x;
    if (i >= params.len) {{ return; }}
    let row = i / params.inner;
    var src = params.off_data + (row / params.rows) * params.batch_stride + i % params.inner;
    for (var k = 0u; k < params.n; k++) {{
        var c = i32(idx[params.off_idx + row * params.n + k]);
        if (c < 0) {{ c += i32(params.dims[k]); }}
        src += u32(c) * params.strides[k];
    }}
    outp[params.off_out + i] = data[src];
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
    let i = gid.y * {GRID_LIMIT}u * {WORKGROUP}u + gid.x;
    let n = params.outer * params.inner;
    if (i >= n) {{ return; }}
    let o = i / params.inner;
    let r = i % params.inner;
    var acc = 0.0;
    for (var k = 0u; k < params.k; k++) {{
        acc += f32(inp[params.off_in + (o * params.k + k) * params.inner + r]);
    }}
    outp[params.off_out + i] = {t}(acc);
}}

@compute @workgroup_size({WORKGROUP})
fn reduce_prod_{suf}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.y * {GRID_LIMIT}u * {WORKGROUP}u + gid.x;
    let n = params.outer * params.inner;
    if (i >= n) {{ return; }}
    let o = i / params.inner;
    let r = i % params.inner;
    var acc = 1.0;
    for (var k = 0u; k < params.k; k++) {{
        acc *= f32(inp[params.off_in + (o * params.k + k) * params.inner + r]);
    }}
    outp[params.off_out + i] = {t}(acc);
}}

@compute @workgroup_size({WORKGROUP})
fn reduce_max_{suf}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.y * {GRID_LIMIT}u * {WORKGROUP}u + gid.x;
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
    let i = gid.y * {GRID_LIMIT}u * {WORKGROUP}u + gid.x;
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
fn reduce_mean_{suf}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.y * {GRID_LIMIT}u * {WORKGROUP}u + gid.x;
    let n = params.outer * params.inner;
    if (i >= n) {{ return; }}
    let o = i / params.inner;
    let r = i % params.inner;
    var acc = 0.0;
    for (var k = 0u; k < params.k; k++) {{
        acc += f32(inp[params.off_in + (o * params.k + k) * params.inner + r]);
    }}
    outp[params.off_out + i] = {t}(acc / f32(params.k));
}}

@compute @workgroup_size({WORKGROUP})
fn reduce_mean_of_squares_{suf}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.y * {GRID_LIMIT}u * {WORKGROUP}u + gid.x;
    let n = params.outer * params.inner;
    if (i >= n) {{ return; }}
    let o = i / params.inner;
    let r = i % params.inner;
    var acc = 0.0;
    for (var k = 0u; k < params.k; k++) {{
        let v = f32(inp[params.off_in + (o * params.k + k) * params.inner + r]);
        acc += v * v;
    }}
    outp[params.off_out + i] = {t}(acc / f32(params.k));
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
var<workgroup> pool_scratch: array<f32, {WORKGROUP}>;

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
fn global_avg_pool_2d_{suf}(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>
) {{
    let nc = i32(group.x);
    if (nc >= params.n * params.c) {{ return; }}
    let c = nc % params.c;
    let n = nc / params.c;
    let spatial = params.ih * params.iw;
    var acc = 0.0;
    for (var i = i32(local.x); i < spatial; i += {WORKGROUP}) {{
        let h = i / params.iw;
        let w = i - h * params.iw;
        var idx: u32;
        if (params.channels_last != 0) {{
            idx = in_idx_nhwc(n, h, w, c);
        }} else {{
            idx = in_idx_nchw(n, c, h, w);
        }}
        acc += f32(inp[params.off_in + idx]);
    }}
    pool_scratch[local.x] = acc;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride >>= 1u) {{
        if (local.x < stride) {{
            pool_scratch[local.x] += pool_scratch[local.x + stride];
        }}
        workgroupBarrier();
    }}
    if (local.x == 0u) {{
        var oidx: u32;
        if (params.channels_last != 0) {{
            oidx = out_idx_nhwc(n, 0, 0, c);
        }} else {{
            oidx = out_idx_nchw(n, c, 0, 0);
        }}
        outp[params.off_out + oidx] = {t}(pool_scratch[0] / f32(spatial));
    }}
}}

@compute @workgroup_size({WORKGROUP})
fn max_pool_2d_{suf}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = i32(gid.y * {GRID_LIMIT}u * {WORKGROUP}u + gid.x);
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
    let i = i32(gid.y * {GRID_LIMIT}u * {WORKGROUP}u + gid.x);
    let nout = params.n * params.oh * params.ow * params.c;
    if (i >= nout) {{ return; }}
    var rest = i;
    let c = rest % params.c; rest = rest / params.c;
    let ow = rest % params.ow; rest = rest / params.ow;
    let oh = rest % params.oh;
    let n = rest / params.oh;
    let h0 = oh * params.stride_h - params.pad_h;
    let w0 = ow * params.stride_w - params.pad_w;
    var acc = 0.0;
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
            acc += f32(inp[params.off_in + idx]);
            count += 1;
        }}
    }}
    if (params.normalize != 0) {{
        var denom = count;
        if (params.count_include_pad != 0) {{
            denom = params.kh * params.kw;
        }}
        if (denom > 0) {{
            acc = acc / f32(denom);
        }}
    }}
    var oidx: u32;
    if (params.channels_last != 0) {{
        oidx = out_idx_nhwc(n, oh, ow, c);
    }} else {{
        oidx = out_idx_nchw(n, c, oh, ow);
    }}
    outp[params.off_out + oidx] = {t}(acc);
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
    let i = gid.y * {GRID_LIMIT}u * {WORKGROUP}u + gid.x;
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
    let i = gid.y * {GRID_LIMIT}u * {WORKGROUP}u + gid.x;
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

/// Workgroup edge when the halo is left to the cache instead of staged.
pub const DW_DIRECT_WG: u32 = 8;

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
    /// Whether to read the input directly rather than stage a halo tile. Taps
    /// this few reuse too little for the staging and its barrier to repay
    /// themselves and the overlap is served from cache; from 7x7 up the tile
    /// wins instead.
    pub fn reads_direct(&self) -> bool {
        self.kh <= 3 && self.kw <= 3
    }

    /// Output values along each edge of one workgroup's tile.
    pub fn wg(&self) -> u32 {
        if self.reads_direct() { DW_DIRECT_WG } else { DW_WG }
    }

    fn tile_h(&self) -> u32 {
        (DW_WG - 1) * self.stride_h + (self.kh - 1) * self.dil_h + 1
    }

    fn tile_w(&self) -> u32 {
        (DW_WG - 1) * self.stride_w + (self.kw - 1) * self.dil_w + 1
    }
}

/// The workgroup arrays the staged body reads through.
fn depthwise_tile_decls(shape: DepthwiseShape) -> String {
    let DepthwiseShape { kh, kw, .. } = shape;
    let (tile_h, tile_w) = (shape.tile_h(), shape.tile_w());
    format!(
        "var<workgroup> x_tile: array<array<f32, {tile_w}>, {tile_h}>;\n\
         var<workgroup> w_tile: array<array<f32, {kw}>, {kh}>;\n\n"
    )
}

/// Every thread stages its share of the workgroup's input halo and the filter
/// in workgroup memory, so each input value is read once rather than once per
/// tap.
fn depthwise_staged_body(shape: DepthwiseShape) -> String {
    let DepthwiseShape { kh, kw, stride_h, stride_w, dil_h, dil_w } = shape;
    let (tile_h, tile_w) = (shape.tile_h(), shape.tile_w());
    let wg = shape.wg();
    format!(
        r#"
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
"#
    )
}

/// Each thread reads its own taps straight from the input, leaving the overlap
/// between neighbours to the cache: with few taps there is too little reuse to
/// repay staging it, and dropping the workgroup arrays and their barrier frees
/// the occupancy that pays instead.
fn depthwise_direct_body(shape: DepthwiseShape) -> String {
    let DepthwiseShape { kh, kw, stride_h, stride_w, dil_h, dil_w } = shape;
    format!(
        r#"
    let oh = oh0 + lid.y;
    let ow = ow0 + lid.x;
    if (oh >= params.oh || ow >= params.ow) {{ return; }}
    var acc = 0.0;
    for (var wr = 0u; wr < {kh}u; wr++) {{
        for (var wc = 0u; wc < {kw}u; wc++) {{
            let ih = ih0 + i32(lid.y) * {stride_h} + i32(wr) * {dil_h};
            let iw = iw0 + i32(lid.x) * {stride_w} + i32(wc) * {dil_w};
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
            let w_idx = params.off_w + u32(
                i32(co) * params.w_so + i32(wr) * params.w_sh + i32(wc) * params.w_sw
            );
            acc = fma(v, f32(wgt[w_idx]), acc);
        }}
    }}
"#
    )
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
    let DepthwiseShape { stride_h, stride_w, .. } = shape;
    let wg = shape.wg();
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
    let (tile_decls, accumulate) = if shape.reads_direct() {
        (String::new(), depthwise_direct_body(shape))
    } else {
        (depthwise_tile_decls(shape), depthwise_staged_body(shape))
    };
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
{tile_decls}@compute @workgroup_size({wg}, {wg}, 1)
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

{accumulate}    var v = {t}(acc);
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
    let i = gid.y * {GRID_LIMIT}u * {WORKGROUP}u + gid.x;
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

/// The `k_info` of a tiled convolution's current `k` slice: for each `k`, the
/// weight offset of its (input channel, kernel row, kernel column), the input
/// offset of its channel, and its row and column displacement in the input.
/// The divisions happen here, sixteen per slice, rather than in every load.
fn conv_k_prologue() -> String {
    let tk = MATMUL_TILE_K;
    format!(
        "        if (lid < {tk}u) {{
            let k = k0 + lid;
            var info = vec4<i32>(0);
            if (k < k_len) {{
                let ci = k / khw;
                let tap = k % khw;
                let kh = tap / params.kw;
                let kw = tap % params.kw;
                info = vec4<i32>(
                    i32(ci) * params.w_si + i32(kh) * params.w_sh + i32(kw) * params.w_sw,
                    i32(g * params.ci_pg + ci) * params.in_sc,
                    i32(kh) * params.dil_h,
                    i32(kw) * params.dil_w,
                );
            }}
            k_info[lid] = info;
        }}
"
    )
}

/// A 2-D convolution as an implicit GEMM, staged through workgroup memory the
/// way [`matmul_tiled_module`] stages a product. Per group, rows are output
/// channels, columns are output pixels (batch, row, column) and the reduction
/// runs over (input channel, kernel row, kernel column) in the weights' OIHW
/// order; the input slice is gathered on load, taps outside the window reading
/// zero. Same `Params`, strides and per-channel epilogue as [`conv_module`].
/// Workgroup `x` tiles the pixels, `y` the output channels of a group, `z` is
/// the group.
pub fn conv_tiled_module(dt: ShaderDtype, epilogue: &[ChainStep], extras: usize) -> String {
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
    let tile = MATMUL_TILE;
    let mut s = preamble(dt);
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

{tiles}
var<workgroup> k_info: array<vec4<i32>, {tk}>;
var<workgroup> pix_info: array<vec4<i32>, {tile}>;

fn ld_a(ctx: vec4<u32>, r: u32, k: u32, guard: bool) -> {t} {{
    if (guard && (r >= params.co_pg || k >= ctx.y)) {{ return {t}(0.0); }}
    let co = ctx.x * params.co_pg + r;
    return wgt[params.off_w + u32(i32(co) * params.w_so + k_info[k % {tk}u].x)];
}}

fn ld_b(ctx: vec4<u32>, k: u32, pix: u32, guard: bool) -> {t} {{
    if (guard && k >= ctx.y) {{ return {t}(0.0); }}
    let ki = k_info[k % {tk}u];
    let p = pix_info[pix % {tile}u];
    let ih = p.y + ki.z;
    let iw = p.z + ki.w;
    if (ih < 0 || ih >= i32(params.ih) || iw < 0 || iw >= i32(params.iw)) {{ return {t}(0.0); }}
    return inp[params.off_in + u32(p.x + ki.y + ih * params.in_sh + iw * params.in_sw)];
}}

fn store_out(g: u32, row: u32, pix: u32, acc: f32) {{
    if (row >= params.co_pg || pix >= params.n * params.oh * params.ow) {{ return; }}
    let co = g * params.co_pg + row;
    let ow = pix % params.ow;
    let rest = pix / params.ow;
    let oh = rest % params.oh;
    let n = rest / params.oh;
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

@compute @workgroup_size({threads})
fn conv2d_tiled_{suf}(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>) {{
    let g = wid.z;
    let khw = params.kh * params.kw;
    let k_len = params.ci_pg * khw;
    let actx = vec4<u32>(g, k_len, 0u, 0u);
    let bctx = actx;
    let npix = params.n * params.oh * params.ow;
    let row0 = wid.y * {tile}u;
    let col0 = wid.x * {tile}u;
    let interior = row0 + {tile}u <= params.co_pg && col0 + {tile}u <= npix;
    if (lid < {tile}u) {{
        let pix = col0 + lid;
        let ow = pix % params.ow;
        let rest = pix / params.ow;
        let oh = rest % params.oh;
        let n = rest / params.oh;
        var info = vec4<i32>(0, -0x40000000, 0, 0);
        if (pix < npix) {{
            info = vec4<i32>(
                i32(n) * params.in_sn,
                i32(oh) * params.stride_h - params.pad_h,
                i32(ow) * params.stride_w - params.pad_w,
                0,
            );
        }}
        pix_info[lid] = info;
    }}
{core}}}
"#,
        tiles = tiled_gemm_tiles(t),
        threads = TILED_THREADS,
        tk = MATMUL_TILE_K,
        core = tiled_gemm_core(t, "true", "true", "g, ", &conv_k_prologue()),
    ));
    s
}

/// Threads per workgroup of the per-pixel convolution.
pub const CONV_PIXEL_WG: u32 = 256;

/// One thread per output pixel computing every output channel, for a dense 3x3
/// convolution over few input channels — the MobileNet stem: the 27 inputs a
/// pixel reads are shared by all its channels, so they are read once per group
/// of four rather than once per output element. The kernel came out of an
/// OpenEvolve search over tract's own conv contract (3.0x on the stem, seed
/// 115us); the taps are unrolled for kh = kw = 3 at dilation 1.
pub fn conv_pixel_module(dt: ShaderDtype, epilogue: &[ChainStep], extras: usize) -> String {
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
    // Same packed params as `conv_module`, see kernels/conv.rs.
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
fn read_inp(ih: i32, iw: i32, ic: u32, n: u32) -> f32 {{
    if (ih < 0 || ih >= i32(params.ih) || iw < 0 || iw >= i32(params.iw)) {{ return 0.0; }}
    return f32(inp[params.off_in + u32(i32(n) * params.in_sn + ih * params.in_sh + iw * params.in_sw + i32(ic) * params.in_sc)]);
}}

@compute @workgroup_size({CONV_PIXEL_WG})
fn conv2d_pixel_{suf}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    let npix = params.n * params.oh * params.ow;
    if (idx >= npix) {{ return; }}
    var rest = idx;
    let ow = rest % params.ow; rest /= params.ow;
    let oh = rest % params.oh; rest /= params.oh;
    let n = rest;
    let ih0 = i32(oh) * params.stride_h - params.pad_h;
    let iw0 = i32(ow) * params.stride_w - params.pad_w;
    // Four channels at a time, in four named accumulators: an indexed private
    // array here lands in memory under Tint, which costs ~20% in Chrome.
    for (var co_grp = 0u; co_grp < params.co; co_grp += 4u) {{
        var acc0 = 0.0;
        var acc1 = 0.0;
        var acc2 = 0.0;
        var acc3 = 0.0;
        for (var ci = 0u; ci < params.ci; ci++) {{
            let i00 = read_inp(ih0, iw0, ci, n);
            let i01 = read_inp(ih0, iw0 + 1, ci, n);
            let i02 = read_inp(ih0, iw0 + 2, ci, n);
            let i10 = read_inp(ih0 + 1, iw0, ci, n);
            let i11 = read_inp(ih0 + 1, iw0 + 1, ci, n);
            let i12 = read_inp(ih0 + 1, iw0 + 2, ci, n);
            let i20 = read_inp(ih0 + 2, iw0, ci, n);
            let i21 = read_inp(ih0 + 2, iw0 + 1, ci, n);
            let i22 = read_inp(ih0 + 2, iw0 + 2, ci, n);
            if (co_grp + 0u < params.co) {{
                let co = co_grp + 0u;
                let wc0 = params.off_w + co * u32(params.w_so) + ci * u32(params.w_si);
                let w00 = f32(wgt[wc0]);
                let w01 = f32(wgt[wc0 + u32(params.w_sw)]);
                let w02 = f32(wgt[wc0 + 2u * u32(params.w_sw)]);
                let w10 = f32(wgt[wc0 + u32(params.w_sh)]);
                let w11 = f32(wgt[wc0 + u32(params.w_sh) + u32(params.w_sw)]);
                let w12 = f32(wgt[wc0 + u32(params.w_sh) + 2u * u32(params.w_sw)]);
                let w20 = f32(wgt[wc0 + 2u * u32(params.w_sh)]);
                let w21 = f32(wgt[wc0 + 2u * u32(params.w_sh) + u32(params.w_sw)]);
                let w22 = f32(wgt[wc0 + 2u * u32(params.w_sh) + 2u * u32(params.w_sw)]);
                acc0 += i00 * w00 + i01 * w01 + i02 * w02 + i10 * w10 + i11 * w11 + i12 * w12 + i20 * w20 + i21 * w21 + i22 * w22;
            }}
            if (co_grp + 1u < params.co) {{
                let co = co_grp + 1u;
                let wc0 = params.off_w + co * u32(params.w_so) + ci * u32(params.w_si);
                let w00 = f32(wgt[wc0]);
                let w01 = f32(wgt[wc0 + u32(params.w_sw)]);
                let w02 = f32(wgt[wc0 + 2u * u32(params.w_sw)]);
                let w10 = f32(wgt[wc0 + u32(params.w_sh)]);
                let w11 = f32(wgt[wc0 + u32(params.w_sh) + u32(params.w_sw)]);
                let w12 = f32(wgt[wc0 + u32(params.w_sh) + 2u * u32(params.w_sw)]);
                let w20 = f32(wgt[wc0 + 2u * u32(params.w_sh)]);
                let w21 = f32(wgt[wc0 + 2u * u32(params.w_sh) + u32(params.w_sw)]);
                let w22 = f32(wgt[wc0 + 2u * u32(params.w_sh) + 2u * u32(params.w_sw)]);
                acc1 += i00 * w00 + i01 * w01 + i02 * w02 + i10 * w10 + i11 * w11 + i12 * w12 + i20 * w20 + i21 * w21 + i22 * w22;
            }}
            if (co_grp + 2u < params.co) {{
                let co = co_grp + 2u;
                let wc0 = params.off_w + co * u32(params.w_so) + ci * u32(params.w_si);
                let w00 = f32(wgt[wc0]);
                let w01 = f32(wgt[wc0 + u32(params.w_sw)]);
                let w02 = f32(wgt[wc0 + 2u * u32(params.w_sw)]);
                let w10 = f32(wgt[wc0 + u32(params.w_sh)]);
                let w11 = f32(wgt[wc0 + u32(params.w_sh) + u32(params.w_sw)]);
                let w12 = f32(wgt[wc0 + u32(params.w_sh) + 2u * u32(params.w_sw)]);
                let w20 = f32(wgt[wc0 + 2u * u32(params.w_sh)]);
                let w21 = f32(wgt[wc0 + 2u * u32(params.w_sh) + u32(params.w_sw)]);
                let w22 = f32(wgt[wc0 + 2u * u32(params.w_sh) + 2u * u32(params.w_sw)]);
                acc2 += i00 * w00 + i01 * w01 + i02 * w02 + i10 * w10 + i11 * w11 + i12 * w12 + i20 * w20 + i21 * w21 + i22 * w22;
            }}
            if (co_grp + 3u < params.co) {{
                let co = co_grp + 3u;
                let wc0 = params.off_w + co * u32(params.w_so) + ci * u32(params.w_si);
                let w00 = f32(wgt[wc0]);
                let w01 = f32(wgt[wc0 + u32(params.w_sw)]);
                let w02 = f32(wgt[wc0 + 2u * u32(params.w_sw)]);
                let w10 = f32(wgt[wc0 + u32(params.w_sh)]);
                let w11 = f32(wgt[wc0 + u32(params.w_sh) + u32(params.w_sw)]);
                let w12 = f32(wgt[wc0 + u32(params.w_sh) + 2u * u32(params.w_sw)]);
                let w20 = f32(wgt[wc0 + 2u * u32(params.w_sh)]);
                let w21 = f32(wgt[wc0 + 2u * u32(params.w_sh) + u32(params.w_sw)]);
                let w22 = f32(wgt[wc0 + 2u * u32(params.w_sh) + 2u * u32(params.w_sw)]);
                acc3 += i00 * w00 + i01 * w01 + i02 * w02 + i10 * w10 + i11 * w11 + i12 * w12 + i20 * w20 + i21 * w21 + i22 * w22;
            }}
        }}
        for (var j = 0u; j < 4u; j++) {{
            let co = co_grp + j;
            if (co >= params.co) {{ break; }}
            let acc = select(select(select(acc0, acc1, j == 1u), acc2, j == 2u), acc3, j == 3u);
            let out_i = params.off_out + u32(i32(n) * params.out_sn + i32(co) * params.out_sc + i32(oh) * params.out_sh + i32(ow) * params.out_sw);
            var v = {t}(acc);
{epilogue}
            outp[out_i] = v;
        }}
    }}
}}
"#
    ));
    s
}

fn deconv_wgsl(dt: ShaderDtype) -> String {
    deconv_module(dt, &[], 0)
}

/// The gather-based transposed convolution, with an epilogue as
/// [`conv_module`] takes one.
pub fn deconv_module(dt: ShaderDtype, epilogue: &[ChainStep], extras: usize) -> String {
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
fn conv_transpose2d_{suf}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.y * {GRID_LIMIT}u * {WORKGROUP}u + gid.x;
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
    var v = {t}(acc);
{epilogue}
    outp[out_i] = v;
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
    let i = gid.y * {GRID_LIMIT}u * {WORKGROUP}u + gid.x;
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

/// Resize over the two trailing axes in one launch: each output sums its
/// `win_h x win_w` taps, weights being the product of the two axes' plans.
pub fn resize_2d_module(dt: ShaderDtype) -> String {
    let t = dt.wgsl();
    let suf = dt.suffix();
    let mut s = preamble(dt);
    s.push_str(&format!(
        r#"
struct Params {{
    off_in: u32,
    off_ih: u32,
    off_wh: u32,
    off_iw: u32,
    off_ww: u32,
    off_out: u32,
    outer: u32,
    h_in: u32,
    w_in: u32,
    h_out: u32,
    w_out: u32,
    win_h: u32,
    win_w: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}}

@group(0) @binding(0) var<storage, read> inp: array<{t}>;
@group(0) @binding(1) var<storage, read> idx_h: array<i32>;
@group(0) @binding(2) var<storage, read> wt_h: array<f32>;
@group(0) @binding(3) var<storage, read> idx_w: array<i32>;
@group(0) @binding(4) var<storage, read> wt_w: array<f32>;
@group(0) @binding(5) var<storage, read_write> outp: array<{t}>;
@group(0) @binding(6) var<uniform> params: Params;

@compute @workgroup_size({WORKGROUP})
fn resize_2d_{suf}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.y * {GRID_LIMIT}u * {WORKGROUP}u + gid.x;
    let n = params.outer * params.h_out * params.w_out;
    if (i >= n) {{ return; }}
    let xo = i % params.w_out;
    let t = i / params.w_out;
    let yo = t % params.h_out;
    let outer_i = t / params.h_out;
    let in_base = params.off_in + outer_i * params.h_in * params.w_in;
    var acc = 0.0;
    for (var a = 0u; a < params.win_h; a++) {{
        let iy = u32(idx_h[params.off_ih + yo * params.win_h + a]);
        let wy = wt_h[params.off_wh + yo * params.win_h + a];
        let row = in_base + iy * params.w_in;
        for (var b = 0u; b < params.win_w; b++) {{
            let ix = u32(idx_w[params.off_iw + xo * params.win_w + b]);
            let wx = wt_w[params.off_ww + xo * params.win_w + b];
            acc += f32(inp[row + ix]) * wy * wx;
        }}
    }}
    outp[params.off_out + i] = {t}(acc);
}}
"#
    ));
    s
}

/// Workgroup of the two-tap resize: 8 columns x 8 rows x 2 outer slices.
pub const RESIZE_2D_WG: [u32; 3] = [8, 8, 2];

/// [`resize_2d_module`] for two taps per axis — bilinear — with the taps
/// unrolled and a 3-D grid over (columns, rows, outer slices), so a workgroup
/// writes contiguous rows. Found by an OpenEvolve search scored in Chrome
/// (6.8x over the generic kernel there, 144 -> 21 us per frame of the
/// segmenter's three upsamples).
pub fn resize_2d_bilinear_module(dt: ShaderDtype) -> String {
    let t = dt.wgsl();
    let suf = dt.suffix();
    let [wx, wy, wz] = RESIZE_2D_WG;
    let mut s = preamble(dt);
    s.push_str(&format!(
        r#"
struct Params {{
    off_in: u32,
    off_ih: u32,
    off_wh: u32,
    off_iw: u32,
    off_ww: u32,
    off_out: u32,
    outer: u32,
    h_in: u32,
    w_in: u32,
    h_out: u32,
    w_out: u32,
    win_h: u32,
    win_w: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}}

@group(0) @binding(0) var<storage, read> inp: array<{t}>;
@group(0) @binding(1) var<storage, read> idx_h: array<i32>;
@group(0) @binding(2) var<storage, read> wt_h: array<f32>;
@group(0) @binding(3) var<storage, read> idx_w: array<i32>;
@group(0) @binding(4) var<storage, read> wt_w: array<f32>;
@group(0) @binding(5) var<storage, read_write> outp: array<{t}>;
@group(0) @binding(6) var<uniform> params: Params;

@compute @workgroup_size({wx}, {wy}, {wz})
fn resize_2d_bilinear_{suf}(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let yo = wgid.y * {wy}u + lid.y;
    let xo = wgid.x * {wx}u + lid.x;
    if (yo >= params.h_out || xo >= params.w_out) {{ return; }}
    let ih0 = u32(idx_h[params.off_ih + yo * 2u]);
    let ih1 = u32(idx_h[params.off_ih + yo * 2u + 1u]);
    let wh0 = wt_h[params.off_wh + yo * 2u];
    let wh1 = wt_h[params.off_wh + yo * 2u + 1u];
    let iw0 = u32(idx_w[params.off_iw + xo * 2u]);
    let iw1 = u32(idx_w[params.off_iw + xo * 2u + 1u]);
    let ww0 = wt_w[params.off_ww + xo * 2u];
    let ww1 = wt_w[params.off_ww + xo * 2u + 1u];
    let row0 = ih0 * params.w_in;
    let row1 = ih1 * params.w_in;
    let out_pos = yo * params.w_out + xo;
    let c = wgid.z * {wz}u + lid.z;
    if (c < params.outer) {{
        let in_base = params.off_in + c * params.h_in * params.w_in;
        let out_base = params.off_out + c * params.h_out * params.w_out;
        let acc = f32(inp[in_base + row0 + iw0]) * wh0 * ww0
            + f32(inp[in_base + row0 + iw1]) * wh0 * ww1
            + f32(inp[in_base + row1 + iw0]) * wh1 * ww0
            + f32(inp[in_base + row1 + iw1]) * wh1 * ww1;
        outp[out_base + out_pos] = {t}(acc);
    }}
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
    let i = gid.y * {GRID_LIMIT}u * {WORKGROUP}u + gid.x;
    let n = params.outer * params.inner;
    if (i >= n) {{ return; }}
    let o = i / params.inner;
    let r = i % params.inner;
    var m = f32(inp[params.off_in + (o * params.k) * params.inner + r]);
    for (var k = 1u; k < params.k; k++) {{
        m = max(m, f32(inp[params.off_in + (o * params.k + k) * params.inner + r]));
    }}
    var sum = 0.0;
    for (var k = 0u; k < params.k; k++) {{
        sum += exp(f32(inp[params.off_in + (o * params.k + k) * params.inner + r]) - m);
    }}
    let inv = 1.0 / sum;
    for (var k = 0u; k < params.k; k++) {{
        let x = f32(inp[params.off_in + (o * params.k + k) * params.inner + r]);
        outp[params.off_out + (o * params.k + k) * params.inner + r] = {t}(exp(x - m) * inv);
    }}
}}
"#
    ));
    s
}

/// Applies the fused steps to `v`, reading each extra operand at `index`.
fn epilogue_body(steps: &[ChainStep], index: &str) -> String {
    epilogue_body_at(steps, index, "0u")
}

/// As [`epilogue_body`], where a mode-2 operand is read at `element`, the
/// output element's own offset.
fn epilogue_body_at(steps: &[ChainStep], index: &str, element: &str) -> String {
    let mut s = String::new();
    for step in steps {
        match step {
            ChainStep::Unary(op) => s.push_str(&format!("    v = op_{op}(v);\n")),
            ChainStep::Binary { op, rhs, swapped } => {
                let i = rhs - 1;
                s.push_str(&format!(
                    "    let e{i} = extra{i}[params.off_extra[{i}u] + select(select(0u, {index}, params.mode_extra[{i}u] == 1u), {element}, params.mode_extra[{i}u] == 2u)];\n"
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
    let epilogue = epilogue_body_at(epilogue, "col", "oe");
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
    let i = gid.y * {GRID_LIMIT}u * {WORKGROUP}u + gid.x;
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
    let oe = o_i - params.off_out + row * o_row_s + col * o_col_s;
    var v = {t}(acc);
{epilogue}
    outp[o_i + row * o_row_s + col * o_col_s] = v;
}}
"#
    ));
    s
}

/// Threads of a tiled matmul or convolution workgroup: each computes
/// [`TILED_ROWS`] rows by [`TILED_COLS`] columns of the 64-square output tile.
pub const TILED_THREADS: usize = 128;
const TILED_ROWS: usize = 8;
const TILED_COLS: usize = 4;

/// The body shared by [`matmul_tiled_module`] and [`conv_tiled_module`], from
/// the tile origin to the stores. Expects `lid`, `row0`, `col0`, `k_len` (the
/// reduction length) and `interior` (the output tile lies inside the result) in
/// scope, `ld_a(actx, row, k, guard)` and `ld_b(bctx, k, col, guard)` reading
/// one operand value (zero outside it when `guard`) given the `vec4<u32>`
/// contexts `actx` and `bctx` the kernel sets up, and `store_out(<store_args>row,
/// col, acc)`. `k_prologue`, when not empty, runs at the top of each `k` slice
/// (`k0` its first index) ahead of a barrier and the loads. `a_k_fast` and
/// `b_n_fast` are WGSL booleans telling whether `a` is
/// contiguous along `k` and `b` along columns; either way a thread fills one
/// whole `vec4` of the tile, and neighbouring threads walk the contiguous axis
/// so their loads coalesce. Tiles are only ever stored a whole `vec4` at a
/// time: a store to one lane at a runtime index goes wrong on Metal.
///
/// `a` is staged as rows of `vec4` along `k` and `b` as `k` rows of `vec4`
/// along columns, in the tensor type: one workgroup load hands a thread four
/// `k` steps of a row, and f16 tiles halve the workgroup-memory traffic that
/// bounds this kernel. Accumulation is in f32, one register per output.
fn tiled_gemm_core(
    t: &str,
    a_k_fast: &str,
    b_n_fast: &str,
    store_args: &str,
    k_prologue: &str,
) -> String {
    let tile = MATMUL_TILE;
    let tk = MATMUL_TILE_K;
    let tk4 = tk / 4;
    let tile4 = tile / 4;
    let threads = TILED_THREADS;
    let (tm, tn) = (TILED_ROWS, TILED_COLS);
    let cols_threads = tile / tn;
    let mut s = format!(
        "    let tx = lid % {cols_threads}u;
    let ty = lid / {cols_threads}u;
    let a_k_fast = {a_k_fast};
    let b_n_fast = {b_n_fast};
"
    );
    for i in 0..tm {
        for j in 0..tn {
            s += &format!("    var c{i}_{j} = 0.0;\n");
        }
    }
    let k_prologue = if k_prologue.is_empty() {
        String::new()
    } else {
        format!("{k_prologue}        workgroupBarrier();\n")
    };
    s += &format!(
        "    for (var k0 = 0u; k0 < k_len; k0 += {tk}u) {{
        let guard = !interior || k0 + {tk}u > k_len;
{k_prologue}        for (var i = lid; i < {aq}u; i += {threads}u) {{
            if (a_k_fast) {{
                let r = i / {tk4}u;
                let kq = i % {tk4}u;
                let k = k0 + kq * 4u;
                a_tile[r * {tk4}u + kq] = vec4<{t}>(ld_a(actx, row0 + r, k, guard), ld_a(actx, row0 + r, k + 1u, guard), ld_a(actx, row0 + r, k + 2u, guard), ld_a(actx, row0 + r, k + 3u, guard));
            }} else {{
                let r = i % {tile}u;
                let kq = i / {tile}u;
                let k = k0 + kq * 4u;
                a_tile[r * {tk4}u + kq] = vec4<{t}>(ld_a(actx, row0 + r, k, guard), ld_a(actx, row0 + r, k + 1u, guard), ld_a(actx, row0 + r, k + 2u, guard), ld_a(actx, row0 + r, k + 3u, guard));
            }}
        }}
        for (var i = lid; i < {bq}u; i += {threads}u) {{
            if (b_n_fast) {{
                let kk = i / {tile4}u;
                let cq = i % {tile4}u;
                let c = col0 + cq * 4u;
                b_tile[kk * {tile4}u + cq] = vec4<{t}>(ld_b(bctx, k0 + kk, c, guard), ld_b(bctx, k0 + kk, c + 1u, guard), ld_b(bctx, k0 + kk, c + 2u, guard), ld_b(bctx, k0 + kk, c + 3u, guard));
            }} else {{
                let kk = i % {tk}u;
                let cq = i / {tk}u;
                let c = col0 + cq * 4u;
                b_tile[kk * {tile4}u + cq] = vec4<{t}>(ld_b(bctx, k0 + kk, c, guard), ld_b(bctx, k0 + kk, c + 1u, guard), ld_b(bctx, k0 + kk, c + 2u, guard), ld_b(bctx, k0 + kk, c + 3u, guard));
            }}
        }}
        workgroupBarrier();
        for (var kq = 0u; kq < {tk4}u; kq++) {{
",
        aq = tile * tk4,
        bq = tk * tile4,
    );
    for i in 0..tm {
        s += &format!(
            "            let a{i} = vec4<f32>(a_tile[(ty * {tm}u + {i}u) * {tk4}u + kq]);\n"
        );
    }
    for (sub, comp) in ["x", "y", "z", "w"].iter().enumerate() {
        s += &format!(
            "            let b{sub} = vec4<f32>(b_tile[(kq * 4u + {sub}u) * {tile4}u + tx]);\n"
        );
        for i in 0..tm {
            for (j, cj) in ["x", "y", "z", "w"].iter().enumerate() {
                s += &format!("            c{i}_{j} = fma(a{i}.{comp}, b{sub}.{cj}, c{i}_{j});\n");
            }
        }
    }
    s += "        }\n        workgroupBarrier();\n    }\n";
    s += &format!("    let r = row0 + ty * {tm}u;\n    let c = col0 + tx * {tn}u;\n");
    for i in 0..tm {
        for j in 0..tn {
            s += &format!("    store_out({store_args}r + {i}u, c + {j}u, c{i}_{j});\n");
        }
    }
    s
}

/// The workgroup tiles [`tiled_gemm_core`] stages its operands in.
fn tiled_gemm_tiles(t: &str) -> String {
    let tile = MATMUL_TILE;
    let tk = MATMUL_TILE_K;
    format!(
        "var<workgroup> a_tile: array<vec4<{t}>, {}>;\nvar<workgroup> b_tile: array<vec4<{t}>, {}>;\n",
        tile * tk / 4,
        tile * tk / 4
    )
}

/// Output rows and columns of each tiled matmul workgroup.
pub const MATMUL_TILE: usize = 64;
/// `k` values staged in workgroup memory per step of the tiled matmul.
pub const MATMUL_TILE_K: usize = 16;

/// The general matmul, staged through workgroup memory: each workgroup owns a
/// [`MATMUL_TILE`]-square block of one prefix slice of the output and walks `k`
/// in [`MATMUL_TILE_K`]-deep slices, with the loads, layout and accumulation of
/// [`tiled_gemm_core`]. Each operand is loaded along whichever of its two axes
/// is unit-strided, so loads coalesce for any transposition. Same `Params`,
/// strides and epilogue as [`matmul_module`], and no constraint on `m`, `k` or
/// `n`: partial tiles are guarded. Workgroup `x` tiles `n`, `y` tiles `m`, `z`
/// is the prefix index.
pub fn matmul_tiled_module(dt: ShaderDtype, epilogue: &[ChainStep], extras: usize) -> String {
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
    let epilogue = epilogue_body_at(epilogue, "col", "oe");
    let tile = MATMUL_TILE;
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

{tiles}
fn ld_a(ctx: vec4<u32>, r: u32, k: u32, guard: bool) -> {t} {{
    if (guard && (r >= params.m || k >= params.k)) {{ return {t}(0.0); }}
    return a[ctx.x + r * ctx.y + k * ctx.z];
}}

fn ld_b(ctx: vec4<u32>, k: u32, c: u32, guard: bool) -> {t} {{
    if (guard && (k >= params.k || c >= params.n)) {{ return {t}(0.0); }}
    return b[ctx.x + k * ctx.y + c * ctx.z];
}}

fn store_out(o_i: u32, o_row_s: u32, o_col_s: u32, row: u32, col: u32, acc: f32) {{
    if (row >= params.m || col >= params.n) {{ return; }}
    let oe = o_i - params.off_out + row * o_row_s + col * o_col_s;
    var v = {t}(acc);
{epilogue}
    outp[o_i + row * o_row_s + col * o_col_s] = v;
}}

@compute @workgroup_size({threads})
fn matmul_tiled_{suf}(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>) {{
    var a_i = params.off_a;
    var b_i = params.off_b;
    var o_i = params.off_out;
    var restp = wid.z;
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
    let actx = vec4<u32>(a_i, a_row_s, a_col_s, 0u);
    let bctx = vec4<u32>(b_i, b_row_s, b_col_s, 0u);
    let o_row_s = select(at8(params.out_s0, params.out_s1, 6u), at8(params.out_s0, params.out_s1, 7u), params.tc != 0u);
    let o_col_s = select(at8(params.out_s0, params.out_s1, 7u), at8(params.out_s0, params.out_s1, 6u), params.tc != 0u);

    let row0 = wid.y * {tile}u;
    let col0 = wid.x * {tile}u;
    let k_len = params.k;
    let interior = row0 + {tile}u <= params.m && col0 + {tile}u <= params.n;
{core}}}
"#,
        tiles = tiled_gemm_tiles(t),
        threads = TILED_THREADS,
        core = tiled_gemm_core(t, "a_col_s == 1u", "b_col_s == 1u", "o_i, o_row_s, o_col_s, ", ""),
    ));
    s
}

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
    let i = gid.y * {GRID_LIMIT}u * {WORKGROUP}u + gid.x;
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
    scale: vec4<f32>,
    offset: vec4<f32>,
}}

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> outp: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size({WORKGROUP})
fn rgba_to_nchw_f32(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.y * {GRID_LIMIT}u * {WORKGROUP}u + gid.x;
    let n = params.width * params.height;
    if (i >= n) {{ return; }}
    let x = i % params.width;
    let y = i / params.width;
    let px = textureLoad(src, vec2<i32>(i32(x), i32(y)), 0) * params.scale + params.offset;
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

/// Threads per workgroup of the split-k matmul: 32 tiles, four lanes each.
pub const MATMUL_SPLITK_WG: u32 = 128;
/// Below this `m` the blocked matmul runs split over k.
pub const MATMUL_SPLITK_MAX_M: usize = 2304;

/// The blocked matmul for small `m`: four lanes share a four-by-four output
/// tile, each summing a quarter of `k`, and reduce through workgroup memory.
/// The squeeze-excitation projections (m = 144, k up to 128) put only a few
/// hundred threads on the device with the plain kernel, so its time is one
/// thread's k loop; four lanes cut that chain and came out of a search scored
/// in Chrome at 2-2.5x on those shapes (m = 144: 56 -> 22 us, m = 576 with
/// k >= 72: 47 -> 23 us). Same contract and epilogue as
/// [`matmul_blocked_module`]; `k` a multiple of four.
pub fn matmul_splitk_module(dt: ShaderDtype, epilogue: &[ChainStep], extras: usize) -> String {
    let t = dt.wgsl();
    let suf = dt.suffix();
    let out_binding = 2 + extras;
    let uniform_binding = 3 + extras;
    let extra_bindings = (0..extras)
        .map(|i| {
            format!("@group(0) @binding({}) var<storage, read> extra{i}: array<{t}>;\n", 2 + i)
        })
        .collect::<String>();
    let lib = format!("{}{}", unary_ops_wgsl("f32"), binary_ops_wgsl("f32"));
    let epi = epilogue_body_at(epilogue, "col", "oe");
    let mut s = preamble(dt);
    s.push_str(&format!(
        r#"
struct Params {{
    m: u32,
    k: u32,
    n: u32,
    off_a: u32,
    off_b: u32,
    off_out: u32,
    _p0: u32,
    _p1: u32,
    off_extra: vec4<u32>,
    mode_extra: vec4<u32>,
}}

@group(0) @binding(0) var<storage, read> a4: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> b4: array<vec4<f32>>;
{extra_bindings}@group(0) @binding({out_binding}) var<storage, read_write> c4: array<vec4<f32>>;
@group(0) @binding({uniform_binding}) var<uniform> params: Params;
{lib}
var<workgroup> sh0: array<vec4<f32>, {MATMUL_SPLITK_WG}>;
var<workgroup> sh1: array<vec4<f32>, {MATMUL_SPLITK_WG}>;
var<workgroup> sh2: array<vec4<f32>, {MATMUL_SPLITK_WG}>;
var<workgroup> sh3: array<vec4<f32>, {MATMUL_SPLITK_WG}>;

fn epi(v_in: f32, col: u32, oe: u32) -> f32 {{
    var v = v_in;
{epi}
    return v;
}}

@compute @workgroup_size({MATMUL_SPLITK_WG})
fn matmul_splitk_{suf}(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let m4 = params.m / 4u;
    let n4 = params.n / 4u;
    let tile_global = gid.x >> 2u;
    let lane = gid.x & 3u;
    let row_tile_raw = tile_global % m4;
    let col_tile_raw = tile_global / m4;
    let valid = (row_tile_raw < m4) && (col_tile_raw < n4);
    let row_tile = min(row_tile_raw, m4 - 1u);
    let col_tile = min(col_tile_raw, n4 - 1u);

    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);

    let chunk = params.k >> 2u;
    let k_start = lane * chunk;
    let unroll_end = k_start + (chunk & ~3u);
    var kk = k_start;
    loop {{
        if (kk >= unroll_end) {{ break; }}
        let ab0 = params.off_a + kk * m4 + row_tile;
        let bb0 = params.off_b + kk * n4 + col_tile;
        let ab1 = ab0 + m4;
        let bb1 = bb0 + n4;
        let ab2 = ab1 + m4;
        let bb2 = bb1 + n4;
        let ab3 = ab2 + m4;
        let bb3 = bb2 + n4;
        let a0v = a4[ab0]; let b0v = b4[bb0];
        let a1v = a4[ab1]; let b1v = b4[bb1];
        let a2v = a4[ab2]; let b2v = b4[bb2];
        let a3v = a4[ab3]; let b3v = b4[bb3];
        acc0 += a0v * b0v.x + a1v * b1v.x + a2v * b2v.x + a3v * b3v.x;
        acc1 += a0v * b0v.y + a1v * b1v.y + a2v * b2v.y + a3v * b3v.y;
        acc2 += a0v * b0v.z + a1v * b1v.z + a2v * b2v.z + a3v * b3v.z;
        acc3 += a0v * b0v.w + a1v * b1v.w + a2v * b2v.w + a3v * b3v.w;
        kk += 4u;
    }}
    let k_end = k_start + chunk;
    loop {{
        if (kk >= k_end) {{ break; }}
        let ab = params.off_a + kk * m4 + row_tile;
        let bb = params.off_b + kk * n4 + col_tile;
        let av = a4[ab];
        let bv = b4[bb];
        acc0 += av * bv.x;
        acc1 += av * bv.y;
        acc2 += av * bv.z;
        acc3 += av * bv.w;
        kk += 1u;
    }}

    let base = lid.x - lane;
    sh0[lid.x] = acc0;
    sh1[lid.x] = acc1;
    sh2[lid.x] = acc2;
    sh3[lid.x] = acc3;
    workgroupBarrier();

    if (lane == 0u) {{
        let s0 = sh0[base] + sh0[base + 1u] + sh0[base + 2u] + sh0[base + 3u];
        let s1 = sh1[base] + sh1[base + 1u] + sh1[base + 2u] + sh1[base + 3u];
        let s2 = sh2[base] + sh2[base + 1u] + sh2[base + 2u] + sh2[base + 3u];
        let s3 = sh3[base] + sh3[base + 1u] + sh3[base + 2u] + sh3[base + 3u];
        if (valid) {{
            let col0 = col_tile * 4u;
            let cc = params.off_out + col0 * m4 + row_tile;
            let row_base = row_tile * 4u;
            let e0 = col0 * params.m + row_base;
            let e1 = e0 + params.m;
            let e2 = e1 + params.m;
            let e3 = e2 + params.m;
            c4[cc] = vec4<f32>(epi(s0.x, col0, e0), epi(s0.y, col0, e0 + 1u), epi(s0.z, col0, e0 + 2u), epi(s0.w, col0, e0 + 3u));
            c4[cc + m4] = vec4<f32>(epi(s1.x, col0 + 1u, e1), epi(s1.y, col0 + 1u, e1 + 1u), epi(s1.z, col0 + 1u, e1 + 2u), epi(s1.w, col0 + 1u, e1 + 3u));
            c4[cc + 2u * m4] = vec4<f32>(epi(s2.x, col0 + 2u, e2), epi(s2.y, col0 + 2u, e2 + 1u), epi(s2.z, col0 + 2u, e2 + 2u), epi(s2.w, col0 + 2u, e2 + 3u));
            c4[cc + 3u * m4] = vec4<f32>(epi(s3.x, col0 + 3u, e3), epi(s3.y, col0 + 3u, e3 + 1u), epi(s3.z, col0 + 3u, e3 + 2u), epi(s3.w, col0 + 3u, e3 + 3u));
        }}
    }}
}}
"#
    ));
    s
}

pub fn matmul_blocked_module(dt: ShaderDtype, epilogue: &[ChainStep], extras: usize) -> String {
    let t = dt.wgsl();
    let suf = dt.suffix();
    let out_binding = 2 + extras;
    let uniform_binding = 3 + extras;
    let extra_bindings = (0..extras)
        .map(|i| {
            format!("@group(0) @binding({}) var<storage, read> extra{i}: array<{t}>;\n", 2 + i)
        })
        .collect::<String>();
    let lib = format!("{}{}", unary_ops_wgsl("f32"), binary_ops_wgsl("f32"));
    let epi = epilogue_body_at(epilogue, "col", "oe");
    let mut s = preamble(dt);
    s.push_str(&format!(
        r#"
struct Params {{
    m: u32,
    k: u32,
    n: u32,
    off_a: u32,
    off_b: u32,
    off_out: u32,
    _p0: u32,
    _p1: u32,
    off_extra: vec4<u32>,
    mode_extra: vec4<u32>,
}}

@group(0) @binding(0) var<storage, read> a4: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> b4: array<vec4<f32>>;
{extra_bindings}@group(0) @binding({out_binding}) var<storage, read_write> c4: array<vec4<f32>>;
@group(0) @binding({uniform_binding}) var<uniform> params: Params;
{lib}

fn epi(v_in: f32, col: u32, oe: u32) -> f32 {{
    var v = v_in;
{epi}
    return v;
}}

@compute @workgroup_size({MATMUL_BLOCKED_WG})
fn matmul_blocked_{suf}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let m4 = params.m / 4u;
    let n4 = params.n / 4u;
    let tile = gid.x;
    let row_tile = tile % m4;
    let col_tile = tile / m4;
    if (col_tile >= n4) {{ return; }}
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);
    var kk = 0u;
    for (; kk + 8u <= params.k; kk += 8u) {{
        let ab = params.off_a + kk * m4 + row_tile;
        let bb = params.off_b + kk * n4 + col_tile;
        let a0 = a4[ab];
        let a1 = a4[ab + m4];
        let a2 = a4[ab + 2u * m4];
        let a3 = a4[ab + 3u * m4];
        let a4v = a4[ab + 4u * m4];
        let a5 = a4[ab + 5u * m4];
        let a6 = a4[ab + 6u * m4];
        let a7 = a4[ab + 7u * m4];
        let b0 = b4[bb];
        let b1 = b4[bb + n4];
        let b2 = b4[bb + 2u * n4];
        let b3 = b4[bb + 3u * n4];
        let b4v = b4[bb + 4u * n4];
        let b5 = b4[bb + 5u * n4];
        let b6 = b4[bb + 6u * n4];
        let b7 = b4[bb + 7u * n4];
        acc0 += a0 * b0.x + a1 * b1.x + a2 * b2.x + a3 * b3.x
              + a4v * b4v.x + a5 * b5.x + a6 * b6.x + a7 * b7.x;
        acc1 += a0 * b0.y + a1 * b1.y + a2 * b2.y + a3 * b3.y
              + a4v * b4v.y + a5 * b5.y + a6 * b6.y + a7 * b7.y;
        acc2 += a0 * b0.z + a1 * b1.z + a2 * b2.z + a3 * b3.z
              + a4v * b4v.z + a5 * b5.z + a6 * b6.z + a7 * b7.z;
        acc3 += a0 * b0.w + a1 * b1.w + a2 * b2.w + a3 * b3.w
              + a4v * b4v.w + a5 * b5.w + a6 * b6.w + a7 * b7.w;
    }}
    for (; kk < params.k; kk += 4u) {{
        let ab = params.off_a + kk * m4 + row_tile;
        let bb = params.off_b + kk * n4 + col_tile;
        let a0 = a4[ab];
        let a1 = a4[ab + m4];
        let a2 = a4[ab + 2u * m4];
        let a3 = a4[ab + 3u * m4];
        let b0 = b4[bb];
        let b1 = b4[bb + n4];
        let b2 = b4[bb + 2u * n4];
        let b3 = b4[bb + 3u * n4];
        acc0 += a0 * b0.x + a1 * b1.x + a2 * b2.x + a3 * b3.x;
        acc1 += a0 * b0.y + a1 * b1.y + a2 * b2.y + a3 * b3.y;
        acc2 += a0 * b0.z + a1 * b1.z + a2 * b2.z + a3 * b3.z;
        acc3 += a0 * b0.w + a1 * b1.w + a2 * b2.w + a3 * b3.w;
    }}
    let col0 = col_tile * 4u;
    let cc = params.off_out + col0 * m4 + row_tile;
    let e0 = (col0 * m4 + row_tile) * 4u;
    let e1 = e0 + m4 * 4u;
    let e2 = e1 + m4 * 4u;
    let e3 = e2 + m4 * 4u;
    c4[cc] = vec4<f32>(
        epi(acc0.x, col0, e0), epi(acc0.y, col0, e0 + 1u),
        epi(acc0.z, col0, e0 + 2u), epi(acc0.w, col0, e0 + 3u));
    c4[cc + m4] = vec4<f32>(
        epi(acc1.x, col0 + 1u, e1), epi(acc1.y, col0 + 1u, e1 + 1u),
        epi(acc1.z, col0 + 1u, e1 + 2u), epi(acc1.w, col0 + 1u, e1 + 3u));
    c4[cc + 2u * m4] = vec4<f32>(
        epi(acc2.x, col0 + 2u, e2), epi(acc2.y, col0 + 2u, e2 + 1u),
        epi(acc2.z, col0 + 2u, e2 + 2u), epi(acc2.w, col0 + 2u, e2 + 3u));
    c4[cc + 3u * m4] = vec4<f32>(
        epi(acc3.x, col0 + 3u, e3), epi(acc3.y, col0 + 3u, e3 + 1u),
        epi(acc3.z, col0 + 3u, e3 + 2u), epi(acc3.w, col0 + 3u, e3 + 3u));
}}
"#
    ));
    s
}

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
