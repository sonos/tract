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

use crate::kernels::bin_ops::{BINARY_OPS, binary_wgsl};
use crate::kernels::cast::cast_wgsl;
use crate::kernels::conv::conv_wgsl;
use crate::kernels::copy::{copy_wgsl, transpose_wgsl};
use crate::kernels::deconv::deconv_wgsl;
use crate::kernels::element_wise::{ELEMENT_WISE_OPS, element_wise_wgsl};
use crate::kernels::ingest::{export_wgsl, ingest_wgsl};
use crate::kernels::matmul::matmul_wgsl;
use crate::kernels::pool::pool_wgsl;
use crate::kernels::reduce::reduce_wgsl;
use crate::kernels::resize::resize_wgsl;
use crate::kernels::softmax::softmax_wgsl;

pub const WORKGROUP: u32 = 64;

/// WebGPU's cap on workgroups per dispatch dimension. A kernel that walks a
/// tensor linearly takes the overflow on y and folds it back into its index.
pub const GRID_LIMIT: u32 = 65535;

/// Reads one of the eight values a uniform packs as two `vec4<u32>`.
pub(crate) const AT8: &str = "fn at8(a: vec4<u32>, b: vec4<u32>, i: u32) -> u32 {
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
    /// `n` storage buffers of which the last `writes` are writable.
    TailWrites {
        n: u8,
        writes: u8,
    },
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
            Self::TailWrites { .. } => "tract-wgpu-tail-writes-layout",
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
            Self::TailWrites { n, .. } => n as u32,
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
        let writes = match self {
            Self::TailWrites { writes, .. } => writes as u32,
            _ => 1,
        };
        for i in 0..n {
            let read_only = i + writes < n;
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

pub(crate) fn preamble(dt: ShaderDtype) -> String {
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

/// A four-wide wrapper for each op, so a kernel that moves whole `vec4`s can
/// still call them. The bodies stay scalar: the win here is the wider load and
/// store, not the arithmetic.
pub(crate) fn unary_ops_vec4_wgsl(t: &str) -> String {
    let mut s = String::new();
    for name in ELEMENT_WISE_OPS {
        s.push_str(&format!(
            "fn op_{name}_v(x: vec4<{t}>) -> vec4<{t}> {{ return vec4<{t}>(op_{name}(x.x), op_{name}(x.y), op_{name}(x.z), op_{name}(x.w)); }}\n"
        ));
    }
    s
}

pub(crate) fn binary_ops_vec4_wgsl(t: &str) -> String {
    let mut s = String::new();
    for name in BINARY_OPS {
        s.push_str(&format!(
            "fn op_{name}_v(a: vec4<{t}>, b: vec4<{t}>) -> vec4<{t}> {{ return vec4<{t}>(op_{name}(a.x, b.x), op_{name}(a.y, b.y), op_{name}(a.z, b.z), op_{name}(a.w, b.w)); }}\n"
        ));
    }
    s
}

/// Every unary `op_*` a kernel may call, plus the erf polynomial they share.
pub(crate) fn unary_ops_wgsl(t: &str) -> String {
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
pub(crate) fn binary_ops_wgsl(t: &str) -> String {
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

/// Threads of each layer norm workgroup.
pub const LAYER_NORM_WG: u32 = 256;

/// Applies the fused steps to `v`, reading each extra operand at `index`.
pub(crate) fn epilogue_body(steps: &[ChainStep], index: &str) -> String {
    epilogue_body_at(steps, index, "0u")
}

/// As [`epilogue_body`], where a mode-2 operand is read at `element`, the
/// output element's own offset.
pub(crate) fn epilogue_body_at(steps: &[ChainStep], index: &str, element: &str) -> String {
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
