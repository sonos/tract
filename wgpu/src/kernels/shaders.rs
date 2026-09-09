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
}

impl LayoutKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::Unary => "tract-wgpu-unary-layout",
        }
    }

    pub fn storage_count(self) -> u32 {
        match self {
            Self::Unary => 2,
        }
    }

    pub fn bind_group_layout_entries(self) -> Vec<wgpu::BindGroupLayoutEntry> {
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
    Copy,
    Cast,
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
            ModuleKind::Copy | ModuleKind::Cast => LayoutKind::Unary,
        }
    }

    pub fn wgsl(self) -> String {
        match self.kind {
            ModuleKind::Copy => copy_wgsl(self.dtype),
            ModuleKind::Cast => cast_wgsl(self.dtype),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PipelineKey {
    pub module: ModuleKey,
    pub entry: EntryPoint,
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

fn preamble(dt: ShaderDtype) -> String {
    let enable = match dt {
        ShaderDtype::F16 => "enable f16;\n",
        ShaderDtype::F32 => "",
    };
    enable.to_string()
}

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
