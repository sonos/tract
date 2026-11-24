//! Experimental subgroup GEMV and subgroup-matrix GEMM.
//!
//! `TRACT_WGPU_SUBGROUPS` defaults off. The device is created once, so the
//! feature is requested only when the knob is already on at that moment.
//! Dispatch reads the knob again and will not use the kernel when a later
//! scope turns it off.
//!
//! Sizes and tiles come from the adapter. Chrome 158 on this Mac advertises
//! `subgroups` and `subgroup-size-control` at 32/32 and does not advertise
//! `chromium-experimental-subgroup-matrix` unless unsafe WebGPU is on.
//! Safari 27.2 advertises neither. STP 257 advertises `subgroups` at 32/32
//! and not size control. A device with nothing usable keeps the existing
//! kernels.
//!
//! Dawn WGSL needs `enable subgroups;` and, for the matrix tile,
//! `enable chromium_experimental_subgroup_matrix;`. Naga 30 rejects
//! `enable subgroups;`, so the native shader uses `enable wgpu_cooperative_matrix;`
//! and only the square `coop_mat8x8` / `coop_mat16x16` tiles Naga can parse.
//! The matrix tile runs only when the subgroup size is one value. Equal
//! adapter bounds are that value. A range is probed: one workgroup writes
//! `subgroup_size`, and a single power of two inside the range becomes the
//! size. wgpu-hal reports 4..64 for every Metal device; Apple's simdgroup is
//! the probed value, not a guess.
//!
//! Every subgroup owns an output tile. A wider workgroup cannot return before
//! a cooperative op. The workgroup holds up to eight subgroups, and each
//! subgroup keeps at most four accumulators. Splitting K across those
//! subgroups was slower than this on the fat 1×1s, so the subgroups split the
//! output instead. GEMV still runs on a range the probe could not settle:
//! workgroup X is the advertised max and only subgroup 0 accumulates.
//!
//! Short K and a product that does not divide the tile stay on the existing
//! kernel. A remainder of M is a second dispatch.

use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

use crate::context::WgpuQueue;
use crate::kernels::TRACT_WGPU_SUBGROUPS;
use crate::kernels::matmul::Transposes;
use crate::kernels::shaders::{
    AT8, ChainStep, EntryPoint, GRID_LIMIT, LayoutKind, ShaderDtype, binary_ops_wgsl,
    epilogue_body_at, pack_u32s, rpad8_dims, rpad8_strides, unary_ops_wgsl,
};
use crate::utils::{element_offset, get_wgpu_buffer};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;

/// One advertised matrix tile. `m`/`n`/`k` are the tile, not the product.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MatrixTile {
    pub m: u32,
    pub n: u32,
    pub k: u32,
}

/// What this device can actually run. All false when the knob was off at
/// device creation, or the adapter advertised nothing the shader can use.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SubgroupCaps {
    pub subgroups: bool,
    pub min_size: u32,
    pub max_size: u32,
    /// `subgroup-size-control` was granted. Dawn-only. Native wgpu cannot pin
    /// `@subgroup_size`. A probed [`Self::fixed_size`] is what lets the matrix
    /// tile run when the adapter reports a range.
    pub size_control: bool,
    pub matrix: bool,
    pub f32_tile: Option<MatrixTile>,
    pub f16_tile: Option<MatrixTile>,
    /// Tile that multiplies `f16` operands into an `f32` accumulator.
    pub f16_mixed_tile: Option<MatrixTile>,
    /// Subgroup size the kernels actually use. `min == max`, or the one size
    /// a probe observed. Zero when the size is still a range.
    pub fixed_size: u32,
}

/// How a kernel should size its workgroup.
///
/// `pin` asks Dawn for `@subgroup_size`. `varying` means the implementation
/// may pick any power of two in range. GEMV then uses a workgroup of `size`
/// (the max) and only subgroup 0 runs, stepping by the `subgroup_size` builtin.
/// The matrix tile is not dispatched on that path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SubgroupChoice {
    pub size: u32,
    pub pin: bool,
    pub varying: bool,
}

/// Equal min and max is that size, one subgroup per workgroup. A range with
/// `subgroup-size-control` is pinned to the max on Dawn, also one subgroup.
/// Otherwise the workgroup is the max. GEMV runs only subgroup 0 and steps by
/// the `subgroup_size` builtin, which stays correct for a fixed 32 and for
/// wgpu's 4..64 Metal report of the same GPU. The matrix tile needs the
/// fixed or pinned case: every lane has to reach the cooperative op.
pub(crate) fn chosen_subgroup(caps: &SubgroupCaps) -> Option<SubgroupChoice> {
    let (min, max) = (caps.min_size, caps.max_size);
    if min < 4 || max > 128 || min > max || min == 0 {
        return None;
    }
    if !min.is_power_of_two() || !max.is_power_of_two() {
        return None;
    }
    if (4..=max).contains(&caps.fixed_size)
        && caps.fixed_size >= min
        && caps.fixed_size.is_power_of_two()
    {
        return Some(SubgroupChoice { size: caps.fixed_size, pin: false, varying: false });
    }
    if min == max {
        return Some(SubgroupChoice { size: min, pin: false, varying: false });
    }
    if caps.size_control {
        #[cfg(target_arch = "wasm32")]
        {
            return Some(SubgroupChoice { size: max, pin: true, varying: false });
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = max;
        }
    }
    Some(SubgroupChoice { size: max, pin: false, varying: true })
}

/// Columns one GEMV workgroup owns. The product with the subgroup size stays
/// inside 256 and inside the device's workgroup limit.
pub(crate) fn gemv_cols(subgroup: u32, max_invocations: u32, max_x: u32) -> Option<u32> {
    let cap = max_invocations.min(max_x).min(256);
    if subgroup == 0 || subgroup > cap {
        return None;
    }
    let cols = cap / subgroup;
    if cols == 0 {
        return None;
    }
    Some(cols)
}

fn larger_tile(candidate: MatrixTile, best: Option<MatrixTile>) -> Option<MatrixTile> {
    match best {
        None => Some(candidate),
        Some(have) => {
            let area = |t: MatrixTile| t.m as u64 * t.n as u64 * t.k as u64;
            if area(candidate) > area(have)
                || (area(candidate) == area(have) && candidate.m > have.m)
            {
                Some(candidate)
            } else {
                Some(have)
            }
        }
    }
}

/// The 8x8 tile that multiplies `f16` operands into an `f32` accumulator.
pub(crate) fn pick_native_mixed_tile(
    props: &[wgpu::CooperativeMatrixProperties],
) -> Option<MatrixTile> {
    props
        .iter()
        .find(|p| {
            !p.saturating_accumulation
                && p.ab_type == wgpu::CooperativeScalarType::F16
                && p.cr_type == wgpu::CooperativeScalarType::F32
                && (p.m_size, p.n_size, p.k_size) == (8, 8, 8)
        })
        .map(|p| MatrixTile { m: p.m_size, n: p.n_size, k: p.k_size })
}

/// Largest square 8 or 16 tile whose inputs and result share `scalar`,
/// without saturating accumulation. Naga parses only those two shapes.
pub(crate) fn pick_native_tile(
    props: &[wgpu::CooperativeMatrixProperties],
    scalar: wgpu::CooperativeScalarType,
) -> Option<MatrixTile> {
    let mut best = None;
    for prop in props {
        if prop.saturating_accumulation || prop.ab_type != scalar || prop.cr_type != scalar {
            continue;
        }
        if prop.m_size != prop.n_size || prop.n_size != prop.k_size {
            continue;
        }
        if prop.m_size != 8 && prop.m_size != 16 {
            continue;
        }
        best = larger_tile(MatrixTile { m: prop.m_size, n: prop.n_size, k: prop.k_size }, best);
    }
    best
}

#[cfg(any(target_arch = "wasm32", test))]
fn norm_component(name: &str) -> Option<&'static str> {
    match name.trim().to_ascii_lowercase().as_str() {
        "f32" | "float32" | "fp32" => Some("f32"),
        "f16" | "float16" | "fp16" => Some("f16"),
        _ => None,
    }
}

/// Largest listed Dawn tile whose component type and result type are both `want`.
/// Mixed results are skipped: there is no element extract, so an f32 accumulator
/// cannot be stored into an f16 buffer.
#[cfg(any(target_arch = "wasm32", test))]
pub(crate) fn pick_dawn_tile(
    configs: &[(String, String, u32, u32, u32)],
    want: &str,
) -> Option<MatrixTile> {
    let mut best = None;
    for (component, result, m, n, k) in configs {
        if m == &0 || n == &0 || k == &0 {
            continue;
        }
        if norm_component(component) != Some(want) || norm_component(result) != Some(want) {
            continue;
        }
        best = larger_tile(MatrixTile { m: *m, n: *n, k: *k }, best);
    }
    best
}

/// Native caps from the device that was actually created. The adapter's
/// advertised sizes are ignored unless the `subgroups` feature was granted.
pub fn caps_from_native(
    device: &wgpu::Device,
    adapter: &wgpu::Adapter,
    shader_f16: bool,
) -> SubgroupCaps {
    let mut caps = SubgroupCaps::default();
    let features = device.features();
    if features.contains(wgpu::Features::SUBGROUP) {
        let info = adapter.get_info();
        caps.subgroups = true;
        caps.min_size = info.subgroup_min_size;
        caps.max_size = info.subgroup_max_size;
        if caps.min_size > 0 && caps.min_size == caps.max_size {
            caps.fixed_size = caps.min_size;
        }
    }
    if features.contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) {
        caps.matrix = true;
        let props = adapter.cooperative_matrix_properties();
        caps.f32_tile = pick_native_tile(&props, wgpu::CooperativeScalarType::F32);
        if shader_f16 {
            caps.f16_tile = pick_native_tile(&props, wgpu::CooperativeScalarType::F16);
            caps.f16_mixed_tile = pick_native_mixed_tile(&props);
        }
    }
    caps
}

/// One size, if every subgroup that wrote `subgroup_size` wrote that size.
pub(crate) fn consensus_size(sizes: &[u32]) -> Option<u32> {
    let mut found = None;
    for &size in sizes {
        if size == 0 {
            continue;
        }
        if !(4..=128).contains(&size) || !size.is_power_of_two() {
            return None;
        }
        match found {
            None => found = Some(size),
            Some(prev) if prev != size => return None,
            Some(_) => {}
        }
    }
    found
}

/// Read the subgroup size this device actually runs. wgpu-hal's Metal backend
/// reports 4..64 for every GPU, which is the range the API allows, not the
/// simdgroup width. A workgroup of the advertised max writes one
/// `subgroup_size` per subgroup. A single power of two is the size the matrix
/// tile may use. Anything mixed, empty, or outside the adapter range leaves
/// the tile off.
pub async fn probe_fixed_size(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    max_size: u32,
) -> Option<u32> {
    if max_size < 4 {
        return None;
    }
    let limits = device.limits();
    let launch = max_size
        .min(limits.max_compute_invocations_per_workgroup)
        .min(limits.max_compute_workgroup_size_x);
    if launch < 4 {
        return None;
    }
    let launch = 1u32 << launch.ilog2();
    if launch < 4 {
        return None;
    }
    let slots = (launch / 4) as usize;
    let src = format!(
        r#"@group(0) @binding(0) var<storage, read_write> sizes: array<u32>;
@compute @workgroup_size({launch})
fn probe(@builtin(subgroup_id) sg: u32, @builtin(subgroup_invocation_id) lane: u32, @builtin(subgroup_size) sz: u32) {{
    if (lane == 0u && sg < {slots}u) {{
        sizes[sg] = sz;
    }}
}}
"#
    );
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("tract-subgroup-probe"),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("tract-subgroup-probe"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("tract-subgroup-probe"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("tract-subgroup-probe"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("probe"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    if scope.pop().await.is_some() {
        log::warn!("tract-wgpu: subgroup size probe shader was rejected");
        return None;
    }
    let storage = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tract-subgroup-probe-sizes"),
        size: (slots * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&storage, 0, &vec![0u8; slots * 4]);
    let read = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tract-subgroup-probe-read"),
        size: (slots * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("tract-subgroup-probe"),
        layout: &layout,
        entries: &[wgpu::BindGroupEntry { binding: 0, resource: storage.as_entire_binding() }],
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("tract-subgroup-probe"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("tract-subgroup-probe"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&storage, 0, &read, 0, (slots * 4) as u64);
    queue.submit(Some(encoder.finish()));
    let slice = read.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    if device.poll(wgpu::PollType::wait_indefinitely()).is_err() {
        return None;
    }
    rx.recv().ok()?.ok()?;
    let copied: Vec<u32> = {
        let view = slice.get_mapped_range().ok()?;
        view.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    };
    read.unmap();
    let size = consensus_size(&copied)?;
    log::info!("tract-wgpu: probed subgroup size {size} from {copied:?}");
    Some(size)
}

/// Install the Dawn `requestDevice` hook and remember whether this creation
/// wants the features. The hook appends a feature only when the adapter
/// already lists it. A rejected request is retried with the original
/// descriptor so a buggy driver does not sink device creation.
#[cfg(target_arch = "wasm32")]
pub fn prepare_wasm_request(want: bool) {
    let global = js_sys::global();
    let _ = js_sys::Reflect::set(
        &global,
        &wasm_bindgen::JsValue::from_str("__tractWantSubgroups"),
        &wasm_bindgen::JsValue::from_bool(want),
    );
    let script = r#"
(() => {
  const g = globalThis;
  if (typeof GPUAdapter === "undefined" || GPUAdapter.prototype.__tractSubgroupsHook) return;
  const orig = GPUAdapter.prototype.requestDevice;
  GPUAdapter.prototype.__tractSubgroupsHook = true;
  GPUAdapter.prototype.requestDevice = function (desc) {
    const want = g.__tractWantSubgroups === true;
    const features = [];
    const listed = desc && desc.requiredFeatures ? desc.requiredFeatures : [];
    for (const f of listed) features.push(f);
    const has = (name) => !!(this.features && this.features.has && this.features.has(name));
    const info = this.info || {};
    const min = info.subgroupMinSize | 0;
    const max = info.subgroupMaxSize | 0;
    const hasSubgroups = has("subgroups");
    const hasSize = has("subgroup-size-control");
    const hasMatrix = has("chromium-experimental-subgroup-matrix");
    if (want && hasSubgroups && features.indexOf("subgroups") < 0) features.push("subgroups");
    if (want && hasSize && min > 0 && max > min && features.indexOf("subgroup-size-control") < 0) {
      features.push("subgroup-size-control");
    }
    if (want && hasMatrix && features.indexOf("chromium-experimental-subgroup-matrix") < 0) {
      features.push("chromium-experimental-subgroup-matrix");
    }
    const configs = [];
    const raw = info.subgroupMatrixConfigs || info.subgroup_matrix_configs;
    if (raw && raw.length) {
      for (let i = 0; i < raw.length; i++) {
        const c = raw[i] || {};
        configs.push({
          componentType: String(c.componentType || c.component_type || ""),
          resultComponentType: String(c.resultComponentType || c.result_component_type || ""),
          M: c.M | 0,
          N: c.N | 0,
          K: c.K | 0
        });
      }
    }
    g.__tractSubgroupInfo = { min: min, max: max, hasSubgroups: hasSubgroups, hasSize: hasSize, hasMatrix: hasMatrix, configs: configs };
    const added = features.length !== listed.length;
    const next = Object.assign({}, desc || {}, { requiredFeatures: features });
    const pending = orig.call(this, next);
    if (!want || !added) return pending;
    return Promise.resolve(pending).catch((err) => {
      console.warn("tract-wgpu: subgroup requestDevice failed; retrying without those features", err);
      g.__tractSubgroupInfo = { min: min, max: max, hasSubgroups: false, hasSize: false, hasMatrix: false, configs: [] };
      return orig.call(this, desc);
    });
  };
})()
"#;
    let _ = js_sys::eval(script);
}

#[cfg(target_arch = "wasm32")]
fn js_device_has(device: &wgpu::Device, name: &str) -> bool {
    let Some(raw) = device.as_webgpu() else {
        return false;
    };
    let Ok(features) =
        js_sys::Reflect::get(raw.as_ref(), &wasm_bindgen::JsValue::from_str("features"))
    else {
        return false;
    };
    let Ok(has) = js_sys::Reflect::get(&features, &wasm_bindgen::JsValue::from_str("has")) else {
        return false;
    };
    let Ok(fun) = has.dyn_into::<js_sys::Function>() else {
        return false;
    };
    fun.call1(&features, &wasm_bindgen::JsValue::from_str(name))
        .ok()
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
}

#[cfg(target_arch = "wasm32")]
fn js_u32(value: &wasm_bindgen::JsValue, key: &str) -> u32 {
    js_sys::Reflect::get(value, &wasm_bindgen::JsValue::from_str(key))
        .ok()
        .and_then(|v| v.as_f64())
        .map(|n| n as u32)
        .unwrap_or(0)
}

#[cfg(target_arch = "wasm32")]
fn js_configs(value: &wasm_bindgen::JsValue) -> Vec<(String, String, u32, u32, u32)> {
    let Ok(configs) = js_sys::Reflect::get(value, &wasm_bindgen::JsValue::from_str("configs"))
    else {
        return Vec::new();
    };
    let Ok(array) = configs.dyn_into::<js_sys::Array>() else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for item in array.iter() {
        let component =
            js_sys::Reflect::get(&item, &wasm_bindgen::JsValue::from_str("componentType"))
                .ok()
                .and_then(|v| v.as_string())
                .unwrap_or_default();
        let result =
            js_sys::Reflect::get(&item, &wasm_bindgen::JsValue::from_str("resultComponentType"))
                .ok()
                .and_then(|v| v.as_string())
                .unwrap_or_default();
        out.push((component, result, js_u32(&item, "M"), js_u32(&item, "N"), js_u32(&item, "K")));
    }
    out
}

/// Caps from the JS device after the hook ran. wgpu's own feature bits do
/// not include `subgroups` on the web backend, so the page's `GPUDevice` is
/// what decides.
#[cfg(target_arch = "wasm32")]
pub fn caps_from_wasm(device: &wgpu::Device, shader_f16: bool) -> SubgroupCaps {
    let mut caps = SubgroupCaps::default();
    let global = js_sys::global();
    let Ok(info) =
        js_sys::Reflect::get(&global, &wasm_bindgen::JsValue::from_str("__tractSubgroupInfo"))
    else {
        return caps;
    };
    if info.is_undefined() || info.is_null() {
        return caps;
    }
    if js_device_has(device, "subgroups") {
        caps.subgroups = true;
        caps.min_size = js_u32(&info, "min");
        caps.max_size = js_u32(&info, "max");
        caps.size_control = js_device_has(device, "subgroup-size-control");
        if caps.min_size > 0 && caps.min_size == caps.max_size {
            caps.fixed_size = caps.min_size;
        }
    }
    if js_device_has(device, "chromium-experimental-subgroup-matrix") {
        caps.matrix = true;
        let configs = js_configs(&info);
        caps.f32_tile = pick_dawn_tile(&configs, "f32");
        if shader_f16 {
            caps.f16_tile = pick_dawn_tile(&configs, "f16");
        }
    }
    caps
}

/// Which logical axis is contiguous. `Row` is the packed row-major load
/// (`coopLoadT` / `row_major`). `Col` is the transposed channels-first 1×1,
/// where M is the fast axis (`coopLoad` / `col_major`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum AxisMajor {
    Row,
    Col,
}

fn contiguous_major(row: isize, col: isize) -> Option<AxisMajor> {
    if col == 1 && row > 0 {
        Some(AxisMajor::Row)
    } else if row == 1 && col > 0 {
        Some(AxisMajor::Col)
    } else {
        None
    }
}

/// The stride that is not 1 has to clear the tile along that axis.
fn major_fits(
    major: AxisMajor,
    row: isize,
    col: isize,
    along_row: isize,
    along_col: isize,
) -> bool {
    match major {
        AxisMajor::Row => row >= along_row,
        AxisMajor::Col => col >= along_col,
    }
}

struct GemvPlan {
    size: u32,
    cols: u32,
    pin: bool,
    varying: bool,
}

struct MatrixPlan {
    tile: MatrixTile,
    size: u32,
    pin: bool,
    major: [AxisMajor; 3],
    block: OutputBlock,
}

/// How many hardware tiles one subgroup owns, how the subgroups tile the
/// workgroup, and the M remainder a second dispatch covers. `rows` is the
/// main region, starting at row 0. `sg_k` is 4 when subgroups split K, and
/// 1 when they split the output instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct OutputBlock {
    pub tm: u32,
    pub tn: u32,
    pub sg_m: u32,
    pub sg_n: u32,
    pub sg_k: u32,
    pub rows: usize,
    pub edge_tm: u32,
    pub edge_tn: u32,
    pub edge_sg_m: u32,
    pub edge_sg_n: u32,
    pub edge_rows: usize,
}

/// Register tile and subgroup grid. A full cover of M beats a larger tile
/// count: the remainder is a second dispatch and that lost to split-K.
/// Among full covers, the workgroup closest to [`TARGET_THREADS`] wins, then
/// intensity `tm*tn/(tm+tn)`, then tile count. K under 256 and a lone 8×8
/// stay on the old kernel. Four accumulators is the cap because sixteen
/// spilled. Eight subgroups of 32 also lost, so 256 threads is not the goal.
/// A 4-way split of K lost to this schedule on MODNet 144 and RVM.
const MAX_ACCUMULATORS: u32 = 4;
const MAX_SUBGROUPS: u32 = 8;
const TARGET_THREADS: u32 = 128;
/// K under this stays on split-K. 160 (RVM) tied the fast split-K floor and
/// lost to Chrome's column-first kernel. 320 with N = 1280 (MODNet 144) is
/// where the unrolled tile is faster, on both bias flags. N < K stays on
/// split-K as well: 144×960×320 did not.
const MIN_MATRIX_K: usize = 256;
const MIN_MMAS: u32 = 64;

fn block_score(
    sg_size: u32,
    sg_m: u32,
    sg_n: u32,
    tm: u32,
    tn: u32,
    covered: usize,
    m: usize,
) -> (u32, u32, u32, u32, usize) {
    let threads = sg_size * sg_m * sg_n;
    let full = u32::from(covered == m);
    let distance = threads.abs_diff(TARGET_THREADS);
    let tiles = sg_m * sg_n * tm * tn;
    let intensity = tm * tn * 100 / (tm + tn);
    (full, u32::MAX - distance, intensity, tiles, covered)
}

pub(crate) fn output_block(
    tile: MatrixTile,
    m: usize,
    k: usize,
    n: usize,
    sg_size: u32,
    max_threads: u32,
) -> Option<OutputBlock> {
    if tile.m == 0
        || sg_size < 4
        || max_threads < sg_size
        || !m.is_multiple_of(tile.m as usize)
        || !n.is_multiple_of(tile.n as usize)
        || !k.is_multiple_of(tile.k as usize)
        || k < MIN_MATRIX_K
        // N narrower than K is the split-K regime: 144×960×320 ran about
        // twice as slow on this tile with no bias. The wide layer
        // (N = 1280, K = 320) is where the unrolled tile is faster.
        || n < k
    {
        return None;
    }
    let max_sg = (max_threads / sg_size).clamp(1, MAX_SUBGROUPS);
    let k_tiles = (k / tile.k as usize) as u32;
    let mut best: Option<(u32, u32, u32, u32, usize)> = None;
    let mut best_key = (0u32, 0u32, 0u32, 0u32, 0usize);
    for tm in 1..=MAX_ACCUMULATORS {
        for tn in 1..=MAX_ACCUMULATORS {
            if tm * tn > MAX_ACCUMULATORS || (tm == 1 && tn == 1) {
                continue;
            }
            // tn == 1 never reuses A across N. 9216×144×24 took that block and
            // lost to split-K by about half. A tile has to reuse A.
            if tn == 1 || tm * tn * k_tiles < MIN_MMAS {
                continue;
            }
            for sg_m in 1..=max_sg {
                for sg_n in 1..=max_sg {
                    if sg_m * sg_n > max_sg {
                        continue;
                    }
                    let macro_m = (sg_m * tm * tile.m) as usize;
                    let macro_n = (sg_n * tn * tile.n) as usize;
                    if macro_m > m || macro_n > n || !n.is_multiple_of(macro_n) {
                        continue;
                    }
                    let covered = m - (m % macro_m);
                    if covered == 0 {
                        continue;
                    }
                    let key = block_score(sg_size, sg_m, sg_n, tm, tn, covered, m);
                    if best.is_none() || key > best_key {
                        best_key = key;
                        best = Some((tm, tn, sg_m, sg_n, covered));
                    }
                }
            }
        }
    }
    let (tm, tn, sg_m, sg_n, rows) = best?;
    let edge_rows = m - rows;
    if edge_rows == 0 {
        return Some(OutputBlock {
            tm,
            tn,
            sg_m,
            sg_n,
            sg_k: 1,
            rows,
            edge_tm: 1,
            edge_tn: 1,
            edge_sg_m: 1,
            edge_sg_n: 1,
            edge_rows: 0,
        });
    }
    let (edge_tm, edge_tn, edge_sg_m, edge_sg_n) =
        cover_block(tile, edge_rows, n, sg_size, max_sg)?;
    Some(OutputBlock {
        tm,
        tn,
        sg_m,
        sg_n,
        sg_k: 1,
        rows,
        edge_tm,
        edge_tn,
        edge_sg_m,
        edge_sg_n,
        edge_rows,
    })
}

/// A block that divides `m` and `n` exactly. The remainder path may be a
/// single hardware tile; it only has to finish the rows the main block left.
fn cover_block(
    tile: MatrixTile,
    m: usize,
    n: usize,
    sg_size: u32,
    max_sg: u32,
) -> Option<(u32, u32, u32, u32)> {
    let mut best = None;
    let mut best_key = (0u32, 0u32, 0u32, 0u32, 0usize);
    for tm in 1..=MAX_ACCUMULATORS {
        for tn in 1..=MAX_ACCUMULATORS {
            if tm * tn > MAX_ACCUMULATORS {
                continue;
            }
            for sg_m in 1..=max_sg {
                for sg_n in 1..=max_sg {
                    if sg_m * sg_n > max_sg {
                        continue;
                    }
                    let macro_m = (sg_m * tm * tile.m) as usize;
                    let macro_n = (sg_n * tn * tile.n) as usize;
                    if macro_m == 0
                        || macro_n == 0
                        || !m.is_multiple_of(macro_m)
                        || !n.is_multiple_of(macro_n)
                    {
                        continue;
                    }
                    let key = block_score(sg_size, sg_m, sg_n, tm, tn, m, m);
                    if best.is_none() || key > best_key {
                        best_key = key;
                        best = Some((tm, tn, sg_m, sg_n));
                    }
                }
            }
        }
    }
    best
}

pub(crate) fn knob_on(caps: &SubgroupCaps) -> bool {
    caps.subgroups && TRACT_WGPU_SUBGROUPS.get()
}

pub(crate) fn limits_of(q: &WgpuQueue) -> (u32, u32) {
    let limits = q.context().device().limits();
    (limits.max_compute_invocations_per_workgroup, limits.max_compute_workgroup_size_x)
}

fn gemv_plan(
    caps: &SubgroupCaps,
    max_inv: u32,
    max_x: u32,
    m: usize,
    epilogue_empty: bool,
) -> Option<GemvPlan> {
    if !epilogue_empty || m != 1 || !knob_on(caps) {
        return None;
    }
    let choice = chosen_subgroup(caps)?;
    if choice.varying {
        if choice.size > max_inv || choice.size > max_x {
            return None;
        }
        return Some(GemvPlan { size: choice.size, cols: 1, pin: false, varying: true });
    }
    let cols = gemv_cols(choice.size, max_inv, max_x)?;
    Some(GemvPlan { size: choice.size, cols, pin: choice.pin, varying: false })
}

pub(crate) fn stride_pair(tensor: &DeviceTensor, transpose: bool) -> (isize, isize) {
    let rank = tensor.rank();
    let row = tensor.strides()[rank - if transpose { 1 } else { 2 }];
    let col = tensor.strides()[rank - if transpose { 2 } else { 1 }];
    (row, col)
}

#[allow(clippy::too_many_arguments)]
fn matrix_plan(
    caps: &SubgroupCaps,
    max_inv: u32,
    max_x: u32,
    dt: ShaderDtype,
    f16_accumulation: bool,
    m: usize,
    k: usize,
    n: usize,
    prefix: usize,
    a: &DeviceTensor,
    b: &DeviceTensor,
    output: &DeviceTensor,
    t: Transposes,
) -> Option<MatrixPlan> {
    if !knob_on(caps) || !caps.matrix {
        return None;
    }
    let tile = match dt {
        ShaderDtype::F32 => caps.f32_tile?,
        ShaderDtype::F16 => {
            if !f16_accumulation {
                return None;
            }
            caps.f16_tile?
        }
    };
    let choice = chosen_subgroup(caps)?;
    // Every subgroup participates. Returning the extras before a cooperative
    // op is `UniformityRequirements(COOP_OPS)`.
    let max_threads = max_inv.min(max_x).min(256);
    if choice.varying || choice.size > max_threads {
        return None;
    }
    let block = output_block(tile, m, k, n, choice.size, max_threads)?;
    let gx = n / (block.sg_n * block.tn * tile.n) as usize;
    let gy = block.rows / (block.sg_m * block.tm * tile.m) as usize;
    let edge_gx = n / (block.edge_sg_n * block.edge_tn * tile.n) as usize;
    let edge_gy = if block.edge_rows == 0 {
        1
    } else {
        block.edge_rows / (block.edge_sg_m * block.edge_tm * tile.m) as usize
    };
    if gx == 0
        || gy == 0
        || edge_gx == 0
        || edge_gy == 0
        || gx > GRID_LIMIT as usize
        || gy > GRID_LIMIT as usize
        || edge_gx > GRID_LIMIT as usize
        || edge_gy > GRID_LIMIT as usize
        || prefix > GRID_LIMIT as usize
    {
        return None;
    }
    let (a_row, a_col) = stride_pair(a, t.a);
    let (b_row, b_col) = stride_pair(b, t.b);
    let (o_row, o_col) = stride_pair(output, t.c);
    let a_major = contiguous_major(a_row, a_col)?;
    let b_major = contiguous_major(b_row, b_col)?;
    let o_major = contiguous_major(o_row, o_col)?;
    if !major_fits(a_major, a_row, a_col, tile.k as isize, tile.m as isize)
        || !major_fits(b_major, b_row, b_col, tile.n as isize, tile.k as isize)
        || !major_fits(o_major, o_row, o_col, tile.n as isize, tile.m as isize)
    {
        return None;
    }
    Some(MatrixPlan {
        tile,
        size: choice.size,
        pin: choice.pin,
        major: [a_major, b_major, o_major],
        block,
    })
}

pub(crate) fn enables(dt: ShaderDtype, matrix: bool, pin: bool) -> String {
    let mut lines = Vec::new();
    #[cfg(target_arch = "wasm32")]
    {
        if matrix {
            lines.push("enable chromium_experimental_subgroup_matrix;");
        }
        lines.push("enable subgroups;");
        if pin {
            lines.push("enable subgroup_size_control;");
        }
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        if matrix {
            lines.push("enable wgpu_cooperative_matrix;");
        }
        let _ = pin;
    }
    if dt == ShaderDtype::F16 {
        lines.push("enable f16;");
    }
    if lines.is_empty() { String::new() } else { format!("{}\n", lines.join("\n")) }
}

pub(crate) fn bindings(dt: ShaderDtype, extras: usize, a_type: &str, b_type: &str) -> String {
    let t = dt.wgsl();
    let out_binding = 2 + extras;
    let uniform_binding = 3 + extras;
    let extra_bindings = (0..extras)
        .map(|i| {
            format!("@group(0) @binding({}) var<storage, read> extra{i}: array<{t}>;\n", 2 + i)
        })
        .collect::<String>();
    format!(
        r#"struct Params {{
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

@group(0) @binding(0) var<storage, read> a: {a_type};
@group(0) @binding(1) var<storage, read> b: {b_type};
{extra_bindings}@group(0) @binding({out_binding}) var<storage, read_write> outp: array<{t}>;
@group(0) @binding({uniform_binding}) var<uniform> params: Params;
{AT8}
"#
    )
}

fn flat_bindings(dt: ShaderDtype, extras: usize) -> String {
    let flat = format!("array<{}>", dt.wgsl());
    bindings(dt, extras, &flat, &flat)
}

const PREFIX_DECODE: &str = r#"    var a_i = params.off_a;
    var b_i = params.off_b;
    var o_i = params.off_out;
    var restp = pref;
    for (var j = 0u; j < 6u; j++) {
        let ax = 5u - j;
        let dim = max(at8(params.out_sh0, params.out_sh1, ax), 1u);
        let c = restp % dim;
        restp = restp / dim;
        a_i += c * at8(params.a_s0, params.a_s1, ax);
        b_i += c * at8(params.b_s0, params.b_s1, ax);
        o_i += c * at8(params.out_s0, params.out_s1, ax);
    }
"#;

fn pin_attr(pin: bool, size: u32) -> String {
    #[cfg(target_arch = "wasm32")]
    if pin {
        return format!("@subgroup_size({size})\n");
    }
    #[cfg(not(target_arch = "wasm32"))]
    let _ = (pin, size);
    String::new()
}

pub(crate) fn gemv_shader(
    dt: ShaderDtype,
    extras: usize,
    size: u32,
    cols: u32,
    pin: bool,
    varying: bool,
) -> String {
    let t = dt.wgsl();
    let suf = dt.suffix();
    let wg = cols * size;
    let attr = pin_attr(pin, size);
    let mut source = enables(dt, false, pin);
    source.push_str(&flat_bindings(dt, extras));
    let sg_param = if varying { ", @builtin(subgroup_size) sg_size: u32" } else { "" };
    source.push_str(&format!(
        "{attr}@compute @workgroup_size({wg}) fn matmul_gemv_sg_{suf}(@builtin(subgroup_invocation_id) lane: u32, @builtin(subgroup_id) sg: u32, @builtin(workgroup_id) wid: vec3<u32>{sg_param}) {{\n"
    ));
    let step = if varying { "sg_size".to_string() } else { format!("{size}u") };
    let vary_gate = if varying { "    if (sg != 0u) { return; }\n" } else { "" };
    source.push_str(&format!(
        r#"    let cols = {cols}u;
    let tiles_n = (params.n + cols - 1u) / cols;
    let tile = wid.y * {GRID_LIMIT}u + wid.x;
    if (tile >= params.prefix * tiles_n) {{ return; }}
    let pref = tile / tiles_n;
    let col = (tile % tiles_n) * cols + sg;
{vary_gate}"#
    ));
    source.push_str(PREFIX_DECODE);
    source.push_str(&format!(
        r#"    let a_k_s = select(at8(params.a_s0, params.a_s1, 7u), at8(params.a_s0, params.a_s1, 6u), params.ta != 0u);
    let b_k_s = select(at8(params.b_s0, params.b_s1, 6u), at8(params.b_s0, params.b_s1, 7u), params.tb != 0u);
    let b_n_s = select(at8(params.b_s0, params.b_s1, 7u), at8(params.b_s0, params.b_s1, 6u), params.tb != 0u);
    let o_n_s = select(at8(params.out_s0, params.out_s1, 7u), at8(params.out_s0, params.out_s1, 6u), params.tc != 0u);
    var acc = 0.0;
    if (col < params.n) {{
        let b_base = b_i + col * b_n_s;
        for (var k = lane; k < params.k; k += {step}) {{
            acc += f32(a[a_i + k * a_k_s]) * f32(b[b_base + k * b_k_s]);
        }}
    }}
    let sum = subgroupAdd(acc);
    if (lane == 0u && col < params.n) {{
        outp[o_i + col * o_n_s] = {t}(sum);
    }}
}}
"#
    ));
    source
}

fn stride_ident(major: AxisMajor, prefix: &str) -> String {
    match major {
        AxisMajor::Row => format!("{prefix}_row_s"),
        AxisMajor::Col => format!("{prefix}_col_s"),
    }
}

/// Four K tiles per iteration when `unroll` is set, with a constant trip
/// count. Two tiles was the previous native number; a fully expanded chain
/// was several times slower. Short K keeps one tile per step. Dawn's wall
/// time for this shader still trails split-K.
fn k_loop(tile_k: u32, sg_k: u32, k_limit: u32, unroll: bool, loads: &str, mmas: &str) -> String {
    if sg_k > 1 {
        return format!(
            "    let k_slice = params.k / {sg_k}u;\n    for (var part = 0u; part < k_slice; part = part + {tile_k}u) {{\n        let k0 = sg * k_slice + part;\n{loads}{mmas}    }}\n"
        );
    }
    if !unroll {
        return format!(
            "    for (var k0 = 0u; k0 < params.k; k0 += {tile_k}u) {{\n{loads}{mmas}    }}\n"
        );
    }
    let body = format!("{loads}{mmas}");
    // Four tiles when K divides the step, else two. Sixteen accumulators
    // spilled; the loads here are scoped so only the four accumulators stay live.
    let copies = if tile_k > 0 && k_limit.is_multiple_of(tile_k * 4) { 4 } else { 2 };
    let mut block = String::new();
    for i in 0..copies {
        let off = i * tile_k;
        let step =
            if off == 0 { body.clone() } else { body.replace("k0 *", &format!("(k0 + {off}u) *")) };
        block.push_str(&format!("        {{\n{step}        }}\n"));
    }
    let step = tile_k * copies;
    let tail = body.clone();
    format!(
        "    for (var k0 = 0u; k0 + {step}u <= {k_limit}u; k0 = k0 + {step}u) {{\n{block}    }}\n    for (var k0 = {k_limit}u / {step}u * {step}u; k0 < {k_limit}u; k0 = k0 + {tile_k}u) {{\n        {{\n{tail}        }}\n    }}\n"
    )
}

fn stage_base(sg_m: u32, sg_n: u32, sg_k: u32, span: u32) -> String {
    if sg_m * sg_n * sg_k <= 1 {
        "    let base = 0u;\n".to_string()
    } else {
        format!("    let base = sg * {span}u;\n")
    }
}

#[allow(clippy::too_many_arguments)]
fn matrix_body(
    tile: MatrixTile,
    major: [AxisMajor; 3],
    tm: u32,
    tn: u32,
    sg_m: u32,
    sg_n: u32,
    sg_k: u32,
    stage: bool,
    unroll: bool,
    k_limit: u32,
) -> Option<String> {
    if tm == 0 || tn == 0 || sg_m == 0 || sg_n == 0 || sg_k == 0 || tm * tn > MAX_ACCUMULATORS {
        return None;
    }
    let [a_major, b_major, o_major] = major;
    let a_stride = stride_ident(a_major, "a");
    let b_stride = stride_ident(b_major, "b");
    let o_stride = stride_ident(o_major, "o");
    let m = tile.m;
    let n = tile.n;
    let k = tile.k;
    let bm = m * tm;
    let bn = n * tn;
    let mut strides = r#"    let a_row_s = select(at8(params.a_s0, params.a_s1, 6u), at8(params.a_s0, params.a_s1, 7u), params.ta != 0u);
    let a_col_s = select(at8(params.a_s0, params.a_s1, 7u), at8(params.a_s0, params.a_s1, 6u), params.ta != 0u);
    let b_row_s = select(at8(params.b_s0, params.b_s1, 6u), at8(params.b_s0, params.b_s1, 7u), params.tb != 0u);
    let b_col_s = select(at8(params.b_s0, params.b_s1, 7u), at8(params.b_s0, params.b_s1, 6u), params.tb != 0u);
    let o_row_s = select(at8(params.out_s0, params.out_s1, 6u), at8(params.out_s0, params.out_s1, 7u), params.tc != 0u);
    let o_col_s = select(at8(params.out_s0, params.out_s1, 7u), at8(params.out_s0, params.out_s1, 6u), params.tc != 0u);
"#.to_string();
    // `subgroup_id` is uniform inside one subgroup, so the tile origin is a
    // uniform coop operand. No branch on it: Naga rejects a return before a
    // cooperative op.
    let origin = if sg_m == 1 && sg_n == 1 {
        format!(
            "    let row0 = params._p0 + wid.y * {bm}u;\n    let col0 = params._p1 + wid.x * {bn}u;\n"
        )
    } else {
        format!(
            "    let srow = sg / {sg_n}u;\n    let scol = sg % {sg_n}u;\n    let row0 = params._p0 + (wid.y * {sg_m}u + srow) * {bm}u;\n    let col0 = params._p1 + (wid.x * {sg_n}u + scol) * {bn}u;\n"
        )
    };
    strides.push_str(&origin);
    #[cfg(target_arch = "wasm32")]
    {
        let a_layout = match a_major {
            AxisMajor::Row => "row_major",
            AxisMajor::Col => "col_major",
        };
        let b_layout = match b_major {
            AxisMajor::Row => "row_major",
            AxisMajor::Col => "col_major",
        };
        let o_layout = match o_major {
            AxisMajor::Row => "row_major",
            AxisMajor::Col => "col_major",
        };
        let mut decls = String::new();
        let mut loads = String::new();
        let mut mmas = String::new();
        let mut stores = String::new();
        for i in 0..tm {
            for j in 0..tn {
                decls.push_str(&format!(
                    "    var acc_{i}_{j} = subgroup_matrix_result<{{ty}}, {n}, {m}>();\n"
                ));
            }
        }
        for i in 0..tm {
            let row = i * m;
            loads.push_str(&format!(
                "        let a_{i} = subgroupMatrixLoad<subgroup_matrix_left<{{ty}}, {k}, {m}>, {a_layout}>(&a, a_i + (row0 + {row}u) * a_row_s + k0 * a_col_s, {a_stride});\n"
            ));
        }
        for j in 0..tn {
            let col = j * n;
            loads.push_str(&format!(
                "        let b_{j} = subgroupMatrixLoad<subgroup_matrix_right<{{ty}}, {n}, {k}>, {b_layout}>(&b, b_i + k0 * b_row_s + (col0 + {col}u) * b_col_s, {b_stride});\n"
            ));
        }
        for i in 0..tm {
            for j in 0..tn {
                mmas.push_str(&format!(
                    "        acc_{i}_{j} = subgroupMatrixMultiplyAccumulate(a_{i}, b_{j}, acc_{i}_{j});\n"
                ));
            }
        }
        for i in 0..tm {
            for j in 0..tn {
                let row = i * m;
                let col = j * n;
                if stage {
                    stores.push_str(&format!(
                        "    subgroupMatrixStore<col_major>(&stage, base + {col}u * {bm}u + {row}u, acc_{i}_{j}, {bm}u);\n"
                    ));
                } else {
                    stores.push_str(&format!(
                        "    subgroupMatrixStore<{o_layout}>(&outp, o_i + (row0 + {row}u) * o_row_s + (col0 + {col}u) * o_col_s, acc_{i}_{j}, {o_stride});\n"
                    ));
                }
            }
        }
        strides.push_str(&decls);
        strides.push_str(&k_loop(k, sg_k, k_limit, unroll, &loads, &mmas));
        if stage {
            strides.push_str(&stage_base(sg_m, sg_n, sg_k, bm * bn));
        }
        strides.push_str(&stores);
        return Some(strides);
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let name = match (tile.m, tile.n, tile.k) {
            (8, 8, 8) => "coop_mat8x8",
            (16, 16, 16) => "coop_mat16x16",
            _ => return None,
        };
        let a_load = match a_major {
            AxisMajor::Row => "coopLoadT",
            AxisMajor::Col => "coopLoad",
        };
        let b_load = match b_major {
            AxisMajor::Row => "coopLoadT",
            AxisMajor::Col => "coopLoad",
        };
        let o_store = match o_major {
            AxisMajor::Row => "coopStoreT",
            AxisMajor::Col => "coopStore",
        };
        let mut decls = String::new();
        let mut loads = String::new();
        let mut mmas = String::new();
        let mut stores = String::new();
        for i in 0..tm {
            for j in 0..tn {
                decls.push_str(&format!("    var acc_{i}_{j} = {name}<{{ty}}, C>();\n"));
            }
        }
        for i in 0..tm {
            let row = i * m;
            loads.push_str(&format!(
                "        let a_{i} = {a_load}<{name}<{{ty}}, A>>(&a[a_i + (row0 + {row}u) * a_row_s + k0 * a_col_s], {a_stride});\n"
            ));
        }
        for j in 0..tn {
            let col = j * n;
            loads.push_str(&format!(
                "        let b_{j} = {b_load}<{name}<{{ty}}, B>>(&b[b_i + k0 * b_row_s + (col0 + {col}u) * b_col_s], {b_stride});\n"
            ));
        }
        for i in 0..tm {
            for j in 0..tn {
                mmas.push_str(&format!(
                    "        acc_{i}_{j} = coopMultiplyAdd(a_{i}, b_{j}, acc_{i}_{j});\n"
                ));
            }
        }
        for i in 0..tm {
            for j in 0..tn {
                let row = i * m;
                let col = j * n;
                if stage {
                    stores.push_str(&format!(
                        "    coopStore(acc_{i}_{j}, &stage[base + {col}u * {bm}u + {row}u], {bm}u);\n"
                    ));
                } else {
                    stores.push_str(&format!(
                        "    {o_store}(acc_{i}_{j}, &outp[o_i + (row0 + {row}u) * o_row_s + (col0 + {col}u) * o_col_s], {o_stride});\n"
                    ));
                }
            }
        }
        strides.push_str(&decls);
        strides.push_str(&k_loop(k, sg_k, k_limit, unroll, &loads, &mmas));
        if stage {
            strides.push_str(&stage_base(sg_m, sg_n, sg_k, bm * bn));
        }
        strides.push_str(&stores);
        Some(strides)
    }
}

/// Sum the K partials from subgroup 0 and write the output once. The barrier
/// is outside the branch so every subgroup arrives. No cooperative op follows
/// it: a branch on `subgroup_id` before a cooperative op is rejected.
fn k_split_reduce(
    dt: ShaderDtype,
    epilogue: &[ChainStep],
    rows: u32,
    cols: u32,
    size: u32,
    sg_k: u32,
) -> String {
    let t = dt.wgsl();
    let body = epilogue_body_at(epilogue, "col", "oe");
    let sum = (0..sg_k)
        .map(|s| format!("f32(stage[{s}u * span + cc * {rows}u + rr])"))
        .collect::<Vec<_>>()
        .join(" + ");
    format!(
        r#"    workgroupBarrier();
    if (sg == 0u) {{
        let span = {rows}u * {cols}u;
        for (var e = lane; e < span; e = e + {size}u) {{
            let rr = e / {cols}u;
            let cc = e % {cols}u;
            let row = row0 + rr;
            let col = col0 + cc;
            let oe = o_i - params.off_out + row * o_row_s + col * o_col_s;
            var v = {t}({sum});
{body}            outp[o_i + row * o_row_s + col * o_col_s] = v;
        }}
    }}
"#
    )
}

fn matrix_epilogue(
    dt: ShaderDtype,
    epilogue: &[ChainStep],
    rows: u32,
    cols: u32,
    size: u32,
) -> String {
    if epilogue.is_empty() {
        return String::new();
    }
    let t = dt.wgsl();
    let body = epilogue_body_at(epilogue, "col", "oe");
    format!(
        r#"    workgroupBarrier();
    let span = {rows}u * {cols}u;
    for (var e = lane; e < span; e = e + {size}u) {{
        let rr = e / {cols}u;
        let cc = e % {cols}u;
        let row = row0 + rr;
        let col = col0 + cc;
        let oe = o_i - params.off_out + row * o_row_s + col * o_col_s;
        var v = {t}(stage[base + cc * {rows}u + rr]);
{body}        outp[o_i + row * o_row_s + col * o_col_s] = v;
    }}
"#
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn matrix_shader(
    dt: ShaderDtype,
    epilogue: &[ChainStep],
    extras: usize,
    tile: MatrixTile,
    size: u32,
    pin: bool,
    major: [AxisMajor; 3],
    tm: u32,
    tn: u32,
    sg_m: u32,
    sg_n: u32,
    sg_k: u32,
    unroll: bool,
    k_limit: u32,
) -> Option<String> {
    let stage = !epilogue.is_empty() || sg_k > 1;
    let body = matrix_body(tile, major, tm, tn, sg_m, sg_n, sg_k, stage, unroll, k_limit)?;
    let body = body.replace("{ty}", dt.wgsl());
    if !body.contains(dt.wgsl()) {
        return None;
    }
    let threads = size.checked_mul(sg_m)?.checked_mul(sg_n)?.checked_mul(sg_k)?;
    let suf = dt.suffix();
    let attr = pin_attr(pin, size);
    let mut builtins = Vec::new();
    if stage {
        // Subgroup lane, not local_invocation_id. A workgroup of several
        // subgroups would otherwise epilogue the same span from every one.
        builtins.push("@builtin(subgroup_invocation_id) lane: u32");
    }
    if sg_m * sg_n > 1 || sg_k > 1 {
        builtins.push("@builtin(subgroup_id) sg: u32");
    }
    builtins.push("@builtin(workgroup_id) wid: vec3<u32>");
    let mut source = enables(dt, true, pin);
    source.push_str(&flat_bindings(dt, extras));
    if stage {
        let stage_len = sg_m
            .checked_mul(sg_n)?
            .checked_mul(sg_k)?
            .checked_mul(tile.m)?
            .checked_mul(tm)?
            .checked_mul(tile.n)?
            .checked_mul(tn)?;
        // The tile is stored here, the bias is applied, and device memory is
        // written once. Reloading the product from device memory was slower
        // than split-K on the biased convs.
        source.push_str(&format!("var<workgroup> stage: array<{}, {stage_len}>;\n", dt.wgsl()));
        source.push_str(&unary_ops_wgsl(dt.wgsl()));
        source.push_str(&binary_ops_wgsl(dt.wgsl()));
    }
    source.push_str(&format!(
        "{attr}@compute @workgroup_size({threads}) fn matmul_sgemm_{suf}({}) {{\n",
        builtins.join(", ")
    ));
    source.push_str("    let pref = wid.z;\n");
    source.push_str(PREFIX_DECODE);
    source.push_str(&body);
    if sg_k > 1 {
        source.push_str(&k_split_reduce(dt, epilogue, tile.m * tm, tile.n * tn, size, sg_k));
    } else {
        source.push_str(&matrix_epilogue(dt, epilogue, tile.m * tm, tile.n * tn, size));
    }
    source.push_str("}\n");
    Some(source)
}

#[allow(clippy::too_many_arguments)]
fn standard_uniform(
    a: &DeviceTensor,
    b: &DeviceTensor,
    extras: &[&DeviceTensor],
    output: &DeviceTensor,
    out_shape: &[usize],
    prefix: usize,
    m: usize,
    k: usize,
    n: usize,
    t: Transposes,
    row0: u32,
    col0: u32,
) -> TractResult<Vec<u32>> {
    let mut vals = vec![
        element_offset(a, 0) as u32,
        element_offset(b, 0) as u32,
        element_offset(output, 0) as u32,
        prefix as u32,
        m as u32,
        k as u32,
        n as u32,
        t.a as u32,
        t.b as u32,
        t.c as u32,
        row0,
        col0,
    ];
    vals.extend_from_slice(&rpad8_strides(a.shape(), a.strides(), out_shape));
    vals.extend_from_slice(&rpad8_strides(b.shape(), b.strides(), out_shape));
    vals.extend_from_slice(&rpad8_strides(out_shape, output.strides(), out_shape));
    vals.extend_from_slice(&rpad8_dims(out_shape));
    let mut off_extra = [0u32; 4];
    let mut mode_extra = [0u32; 4];
    for (i, extra) in extras.iter().enumerate().take(4) {
        off_extra[i] = element_offset(extra, 0) as u32;
        mode_extra[i] = crate::kernels::matmul::gemm_epilogue_mode(extra, n, output)?;
    }
    vals.extend_from_slice(&off_extra);
    vals.extend_from_slice(&mode_extra);
    Ok(vals)
}

#[allow(clippy::too_many_arguments)]
pub fn try_dispatch(
    q: &WgpuQueue,
    t: Transposes,
    epilogue: &[ChainStep],
    a: &DeviceTensor,
    b: &DeviceTensor,
    extras: &[&DeviceTensor],
    output: &DeviceTensor,
    dt: ShaderDtype,
    f16_accumulation: bool,
    m: usize,
    k: usize,
    n: usize,
    prefix: usize,
    out_shape: &[usize],
) -> TractResult<bool> {
    if prefix == 0 || n == 0 || k == 0 {
        return Ok(false);
    }
    if crate::kernels::coop_gemm::try_dispatch(
        q,
        t,
        epilogue,
        a,
        b,
        extras,
        output,
        dt,
        f16_accumulation,
        m,
        k,
        n,
        prefix,
        out_shape,
    )? {
        return Ok(true);
    }
    let caps = *q.context().subgroup_caps();
    let (max_inv, max_x) = limits_of(q);
    let empty = epilogue.is_empty();
    let layout = LayoutKind::Chain(3 + extras.len() as u8);
    if let Some(plan) =
        matrix_plan(&caps, max_inv, max_x, dt, f16_accumulation, m, k, n, prefix, a, b, output, t)
    {
        let block = plan.block;
        // The bound is baked into the shader. A loop on params.k stayed
        // serial in Dawn and lost to split-K there.
        let unroll = k >= 256;
        let Some(source_main) = matrix_shader(
            dt,
            epilogue,
            extras.len(),
            plan.tile,
            plan.size,
            plan.pin,
            plan.major,
            block.tm,
            block.tn,
            block.sg_m,
            block.sg_n,
            block.sg_k,
            unroll,
            k as u32,
        ) else {
            return Ok(false);
        };
        let source_edge = if block.edge_rows > 0 {
            let Some(source) = matrix_shader(
                dt,
                epilogue,
                extras.len(),
                plan.tile,
                plan.size,
                plan.pin,
                plan.major,
                block.edge_tm,
                block.edge_tn,
                block.edge_sg_m,
                block.edge_sg_n,
                1,
                unroll,
                k as u32,
            ) else {
                return Ok(false);
            };
            Some(source)
        } else {
            None
        };
        let major = plan.major;
        let pipeline = match q.context().chain_pipeline(
            (
                "matmul_sgemm",
                dt,
                plan.tile.m,
                plan.tile.n,
                plan.tile.k,
                plan.size,
                major,
                (block.tm, block.tn, block.sg_m, block.sg_n, block.sg_k, u32::from(unroll)),
                k as u32,
                epilogue.to_vec(),
                extras.len(),
            ),
            layout,
            EntryPoint::typed("matmul_sgemm", dt),
            || source_main,
        ) {
            Ok(pipeline) => pipeline,
            Err(e) => {
                log::warn!(
                    "tract-wgpu: subgroup matmul shader failed ({e}); keeping the existing kernel"
                );
                return Ok(false);
            }
        };
        let edge_pipeline = if let Some(source_edge) = source_edge {
            match q.context().chain_pipeline(
                (
                    "matmul_sgemm",
                    dt,
                    plan.tile.m,
                    plan.tile.n,
                    plan.tile.k,
                    plan.size,
                    major,
                    (
                        block.edge_tm,
                        block.edge_tn,
                        block.edge_sg_m,
                        block.edge_sg_n,
                        1u32,
                        u32::from(unroll),
                    ),
                    k as u32,
                    epilogue.to_vec(),
                    extras.len(),
                ),
                layout,
                EntryPoint::typed("matmul_sgemm", dt),
                || source_edge,
            ) {
                Ok(pipeline) => Some(pipeline),
                Err(e) => {
                    log::warn!(
                        "tract-wgpu: subgroup matmul edge shader failed ({e}); keeping the existing kernel"
                    );
                    return Ok(false);
                }
            }
        } else {
            None
        };
        dispatch_groups(
            q,
            "matmul_sgemm",
            &pipeline,
            layout,
            a,
            b,
            extras,
            output,
            out_shape,
            prefix,
            m,
            k,
            n,
            t,
            0,
            0,
            [
                n as u32 / (block.sg_n * block.tn * plan.tile.n),
                block.rows as u32 / (block.sg_m * block.tm * plan.tile.m),
                prefix as u32,
            ],
        )?;
        if let Some(edge_pipeline) = edge_pipeline {
            dispatch_groups(
                q,
                "matmul_sgemm",
                &edge_pipeline,
                layout,
                a,
                b,
                extras,
                output,
                out_shape,
                prefix,
                m,
                k,
                n,
                t,
                block.rows as u32,
                0,
                [
                    n as u32 / (block.edge_sg_n * block.edge_tn * plan.tile.n),
                    block.edge_rows as u32 / (block.edge_sg_m * block.edge_tm * plan.tile.m),
                    prefix as u32,
                ],
            )?;
        }
        return Ok(true);
    }
    if let Some(plan) = gemv_plan(&caps, max_inv, max_x, m, empty) {
        let pipeline = q.context().chain_pipeline(
            ("matmul_gemv_sg", dt, plan.size, plan.cols, extras.len()),
            layout,
            EntryPoint::typed("matmul_gemv_sg", dt),
            || gemv_shader(dt, extras.len(), plan.size, plan.cols, plan.pin, plan.varying),
        )?;
        let tiles = prefix * n.div_ceil(plan.cols as usize);
        let x = tiles.min(GRID_LIMIT as usize) as u32;
        let y = tiles.div_ceil(GRID_LIMIT as usize) as u32;
        return dispatch_groups(
            q,
            "matmul_gemv_sg",
            &pipeline,
            layout,
            a,
            b,
            extras,
            output,
            out_shape,
            prefix,
            m,
            k,
            n,
            t,
            0,
            0,
            [x, y, 1],
        )
        .map(|_| true);
    }
    Ok(false)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_groups(
    q: &WgpuQueue,
    label: &'static str,
    pipeline: &std::sync::Arc<wgpu::ComputePipeline>,
    layout: LayoutKind,
    a: &DeviceTensor,
    b: &DeviceTensor,
    extras: &[&DeviceTensor],
    output: &DeviceTensor,
    out_shape: &[usize],
    prefix: usize,
    m: usize,
    k: usize,
    n: usize,
    t: Transposes,
    row0: u32,
    col0: u32,
    groups: [u32; 3],
) -> TractResult<()> {
    let mut buffers: Vec<&crate::context::WgpuBuffer> =
        vec![get_wgpu_buffer(a), get_wgpu_buffer(b)];
    buffers.extend(extras.iter().map(|tensor| get_wgpu_buffer(tensor)));
    buffers.push(get_wgpu_buffer(output));
    let bg = q.context().bind_group(layout, &buffers, q.uniform())?;
    let vals = standard_uniform(a, b, extras, output, out_shape, prefix, m, k, n, t, row0, col0)?;
    let dyn_off = q.alloc_uniform(&pack_u32s(&vals))?;
    q.dispatch_grid(label, pipeline, &bg, dyn_off, groups)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn caps(min: u32, max: u32) -> SubgroupCaps {
        SubgroupCaps { subgroups: true, min_size: min, max_size: max, ..SubgroupCaps::default() }
    }

    #[test]
    fn subgroup_size_follows_the_adapter() {
        assert_eq!(
            chosen_subgroup(&caps(32, 32)),
            Some(SubgroupChoice { size: 32, pin: false, varying: false })
        );
        assert_eq!(
            chosen_subgroup(&caps(4, 64)),
            Some(SubgroupChoice { size: 64, pin: false, varying: true })
        );
        assert_eq!(chosen_subgroup(&caps(0, 0)), None);
        assert_eq!(gemv_cols(32, 1024, 1024), Some(8));
        assert_eq!(gemv_cols(64, 1024, 256), Some(4));
        assert_eq!(gemv_cols(128, 1024, 1024), Some(2));
        assert_eq!(gemv_cols(4, 256, 256), Some(64));
        assert_eq!(gemv_cols(32, 16, 1024), None);
        let probed = SubgroupCaps { fixed_size: 32, ..caps(4, 64) };
        assert_eq!(
            chosen_subgroup(&probed),
            Some(SubgroupChoice { size: 32, pin: false, varying: false })
        );
        assert_eq!(consensus_size(&[0, 32, 32, 0]), Some(32));
        assert_eq!(consensus_size(&[32, 64]), None);
        assert_eq!(consensus_size(&[0, 0]), None);
    }

    #[test]
    fn output_block_keeps_short_k_and_covers_rvm() {
        let tile = MatrixTile { m: 8, n: 8, k: 8 };
        // Subgroup 32, workgroup cap 256: up to eight subgroups.
        let pick = |m, k, n| output_block(tile, m, k, n, 32, 256);
        assert_eq!(pick(16, 16, 16), None);
        assert_eq!(pick(576, 88, 24), None);
        assert_eq!(pick(15105, 24, 144), None);
        assert_eq!(pick(9216, 144, 24), None);
        assert_eq!(pick(920, 160, 960), None);
        assert_eq!(pick(56, 128, 32), None);
        // N < K, so the narrow-N gate keeps this on split-K.
        assert_eq!(pick(56, 256, 32), None);
        // Same 2×2, four-subgroup block as MODNet 144, with a grid in both axes.
        // N = K so it is still on the matrix side of the narrow-N gate.
        assert_eq!(
            pick(48, 256, 256),
            Some(OutputBlock {
                tm: 2,
                tn: 2,
                sg_m: 1,
                sg_n: 4,
                sg_k: 1,
                rows: 48,
                edge_tm: 1,
                edge_tn: 1,
                edge_sg_m: 1,
                edge_sg_n: 1,
                edge_rows: 0,
            })
        );
        assert_eq!(
            pick(144, 320, 1280),
            Some(OutputBlock {
                tm: 2,
                tn: 2,
                sg_m: 1,
                sg_n: 4,
                sg_k: 1,
                rows: 144,
                edge_tm: 1,
                edge_tn: 1,
                edge_sg_m: 1,
                edge_sg_n: 1,
                edge_rows: 0,
            })
        );
        assert_eq!(pick(144, 960, 320), None);
        // One subgroup still covers MODNet 144. RVM's K stays on split-K.
        assert_eq!(
            output_block(tile, 144, 320, 1280, 32, 32),
            Some(OutputBlock {
                tm: 2,
                tn: 2,
                sg_m: 1,
                sg_n: 1,
                sg_k: 1,
                rows: 144,
                edge_tm: 1,
                edge_tn: 1,
                edge_sg_m: 1,
                edge_sg_n: 1,
                edge_rows: 0,
            })
        );
        let wide = matrix_shader(
            ShaderDtype::F16,
            &[],
            0,
            tile,
            32,
            false,
            [AxisMajor::Col; 3],
            2,
            2,
            1,
            1,
            1,
            false,
            0,
        )
        .expect("2x2 block");
        assert!(wide.contains("acc_1_1"));
        assert!(wide.contains("row0 + 8u"));
        assert!(wide.contains("@workgroup_size(32)"));
        assert!(!wide.contains("subgroup_id"));
        assert!(!wide.contains("if (sg"));
        let occupied = matrix_shader(
            ShaderDtype::F16,
            &[],
            0,
            tile,
            32,
            false,
            [AxisMajor::Col; 3],
            2,
            2,
            2,
            4,
            1,
            false,
            0,
        )
        .expect("occupied block");
        assert!(occupied.contains("@workgroup_size(256)"));
        assert!(occupied.contains("subgroup_id"));
        assert!(occupied.contains("let srow = sg / 4u"));
        assert!(occupied.contains("wid.y * 2u + srow"));
        assert!(!occupied.contains("if (sg"));
        let split = matrix_shader(
            ShaderDtype::F16,
            &[],
            0,
            tile,
            32,
            false,
            [AxisMajor::Col; 3],
            1,
            4,
            1,
            1,
            4,
            false,
            0,
        )
        .expect("k-split block");
        assert!(split.contains("@workgroup_size(128)"));
        assert!(split.contains("let k_slice = params.k / 4u"));
        assert!(split.contains("let k0 = sg * k_slice + part"));
        assert!(split.contains("if (sg == 0u)"));
        assert!(split.contains("workgroupBarrier"));
        assert!(split.contains("f32(stage[0u * span"));
        assert!(!split.contains("let srow"));
        let piped = matrix_shader(
            ShaderDtype::F16,
            &[],
            0,
            tile,
            32,
            false,
            [AxisMajor::Col; 3],
            2,
            2,
            1,
            4,
            1,
            true,
            320,
        )
        .expect("unrolled block");
        assert!(piped.contains("(k0 + 8u)"));
        assert!(piped.contains("(k0 + 24u)"));
        assert!(piped.contains("<= 320u"));
        assert!(!wide.contains("(k0 + 8u)"));
        #[cfg(not(target_arch = "wasm32"))]
        {
            assert_eq!(wide.matches("coopMultiplyAdd").count(), 4);
            assert_eq!(occupied.matches("coopMultiplyAdd").count(), 4);
        }
        #[cfg(target_arch = "wasm32")]
        {
            assert_eq!(wide.matches("subgroupMatrixMultiplyAccumulate").count(), 4);
            assert_eq!(occupied.matches("subgroupMatrixMultiplyAccumulate").count(), 4);
        }
    }

    #[test]
    fn native_tile_prefers_the_larger_square_match() {
        let props = [
            wgpu::CooperativeMatrixProperties {
                m_size: 8,
                n_size: 8,
                k_size: 8,
                ab_type: wgpu::CooperativeScalarType::F32,
                cr_type: wgpu::CooperativeScalarType::F32,
                saturating_accumulation: false,
            },
            wgpu::CooperativeMatrixProperties {
                m_size: 16,
                n_size: 16,
                k_size: 16,
                ab_type: wgpu::CooperativeScalarType::F16,
                cr_type: wgpu::CooperativeScalarType::F32,
                saturating_accumulation: false,
            },
            wgpu::CooperativeMatrixProperties {
                m_size: 16,
                n_size: 16,
                k_size: 16,
                ab_type: wgpu::CooperativeScalarType::F32,
                cr_type: wgpu::CooperativeScalarType::F32,
                saturating_accumulation: true,
            },
            wgpu::CooperativeMatrixProperties {
                m_size: 16,
                n_size: 16,
                k_size: 16,
                ab_type: wgpu::CooperativeScalarType::F32,
                cr_type: wgpu::CooperativeScalarType::F32,
                saturating_accumulation: false,
            },
        ];
        assert_eq!(
            pick_native_tile(&props, wgpu::CooperativeScalarType::F32),
            Some(MatrixTile { m: 16, n: 16, k: 16 })
        );
        assert_eq!(pick_native_tile(&props, wgpu::CooperativeScalarType::F16), None);
    }

    #[test]
    fn dawn_tile_keeps_any_matching_shape() {
        let configs = vec![
            ("f16".to_string(), "f32".to_string(), 8, 8, 8),
            ("f16".to_string(), "f16".to_string(), 8, 8, 8),
            ("f32".to_string(), "f32".to_string(), 8, 8, 16),
        ];
        assert_eq!(pick_dawn_tile(&configs, "f16"), Some(MatrixTile { m: 8, n: 8, k: 8 }));
        assert_eq!(pick_dawn_tile(&configs, "f32"), Some(MatrixTile { m: 8, n: 8, k: 16 }));
    }

    #[test]
    fn shader_dialect_matches_the_backend() {
        let gemv = gemv_shader(ShaderDtype::F32, 0, 32, 8, false, false);
        assert!(gemv.contains("subgroupAdd"));
        assert!(gemv.contains("subgroup_invocation_id"));
        assert!(gemv.contains("@workgroup_size(256)"));
        assert!(!gemv.contains("enable subgroup_size_control;"));
        let tile = MatrixTile { m: 8, n: 8, k: 8 };
        let row = [AxisMajor::Row; 3];
        let matrix =
            matrix_shader(ShaderDtype::F32, &[], 0, tile, 32, false, row, 1, 1, 1, 1, 1, false, 0)
                .expect("8x8 shader");
        assert!(!matrix.contains("sgid"));
        assert!(!matrix.contains("subgroup_id"));
        assert!(!matrix.contains("if (sgid != 0u)"));
        assert!(matrix.contains("@workgroup_size(32)"));
        #[cfg(target_arch = "wasm32")]
        {
            assert!(gemv.contains("enable subgroups;"));
            assert!(matrix.contains("enable chromium_experimental_subgroup_matrix;"));
            assert!(matrix.contains("enable subgroups;"));
            assert!(
                matrix.contains("subgroupMatrixLoad<subgroup_matrix_left<f32, 8, 8>, row_major>")
            );
            assert!(matrix.contains("subgroupMatrixMultiplyAccumulate"));
            assert!(!matrix.contains("wgpu_cooperative_matrix"));
            let pinned = gemv_shader(ShaderDtype::F16, 0, 32, 8, true, false);
            assert!(pinned.contains("enable subgroup_size_control;"));
            assert!(pinned.contains("@subgroup_size(32)"));
            assert!(pinned.contains("enable f16;"));
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            assert!(!gemv.contains("enable subgroups;"));
            assert!(matrix.contains("enable wgpu_cooperative_matrix;"));
            assert!(matrix.contains("coopLoadT<coop_mat8x8<f32, A>>"));
            assert!(matrix.contains("coopLoadT<coop_mat8x8<f32, B>>"));
            assert!(matrix.contains("coop_mat8x8<f32, C>()"));
            assert!(matrix.contains("coopMultiplyAdd"));
            assert!(matrix.contains("coopStoreT"));
            assert!(!matrix.contains("enable subgroups;"));
            assert!(!matrix.contains("chromium_experimental"));
            let varying = gemv_shader(ShaderDtype::F32, 0, 64, 1, false, true);
            assert!(varying.contains("@builtin(subgroup_size) sg_size"));
            assert!(varying.contains("k += sg_size"));
            assert!(varying.contains("if (sg != 0u) { return; }"));
            assert!(varying.contains("@workgroup_size(64)"));
            assert!(
                matrix_shader(
                    ShaderDtype::F32,
                    &[],
                    0,
                    MatrixTile { m: 32, n: 32, k: 32 },
                    32,
                    false,
                    row,
                    1,
                    1,
                    1,
                    1,
                    1,
                    false,
                    0,
                )
                .is_none()
            );
            let wide = matrix_shader(
                ShaderDtype::F16,
                &[],
                0,
                MatrixTile { m: 16, n: 16, k: 16 },
                32,
                false,
                row,
                1,
                1,
                1,
                1,
                1,
                false,
                0,
            )
            .expect("16x16");
            assert!(wide.contains("coop_mat16x16<f16, A>"));
            assert!(wide.contains("enable f16;"));
        }
    }

    #[test]
    fn column_major_bias_uses_the_other_load() {
        assert_eq!(contiguous_major(1, 144), Some(AxisMajor::Col));
        assert_eq!(contiguous_major(320, 1), Some(AxisMajor::Row));
        assert_eq!(contiguous_major(2, 4), None);
        assert!(major_fits(AxisMajor::Col, 1, 144, 8, 8));
        assert!(!major_fits(AxisMajor::Col, 1, 4, 8, 8));
        let bias = [ChainStep::Binary { op: "add".into(), rhs: 1, swapped: false }];
        let matrix = matrix_shader(
            ShaderDtype::F16,
            &bias,
            1,
            MatrixTile { m: 8, n: 8, k: 8 },
            32,
            false,
            [AxisMajor::Col; 3],
            1,
            1,
            1,
            1,
            1,
            false,
            0,
        )
        .expect("column-major shader");
        assert!(matrix.contains("op_add"));
        assert!(matrix.contains("subgroup_invocation_id"));
        assert!(matrix.contains("workgroupBarrier"));
        assert!(matrix.contains("stage[base"));
        assert!(!matrix.contains("storageBarrier"));
        assert!(!matrix.contains("lid.x"));
        assert!(matrix.contains("a_col_s"));
        #[cfg(target_arch = "wasm32")]
        {
            assert!(matrix.contains(", col_major>"));
            assert!(matrix.contains("subgroupMatrixStore<col_major>"));
            assert!(!matrix.contains("row_major"));
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            assert!(matrix.contains("coopLoad<coop_mat8x8<f16, A>>"));
            assert!(matrix.contains("coopLoad<coop_mat8x8<f16, B>>"));
            assert!(matrix.contains("coopStore("));
            assert!(!matrix.contains("coopLoadT"));
            assert!(!matrix.contains("coopStoreT"));
        }
    }

    /// Numeric check. The device is process-global, so this only runs when
    /// `TRACT_WGPU_SUBGROUPS=1` is set before the first device is created:
    /// `TRACT_WGPU_SUBGROUPS=1 cargo test -p tract-wgpu --lib subgroup_kernels_match_cpu -- --test-threads=1`.
    #[test]
    fn subgroup_kernels_match_cpu() -> TractResult<()> {
        if !TRACT_WGPU_SUBGROUPS.get() {
            eprintln!("subgroup numeric test skipped: TRACT_WGPU_SUBGROUPS is off");
            return Ok(());
        }
        if crate::wgpu_context_created() && !crate::wgpu_context().subgroup_caps().subgroups {
            eprintln!("subgroup numeric test skipped: device was created without subgroups");
            return Ok(());
        }
        use tract_core::ops::einsum::prefix_matmul::PrefixMatMul;
        let ctx = crate::wgpu_context();
        let caps = *ctx.subgroup_caps();
        eprintln!("subgroup caps {caps:?}");
        let bias = [ChainStep::Binary { op: "add".into(), rhs: 1, swapped: false }];
        let col_major = matrix_shader(
            ShaderDtype::F16,
            &bias,
            1,
            MatrixTile { m: 8, n: 8, k: 8 },
            32,
            false,
            [AxisMajor::Col; 3],
            1,
            1,
            1,
            1,
            1,
            true,
            32,
        )
        .context("column-major bias shader")?;
        ctx.chain_pipeline(
            ("subgroup-col-major-bias-compile",),
            LayoutKind::Chain(4),
            EntryPoint::typed("matmul_sgemm", ShaderDtype::F16),
            || col_major,
        )
        .context("column-major bias shader did not compile")?;
        let k_split = matrix_shader(
            ShaderDtype::F16,
            &bias,
            1,
            MatrixTile { m: 8, n: 8, k: 8 },
            32,
            false,
            [AxisMajor::Col; 3],
            1,
            4,
            1,
            1,
            4,
            false,
            0,
        )
        .context("k-split shader")?;
        ctx.chain_pipeline(
            ("subgroup-k-split-compile",),
            LayoutKind::Chain(4),
            EntryPoint::typed("matmul_sgemm", ShaderDtype::F16),
            || k_split,
        )
        .context("k-split shader did not compile")?;
        if !caps.subgroups {
            bail!("TRACT_WGPU_SUBGROUPS is on but the device has no subgroups: {caps:?}");
        }
        let limits = ctx.device().limits();
        if gemv_plan(
            &caps,
            limits.max_compute_invocations_per_workgroup,
            limits.max_compute_workgroup_size_x,
            1,
            true,
        )
        .is_none()
        {
            bail!("subgroup GEMV plan does not apply to this device: {caps:?}");
        }
        let op = PrefixMatMul {
            transpose_a: false,
            transpose_b: false,
            transpose_c: false,
            quantize_output: None,
            operating_dt: None,
        };
        let mut model = TypedModel::default();
        let a = model.add_source("a", f32::fact([1, 96]))?;
        let b = model
            .add_const("b", Tensor::from_shape(&[96, 13], &crate::test_utils::fill(1, 96 * 13))?)?;
        let y = model.wire_node("gemv", op, &[a, b])?[0];
        model.select_output_outlets(&[y])?;
        crate::test_utils::gpu_vs_cpu(
            model,
            Tensor::from_shape(&[1, 96], &crate::test_utils::fill(2, 96))?,
        )?;
        if caps.fixed_size > 0 && caps.f16_tile.is_some() {
            biased_block_matches_cpu()?;
        }
        if let Some(tile) = caps.f32_tile {
            let fixed = chosen_subgroup(&caps).is_some_and(|c| !c.varying);
            if !fixed {
                eprintln!(
                    "matrix tile {}x{}x{} not dispatched: subgroup size {}..{} is not fixed",
                    tile.m, tile.n, tile.k, caps.min_size, caps.max_size
                );
                let source = matrix_shader(
                    ShaderDtype::F32,
                    &[],
                    0,
                    tile,
                    32,
                    false,
                    [AxisMajor::Row; 3],
                    1,
                    1,
                    1,
                    1,
                    1,
                    false,
                    0,
                )
                .with_context(|| format!("no shader for tile {tile:?}"))?;
                let _module = ctx.device().create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("subgroup-matrix-uniform"),
                    source: wgpu::ShaderSource::Wgsl(source.into()),
                });
                eprintln!("matrix shader compiled; this device keeps the existing GEMM kernel");
            } else {
                let m = tile.m as usize * 2;
                let k = tile.k as usize * 2;
                let n = tile.n as usize * 2;
                let mut model = TypedModel::default();
                let a = model.add_source("a", f32::fact([m, k]))?;
                let b = model.add_const(
                    "b",
                    Tensor::from_shape(&[k, n], &crate::test_utils::fill(3, k * n))?,
                )?;
                let y = model.wire_node("sgemm", op, &[a, b])?[0];
                model.select_output_outlets(&[y])?;
                crate::test_utils::gpu_vs_cpu(
                    model,
                    Tensor::from_shape(&[m, k], &crate::test_utils::fill(4, m * k))?,
                )
                .with_context(|| format!("tile {}x{}x{}", tile.m, tile.n, tile.k))?;
            }
        }
        Ok(())
    }

    /// 48×256×256 takes the same 2×2 four-subgroup block as MODNet 144 and is
    /// past both gates, so the dispatch is the unrolled `matmul_sgemm`. Batched,
    /// because a single product of this layout takes the cooperative GEMM. The
    /// values are an f16 dot plus the bias column.
    fn biased_block_matches_cpu() -> TractResult<()> {
        use tract_gpu::tensor::IntoDevice;
        let (m, k, n) = (48usize, 256usize, 256usize);
        let pack = |seed: u64, len: usize| -> Vec<f16> {
            let mut s = seed ^ 0x9e37_79b9_7f4a_7c15;
            (0..len)
                .map(|_| {
                    s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    f16::from_f32(((s >> 33) as f32) / (u32::MAX as f32) * 0.3 - 0.15)
                })
                .collect()
        };
        let a_h = pack(11, k * m);
        let b_h = pack(12, n * k);
        let bias_h = pack(13, n);
        let a =
            Tensor::from_shape(&[2, k, m], &[a_h.clone(), a_h.clone()].concat())?.into_device()?;
        let b =
            Tensor::from_shape(&[2, n, k], &[b_h.clone(), b_h.clone()].concat())?.into_device()?;
        let bias = Tensor::from_shape(&[n], &bias_h)?.into_device()?;
        let extras = [&bias];
        let epilogue = [ChainStep::Binary { op: "add".into(), rhs: 1, swapped: false }];
        let output = DeviceTensor::uninitialized_dt(f16::datum_type(), &[2, n, m])?;
        crate::set_dispatch_capture(true);
        let _ = crate::take_dispatch_labels();
        crate::kernels::matmul::wgpu_matmul_dispatch(
            crate::kernels::matmul::Transposes { a: true, b: true, c: true },
            &epilogue,
            &a,
            &b,
            &extras,
            &output,
            true,
        )?;
        let got = pollster::block_on(crate::to_host_async(&output))?;
        let view = got.to_plain_array_view::<f16>()?;
        let got = view.as_slice().context("block output is not contiguous")?;
        let labels = crate::take_dispatch_labels();
        let mut worst = 0.0f32;
        for mi in 0..m {
            for ni in 0..n {
                let mut acc = f16::from_f32(0.0);
                for ki in 0..k {
                    acc += a_h[ki * m + mi] * b_h[ni * k + ki];
                }
                acc += bias_h[ni];
                for batch in 0..2 {
                    let have = got[(batch * n + ni) * m + mi].to_f32();
                    worst = worst.max((have - acc.to_f32()).abs());
                }
            }
        }
        let shown = labels.join(" ");
        eprintln!("BLOCK m={m} k={k} n={n} max_abs={worst:.4} [{shown}]");
        ensure!(labels.contains(&"matmul_sgemm"), "block ran [{shown}], expected matmul_sgemm");
        ensure!(worst <= 0.08, "block max abs {worst:.4} via [{shown}]");
        Ok(())
    }

    /// GPU time for pointwise shapes from the segmentation models.
    /// Channels-first 1×1 lowering: A is `[k, m]`, B is `[n, k]`, all three
    /// transposes set. Set `TRACT_WGPU_SUBGROUP_TIMING=1`.
    #[test]
    fn subgroup_model_layers() -> TractResult<()> {
        if std::env::var("TRACT_WGPU_SUBGROUP_TIMING").ok().as_deref() != Some("1") {
            eprintln!("subgroup model timing skipped");
            return Ok(());
        }
        let ctx = crate::wgpu_context();
        eprintln!("subgroup caps {:?}", ctx.subgroup_caps());
        let cases = [
            ("mp-se", 1usize, 128usize, 32usize),
            ("mp-se", 1, 32, 128),
            ("mp-se", 1, 96, 24),
            ("mp-se", 1, 24, 96),
            ("mp-se", 1, 16, 8),
            ("mp-pw", 2304, 16, 72),
            ("mp-pw", 9216, 16, 16),
            ("mp-pw", 576, 88, 24),
            ("modnet", 144, 320, 1280),
            ("modnet", 144, 960, 320),
            ("modnet", 36864, 16, 96),
            ("modnet", 9216, 144, 24),
            ("lp-xl", 15105, 24, 144),
            ("lp-xs", 2457, 16, 96),
            ("rvm", 920, 160, 960),
        ];
        // Unchanged kernels swing by about 2× between runs on this machine.
        // Three rounds is enough to take a median of the shapes whose label
        // changes.
        for round in 0..3 {
            eprintln!("ROUND {round}");
            for (name, m, k, n) in cases {
                time_layer(name, m, k, n, false)?;
                time_layer(name, m, k, n, true)?;
            }
        }
        Ok(())
    }

    fn time_layer(name: &str, m: usize, k: usize, n: usize, bias: bool) -> TractResult<()> {
        use tract_core::ops::einsum::prefix_matmul::PrefixMatMul;
        let dt = f16::datum_type();
        let pack = |seed, len| -> Vec<f16> {
            crate::test_utils::fill(seed, len)
                .into_iter()
                .map(|x| f16::from_f32(x * 0.05))
                .collect()
        };
        let mut model = TypedModel::default();
        let a = model.add_source("a", dt.fact([k, m]))?;
        let b = model.add_const("b", Tensor::from_shape(&[n, k], &pack(1, n * k))?)?;
        let op = PrefixMatMul {
            transpose_a: true,
            transpose_b: true,
            transpose_c: true,
            quantize_output: None,
            operating_dt: None,
        };
        let mut y = model.wire_node("mm", op, &[a, b])?[0];
        if bias {
            let bias_t = model.add_const("bias", Tensor::from_shape(&[n, 1], &pack(2, n))?)?;
            y = model.wire_node("bias_add", tract_core::ops::math::add(), &[y, bias_t])?[0];
        }
        model.select_output_outlets(&[y])?;
        let runnable = tract_core::runtime::runtime_for_name("wgpu")?
            .context("wgpu runtime")?
            .prepare(model)?;
        let input = Tensor::from_shape(&[k, m], &pack(3, k * m))?;
        for _ in 0..2 {
            runnable.run(tvec!(input.clone().into()))?;
        }
        let iters: usize = if m == 1 { 40 } else { 8 };
        let (_out, profile) = crate::with_wgpu_queue(|q| {
            q.profile(|| {
                for _ in 0..iters {
                    runnable.run(tvec!(input.clone().into()))?;
                }
                Ok(())
            })
        })?;
        let nanos: f64 = profile.iter().map(|(_, t)| t.nanos).sum();
        let labels: Vec<String> = profile
            .iter()
            .map(|(label, t)| format!("{label}x{}", t.calls / iters.max(1)))
            .collect();
        eprintln!(
            "LAYER {name} m={m} k={k} n={n} bias={bias} {:.2} us [{}]",
            nanos / iters as f64 / 1_000.0,
            labels.join(" ")
        );
        Ok(())
    }
}
