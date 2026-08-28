//! Fused SDPA via tract-owned ports of the MLX attention kernels (see
//! mlx_sdpa.metal): a decode-specialized `sdpa_vector` family (single-pass +
//! split-KV 2-pass) and the `steel_attention` tiled prefill kernel. Native
//! GQA (`gqa_factor`), f32/f16, causal or no mask. Dispatch mirrors MLX's
//! decision tree; unsupported shapes fall back to `MetalMfaSdpa` or the
//! explode path via the chooser translator at the bottom of this file.

use crate::encoder::EncoderExt;
use crate::{ConstantValues, LibraryName, MetalStream, Value};
use anyhow::ensure;
use metal::{MTLSize, NSUInteger};
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

/// Head dims with vector (decode) kernel instantiations.
const VECTOR_DIMS: &[usize] = &[64, 96, 128, 256];
/// Head dims with steel (prefill) kernel instantiations.
const STEEL_DIMS: &[usize] = &[64, 80, 128];
/// Split-KV blocks for the 2-pass vector path (must be a multiple of 32).
const TWO_PASS_BLOCKS: i32 = 32;

/// Last character of `-[MTLDevice architecture].name` (macOS 14+), e.g. 'p' for
/// "applegpu_g16p". mlx classifies GPU size by this suffix and only splits the
/// KV scan across blocks on the large ones.
/// `-[MTLDevice architecture].name` (macOS 14+), e.g. "applegpu_g16p".
// The objc msg_send!/sel! macros expand to a cargo-clippy cfg check, which
// older toolchains report at the call site; the module-level allow covers the
// expansion on every rustc.
#[allow(unexpected_cfgs)]
mod device_arch {
    use metal::foreign_types::ForeignTypeRef;
    use objc::runtime::Object;
    use objc::{msg_send, sel, sel_impl};

    pub fn name() -> Option<String> {
        unsafe {
            let device = metal::Device::system_default()?;
            let dev: *mut Object = device.as_ref().as_ptr() as *mut Object;
            let responds: bool = msg_send![dev, respondsToSelector: sel!(architecture)];
            if !responds {
                return None;
            }
            let arch: *mut Object = msg_send![dev, architecture];
            if arch.is_null() {
                return None;
            }
            let name_obj: *mut Object = msg_send![arch, name];
            if name_obj.is_null() {
                return None;
            }
            let cstr: *const std::os::raw::c_char = msg_send![name_obj, UTF8String];
            if cstr.is_null() {
                return None;
            }
            Some(std::ffi::CStr::from_ptr(cstr).to_string_lossy().into_owned())
        }
    }
}

/// Last character of the GPU architecture name. mlx classifies GPU size by this
/// suffix and only splits the KV scan across blocks on the large ones.
fn device_arch_suffix() -> Option<char> {
    static SUFFIX: std::sync::OnceLock<Option<char>> = std::sync::OnceLock::new();
    *SUFFIX.get_or_init(|| device_arch::name()?.chars().next_back())
}

/// mlx's split-KV routing (cpp:748): only large GPUs split at 1024 keys, and
/// everything else waits for a GQA cache of at least 4096. Splitting earlier
/// costs more than it saves.
fn use_two_pass(hq: usize, hkv: usize, kl: usize) -> bool {
    let large = matches!(device_arch_suffix(), Some('d') | Some('s'));
    (large && kl >= 1024) || (hkv < hq && kl >= 4096)
}

fn vector_tname(dt: DatumType) -> TractResult<&'static str> {
    match dt {
        DatumType::F32 => Ok("float"),
        DatumType::F16 => Ok("float16_t"),
        _ => bail!("MLX sdpa_vector: unsupported dt {dt:?}"),
    }
}

fn steel_tname(dt: DatumType) -> TractResult<&'static str> {
    match dt {
        DatumType::F32 => Ok("float32"),
        DatumType::F16 => Ok("float16"),
        _ => bail!("MLX steel attention: unsupported dt {dt:?}"),
    }
}

fn natural_strides_of(shape: &[usize]) -> TVec<isize> {
    let mut strides = tvec![1isize; shape.len()];
    for ix in (0..shape.len().saturating_sub(1)).rev() {
        strides[ix] = strides[ix + 1] * shape[ix + 1] as isize;
    }
    strides
}

fn ensure_natural(t: &DeviceTensor, what: &str) -> TractResult<()> {
    ensure!(
        t.strides() == natural_strides_of(t.shape()).as_slice(),
        "MLX SDPA expects contiguous {what}, got shape {:?} strides {:?}",
        t.shape(),
        t.strides()
    );
    Ok(())
}

/// Mirror of MLX `AttnParams` (steel/attn/params.h) — keep field order in sync.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct AttnParams {
    b: i32,
    h: i32,
    d: i32,
    ql: i32,
    kl: i32,
    gqa_factor: i32,
    scale: f32,
    nq: i32,
    nk: i32,
    nq_aligned: i32,
    nk_aligned: i32,
    ql_rem: i32,
    kl_rem: i32,
    ql_off: i32,
    q_strides: [i64; 3],
    k_strides: [i64; 3],
    v_strides: [i64; 3],
    o_strides: [i64; 3],
}

/// Mask strides in elements for the vector kernels, which index a mask as
/// `[B, H, qL, kL]` broadcast over any axis of extent 1.
struct MaskStrides {
    head: i32,
    q_seq: i32,
    kv_seq: i32,
}

fn mask_bhqk_strides(
    mask: &DeviceTensor,
    hq: usize,
    ql: usize,
    kl: usize,
) -> TractResult<MaskStrides> {
    ensure!(mask.rank() == 4, "MLX SDPA mask must be rank 4, got {:?}", mask.shape());
    let (sh, st) = (mask.shape(), mask.strides());
    ensure!(
        sh[0] == 1 && (sh[1] == 1 || sh[1] == hq) && (sh[2] == 1 || sh[2] == ql) && sh[3] == kl,
        "MLX SDPA mask shape {sh:?} not broadcastable to [1, {hq}, {ql}, {kl}]"
    );
    let stride_of = |axis: usize| if sh[axis] == 1 { 0 } else { st[axis] as i32 };
    Ok(MaskStrides { head: stride_of(1), q_seq: stride_of(2), kv_seq: st[3] as i32 })
}

/// Single-pass decode kernel: one threadgroup per (batch*head, q position),
/// 32 simdgroups striding over keys. Grid mirrors mlx cpp:358.
#[allow(clippy::too_many_arguments)]
fn dispatch_sdpa_vector_1pass(
    stream: &MetalStream,
    dt: DatumType,
    scale: f32,
    do_causal: bool,
    (b, hq, hkv, ql, kl, d): (usize, usize, usize, usize, usize, usize),
    q: &DeviceTensor,
    k: &DeviceTensor,
    v: &DeviceTensor,
    mask: Option<&DeviceTensor>,
    out: &DeviceTensor,
) -> TractResult<()> {
    let name = format!("sdpa_vector_{t}_{d}_{d}", t = vector_tname(dt)?);
    let mask_strides = mask.map(|m| mask_bhqk_strides(m, hq, ql, kl)).transpose()?;
    let constants = Some(ConstantValues::new(vec![
        (20, Value::Bool(mask.is_some())), // has_mask
        (21, Value::Bool(false)),          // query_transposed
        (22, Value::Bool(do_causal)),      // do_causal
        (23, Value::Bool(false)),          // bool_mask
        (24, Value::Bool(mask.is_some())), // float_mask
        (25, Value::Bool(false)),          // has_sinks
    ]));
    let pipeline = stream.load_pipeline_with_constants(LibraryName::MlxSdpa, &name, constants)?;

    let gqa_factor = (hq / hkv) as i32;
    let command_buffer = stream.command_buffer();
    command_buffer.encode(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_metal_tensor(0, q, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(1, k, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(2, v, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(3, out, metal::MTLResourceUsage::Write);
        encoder.set_slice(4, &[gqa_factor]);
        encoder.set_slice(5, &[kl as i32]);
        encoder.set_slice(6, &[k.strides()[1] as u64]);
        encoder.set_slice(7, &[k.strides()[2] as u64]);
        encoder.set_slice(8, &[v.strides()[1] as u64]);
        encoder.set_slice(9, &[v.strides()[2] as u64]);
        encoder.set_slice(10, &[scale]);
        if let (Some(m), Some(ms)) = (mask, mask_strides.as_ref()) {
            encoder.set_metal_tensor(12, m, metal::MTLResourceUsage::Read);
            encoder.set_slice(13, &[ms.kv_seq]);
            encoder.set_slice(14, &[ms.q_seq]);
            encoder.set_slice(15, &[ms.head]);
        }
        let grid = MTLSize { width: (b * hq) as _, height: ql as _, depth: 1 };
        let group = MTLSize { width: 1024, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(grid, group);
    });
    Ok(())
}

/// Split-KV decode: pass 1 writes per-block partials (one simdgroup per
/// (q head, q pos, kv block)), pass 2 reduces. Grids mirror mlx cpp:484/583.
#[allow(clippy::too_many_arguments)]
fn dispatch_sdpa_vector_2pass(
    stream: &MetalStream,
    dt: DatumType,
    scale: f32,
    do_causal: bool,
    (b, hq, hkv, ql, kl, d): (usize, usize, usize, usize, usize, usize),
    q: &DeviceTensor,
    k: &DeviceTensor,
    v: &DeviceTensor,
    mask: Option<&DeviceTensor>,
    out: &DeviceTensor,
) -> TractResult<()> {
    let mask_strides = mask.map(|m| mask_bhqk_strides(m, hq, ql, kl)).transpose()?;
    let blocks = TWO_PASS_BLOCKS as usize;
    let partials = unsafe { DeviceTensor::uninitialized_dt(dt, &[b, hq, ql, blocks, d])? };
    let sums = unsafe { DeviceTensor::uninitialized_dt(f32::datum_type(), &[b, hq, ql, blocks])? };
    let maxs = unsafe { DeviceTensor::uninitialized_dt(f32::datum_type(), &[b, hq, ql, blocks])? };
    stream.retain_tensor(&partials);
    stream.retain_tensor(&sums);
    stream.retain_tensor(&maxs);

    let tname = vector_tname(dt)?;
    let gqa_factor = hq / hkv;

    // Pass 1
    let name = format!("sdpa_vector_2pass_1_{tname}_{d}_{d}");
    let constants = Some(ConstantValues::new(vec![
        (20, Value::Bool(mask.is_some())),
        (21, Value::Bool(false)),
        (22, Value::Bool(do_causal)),
        (23, Value::Bool(false)),
        (24, Value::Bool(mask.is_some())),
        (25, Value::Bool(false)),
        (26, Value::I32(TWO_PASS_BLOCKS)), // blocks
    ]));
    let pipeline = stream.load_pipeline_with_constants(LibraryName::MlxSdpa, &name, constants)?;
    let command_buffer = stream.command_buffer();
    command_buffer.encode(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_metal_tensor(0, q, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(1, k, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(2, v, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(3, &partials, metal::MTLResourceUsage::Write);
        encoder.set_metal_tensor(4, &sums, metal::MTLResourceUsage::Write);
        encoder.set_metal_tensor(5, &maxs, metal::MTLResourceUsage::Write);
        // buffer 6 intentionally unset (matches mlx host, cpp:534)
        encoder.set_slice(7, &[kl as i32]);
        encoder.set_slice(8, &[k.strides()[1] as u64]);
        encoder.set_slice(9, &[k.strides()[2] as u64]);
        encoder.set_slice(10, &[v.strides()[1] as u64]);
        encoder.set_slice(11, &[v.strides()[2] as u64]);
        encoder.set_slice(12, &[scale]);
        if let (Some(m), Some(ms)) = (mask, mask_strides.as_ref()) {
            encoder.set_metal_tensor(14, m, metal::MTLResourceUsage::Read);
            encoder.set_slice(15, &[ms.kv_seq]);
            encoder.set_slice(16, &[ms.q_seq]);
            encoder.set_slice(17, &[ms.head]);
        }
        let grid = MTLSize { width: hkv as _, height: b as _, depth: blocks as _ };
        let group = MTLSize { width: 32, height: gqa_factor as _, depth: ql as _ };
        encoder.dispatch_thread_groups(grid, group);
    });

    // Pass 2
    let name = format!("sdpa_vector_2pass_2_{tname}_{d}");
    let pipeline = stream.load_pipeline(LibraryName::MlxSdpa, &name)?;
    let command_buffer = stream.command_buffer();
    command_buffer.encode(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_metal_tensor(0, &partials, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(1, &sums, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(2, &maxs, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(3, out, metal::MTLResourceUsage::Write);
        encoder.set_slice(4, &[TWO_PASS_BLOCKS]);
        let grid = MTLSize { width: (b * hq) as _, height: ql as _, depth: 1 };
        let group = MTLSize { width: 1024, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(grid, group);
    });
    Ok(())
}

/// Mirror of MLX `AttnMaskParams` (steel/attn/params.h): mask strides over
/// (B, H, qL), with the kL stride fixed at 1.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct AttnMaskParams {
    m_strides: [i64; 3],
}

/// Steel tiled prefill kernel: 4 simdgroups per threadgroup, one threadgroup
/// per (q block, q head, batch). Grid mirrors mlx cpp:160.
#[allow(clippy::too_many_arguments)]
fn dispatch_steel_attention(
    stream: &MetalStream,
    dt: DatumType,
    scale: f32,
    do_causal: bool,
    (b, hq, hkv, ql, kl, d): (usize, usize, usize, usize, usize, usize),
    q: &DeviceTensor,
    k: &DeviceTensor,
    v: &DeviceTensor,
    mask: Option<&DeviceTensor>,
    out: &DeviceTensor,
) -> TractResult<()> {
    let (bq, bk) = (32usize, if d < 128 { 32usize } else { 16usize });
    let tname = steel_tname(dt)?;
    let name = format!("steel_attention_{tname}_bq{bq}_bk{bk}_bd{d}_wm4_wn1_mask{tname}");
    let mask_params = mask
        .map(|m| -> TractResult<AttnMaskParams> {
            let ms = mask_bhqk_strides(m, hq, ql, kl)?;
            Ok(AttnMaskParams { m_strides: [0, ms.head as i64, ms.q_seq as i64] })
        })
        .transpose()?;
    let constants = Some(ConstantValues::new(vec![
        (200, Value::Bool(ql % bq == 0)),   // align_Q
        (201, Value::Bool(kl % bk == 0)),   // align_K
        (300, Value::Bool(mask.is_some())), // has_mask
        (301, Value::Bool(do_causal)),      // do_causal
        (302, Value::Bool(false)),          // has_sinks
    ]));
    let pipeline = stream.load_pipeline_with_constants(LibraryName::MlxSdpa, &name, constants)?;

    let nq = ql.div_ceil(bq);
    let nk = kl.div_ceil(bk);
    let params = AttnParams {
        b: b as i32,
        h: hq as i32,
        d: d as i32,
        ql: ql as i32,
        kl: kl as i32,
        gqa_factor: (hq / hkv) as i32,
        scale,
        nq: nq as i32,
        nk: nk as i32,
        nq_aligned: (ql / bq) as i32,
        nk_aligned: (kl / bk) as i32,
        ql_rem: (ql % bq) as i32,
        kl_rem: (kl % bk) as i32,
        ql_off: kl.saturating_sub(ql) as i32,
        q_strides: [q.strides()[0] as i64, q.strides()[1] as i64, q.strides()[2] as i64],
        k_strides: [k.strides()[0] as i64, k.strides()[1] as i64, k.strides()[2] as i64],
        v_strides: [v.strides()[0] as i64, v.strides()[1] as i64, v.strides()[2] as i64],
        o_strides: [out.strides()[0] as i64, out.strides()[1] as i64, out.strides()[2] as i64],
    };

    let command_buffer = stream.command_buffer();
    command_buffer.encode(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_metal_tensor(0, q, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(1, k, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(2, v, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(3, out, metal::MTLResourceUsage::Write);
        encoder.set_slice(4, std::slice::from_ref(&params));
        if let (Some(m), Some(ms)) = (mask, mask_params.as_ref()) {
            encoder.set_slice(5, std::slice::from_ref(ms));
            encoder.set_metal_tensor(6, m, metal::MTLResourceUsage::Read);
        }
        let grid = MTLSize { width: nq as _, height: hq as _, depth: b as _ };
        let group = MTLSize { width: 32, height: 4, depth: 1 };
        encoder.dispatch_thread_groups(grid, group);
    });
    Ok(())
}

/// Fused SDPA over `[B,Hq,Sq,D]` Q / `[B,Hkv,Sk,D]` K,V (GQA when Hkv < Hq).
/// Picks the decode (vector / split-KV) or prefill (steel) kernel following
/// MLX's dispatch tree; causal is bottom-right aligned (`qL_off = kL - qL`).
#[allow(clippy::too_many_arguments)]
pub fn dispatch_mlx_sdpa(
    stream: &MetalStream,
    scale: f32,
    is_causal: bool,
    q: &DeviceTensor,
    k: &DeviceTensor,
    v: &DeviceTensor,
    mask: Option<&DeviceTensor>,
    out: &DeviceTensor,
) -> TractResult<()> {
    let dt = q.datum_type();
    ensure!(matches!(dt, DatumType::F32 | DatumType::F16), "MLX SDPA: F32/F16 only");
    ensure!(q.rank() == 4 && k.rank() == 4 && v.rank() == 4, "MLX SDPA expects rank-4 inputs");
    let (b, hq, ql, d) = (q.shape()[0], q.shape()[1], q.shape()[2], q.shape()[3]);
    let (hkv, kl) = (k.shape()[1], k.shape()[2]);
    ensure!(k.shape()[3] == d && v.shape()[3] == d, "MLX SDPA expects equal head dims");
    ensure!(v.shape()[1] == hkv && v.shape()[2] == kl, "K/V layout mismatch");
    ensure!(hq % hkv == 0, "q heads ({hq}) must be a multiple of kv heads ({hkv})");
    for (t, w) in [(q, "Q"), (k, "K"), (v, "V"), (out, "O")] {
        ensure_natural(t, w)?;
    }

    if let Some(m) = mask {
        ensure!(m.datum_type() == dt, "MLX SDPA mask dt {:?} != {dt:?}", m.datum_type());
        ensure_natural(m, "mask")?;
        stream.retain_tensor(m);
    }
    stream.retain_tensor(q);
    stream.retain_tensor(k);
    stream.retain_tensor(v);
    stream.retain_tensor(out);

    let gqa_factor = hq / hkv;
    let shape6 = (b, hq, hkv, ql, kl, d);
    let vector_ok = VECTOR_DIMS.contains(&d) && ql <= 8 && ql <= kl && ql * gqa_factor <= 32;
    if vector_ok {
        // mlx forces causal off for single-position queries (cpp:746)
        let do_causal = is_causal && ql > 1;
        if use_two_pass(hq, hkv, kl) {
            dispatch_sdpa_vector_2pass(stream, dt, scale, do_causal, shape6, q, k, v, mask, out)
        } else {
            dispatch_sdpa_vector_1pass(stream, dt, scale, do_causal, shape6, q, k, v, mask, out)
        }
    } else {
        ensure!(
            STEEL_DIMS.contains(&d),
            "MLX SDPA: no kernel for head dim {d} with query len {ql} (translator gate too wide?)"
        );
        ensure!(!is_causal || ql <= kl, "causal SDPA needs qL <= kL, got {ql} > {kl}");
        dispatch_steel_attention(stream, dt, scale, is_causal, shape6, q, k, v, mask, out)
    }
}

/// Metal device op: fused SDPA via the ported MLX kernels.
#[derive(Debug, Clone)]
pub struct MetalMlxSdpa {
    pub scale: f32,
    pub is_causal: bool,
}

impl PartialEq for MetalMlxSdpa {
    fn eq(&self, o: &Self) -> bool {
        self.scale.to_bits() == o.scale.to_bits() && self.is_causal == o.is_causal
    }
}
impl Eq for MetalMlxSdpa {}
impl std::hash::Hash for MetalMlxSdpa {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.scale.to_bits().hash(state);
        self.is_causal.hash(state);
    }
}

impl Op for MetalMlxSdpa {
    fn name(&self) -> StaticName {
        "MetalMlxSdpa".into()
    }
    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("scale={} causal={}", self.scale, self.is_causal)])
    }
    op_as_typed_op!();
}

impl EvalOp for MetalMlxSdpa {
    fn is_pure_function(&self) -> bool {
        true
    }
    fn eval(&self, ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        use tract_gpu::tensor::DeviceTensorExt;
        ensure!((3..=4).contains(&inputs.len()), "MetalMlxSdpa expects Q,K,V[,mask]");
        let q = inputs[0].to_device_tensor()?;
        let k = inputs[1].to_device_tensor()?;
        let v = inputs[2].to_device_tensor()?;
        let mask = inputs.get(3).map(|m| m.to_device_tensor()).transpose()?;
        ensure!(q.rank() == 4, "expects rank-4 [B,H,Sq,D], got {:?}", q.shape());
        let out = tract_gpu::turn_handler::make_tensor_for_node(ctx, q.datum_type(), q.shape())?;
        crate::with_metal_stream(|stream| {
            dispatch_mlx_sdpa(stream, self.scale, self.is_causal, q, k, v, mask, &out)
        })?;
        Ok(tvec![out.into_tensor().into_tvalue()])
    }
}

impl TypedOp for MetalMlxSdpa {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        tract_gpu::utils::facts_to_device_facts(inputs, |f| Ok(tvec![f[0].without_value()]))
    }
    as_op!();
}

/// Whether an `Sdpa` node can be fused by the MLX kernels: exactly Q,K,V
/// (causal or no external mask), f16/f32, rank-4, concrete heads with
/// `Hq % Hkv == 0`, equal concrete head dim. Steel dims ({64,80,128}) cover
/// any sequence lengths; vector-only dims ({96,256}) additionally need
/// translate-time proof of decode-shape eligibility.
pub fn mlx_sdpa_supported(
    op: &tract_transformers::ops::sdpa::Sdpa,
    in_facts: &[&TypedFact],
) -> bool {
    if !(3..=4).contains(&in_facts.len()) {
        return false;
    }
    let (q, k, v) = (in_facts[0], in_facts[1], in_facts[2]);
    if !matches!(q.datum_type, DatumType::F16 | DatumType::F32) || !op.acc_datum_type.is_float() {
        return false;
    }
    if let Some(mask) = in_facts.get(3) {
        // additive float mask only, [1, 1|H, 1|Sq, Sk], broadcast handled by strides
        if mask.datum_type != q.datum_type || mask.rank() != 4 {
            return false;
        }
        let dims: Option<Vec<usize>> = mask.shape.iter().map(|d| d.to_usize().ok()).collect();
        let (Some(md), Some(qd)) =
            (dims, q.shape.iter().map(|d| d.to_usize().ok()).collect::<Option<Vec<_>>>())
        else {
            return false;
        };
        let kl = k.shape[2].to_usize().ok();
        if md[0] != 1 || md[3] != kl.unwrap_or(usize::MAX) {
            return false;
        }
        if !(md[1] == 1 || md[1] == qd[1]) || !(md[2] == 1 || md[2] == qd[2]) {
            return false;
        }
    }
    if q.rank() != 4 || k.rank() != 4 || v.rank() != 4 {
        return false;
    }
    let heads =
        (q.shape[1].to_usize().ok(), k.shape[1].to_usize().ok(), v.shape[1].to_usize().ok());
    let (Some(qh), Some(kh), Some(vh)) = heads else { return false };
    if kh != vh || qh % kh != 0 {
        return false;
    }
    let dims = (q.shape[3].to_usize().ok(), k.shape[3].to_usize().ok(), v.shape[3].to_usize().ok());
    let (Some(qd), Some(kd), Some(vd)) = dims else { return false };
    if qd != kd || qd != vd {
        return false;
    }
    if STEEL_DIMS.contains(&qd) {
        return true;
    }
    if VECTOR_DIMS.contains(&qd) {
        // No steel fallback at these dims: require provable decode shape.
        let lens = (q.shape[2].to_usize().ok(), k.shape[2].to_usize().ok());
        let (Some(ql), Some(kl)) = lens else { return false };
        return ql <= 8 && ql <= kl && ql * (qh / kh) <= 32;
    }
    false
}

// Single Sdpa translator: prefer the MLX port (GQA, decode kernel), fall back
// to the vendored MFA metallib, else explode via flatten_unfused_sdpa.
crate::register_metal_op!(tract_transformers::ops::sdpa::Sdpa, |source, node, op| {
    let in_facts = source.node_input_facts(node.id)?;
    let mlx = mlx_sdpa_supported(op, &in_facts);
    let mfa = crate::kernels::matmul::mfa::mfa_sdpa_supported(op, &in_facts);
    if !mlx && !mfa {
        return Ok(None);
    }
    let head_dim = in_facts[0].shape[in_facts[0].rank() - 1].to_usize()?;
    let scale = match &op.scale {
        Some(t) => t.cast_to_scalar::<f32>()?,
        None => (head_dim as f32).recip().sqrt(),
    };
    if mlx {
        Ok(Some(Box::new(MetalMlxSdpa { scale, is_causal: op.is_causal }) as Box<dyn TypedOp>))
    } else {
        Ok(Some(Box::new(crate::kernels::matmul::mfa::MetalMfaSdpa {
            scale,
            is_causal: op.is_causal,
        }) as Box<dyn TypedOp>))
    }
});

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::with_borrowed_metal_stream;
    use tract_gpu::tensor::IntoDevice;

    fn cpu_reference(
        dt: DatumType,
        is_causal: bool,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
    ) -> TractResult<Tensor> {
        let cpu = tract_transformers::ops::sdpa::Sdpa {
            scale: None,
            datum_type: dt,
            acc_datum_type: f32::datum_type(),
            is_causal,
        };
        Ok(cpu.eval(
            &EvalContext::pure(),
            tvec![q.clone().into_tvalue(), k.clone().into_tvalue(), v.clone().into_tvalue()],
        )?[0]
            .clone()
            .into_tensor())
    }

    fn pseudo<F: Datum + num_traits::Float>(shape: &[usize], seed: i64) -> Tensor
    where
        f32: num_traits::AsPrimitive<F>,
    {
        use num_traits::AsPrimitive;
        let n: usize = shape.iter().product();
        let data: Vec<F> = (0..n)
            .map(|i| {
                let x = (((i as i64 * 2654435761 + seed).rem_euclid(2000)) as f32 / 1000.0) - 1.0;
                x.as_()
            })
            .collect();
        Tensor::from_shape(shape, &data).unwrap()
    }

    #[allow(clippy::too_many_arguments)]
    fn run_case(
        dt: DatumType,
        b: usize,
        hq: usize,
        hkv: usize,
        ql: usize,
        kl: usize,
        d: usize,
        is_causal: bool,
    ) -> TractResult<()> {
        let (q, k, v) = if dt == f16::datum_type() {
            (
                pseudo::<f16>(&[b, hq, ql, d], 1),
                pseudo::<f16>(&[b, hkv, kl, d], 2),
                pseudo::<f16>(&[b, hkv, kl, d], 3),
            )
        } else {
            (
                pseudo::<f32>(&[b, hq, ql, d], 1),
                pseudo::<f32>(&[b, hkv, kl, d], 2),
                pseudo::<f32>(&[b, hkv, kl, d], 3),
            )
        };
        let reference = cpu_reference(dt, is_causal, &q, &k, &v)?;
        let scale = (d as f32).recip().sqrt();
        let metal = with_borrowed_metal_stream(|stream| {
            let qd = q.clone().into_device()?;
            let kd = k.clone().into_device()?;
            let vd = v.clone().into_device()?;
            let out = unsafe { DeviceTensor::uninitialized_dt(dt, &[b, hq, ql, d])? };
            dispatch_mlx_sdpa(stream, scale, is_causal, &qd, &kd, &vd, None, &out)?;
            stream.wait_until_completed()?;
            Ok(out.to_host()?.into_tensor())
        })?;
        reference.close_enough(&metal, Approximation::Approximate).with_context(|| {
            format!("dt={dt:?} b={b} hq={hq} hkv={hkv} ql={ql} kl={kl} d={d} causal={is_causal}")
        })
    }

    // `use_two_pass` only picks the split-KV kernel on some devices and shapes,
    // so its cases dispatch it directly instead of going through the chooser.
    #[allow(clippy::too_many_arguments)]
    fn run_two_pass_case(
        dt: DatumType,
        b: usize,
        hq: usize,
        hkv: usize,
        ql: usize,
        kl: usize,
        d: usize,
        is_causal: bool,
    ) -> TractResult<()> {
        let (q, k, v) = if dt == f16::datum_type() {
            (
                pseudo::<f16>(&[b, hq, ql, d], 1),
                pseudo::<f16>(&[b, hkv, kl, d], 2),
                pseudo::<f16>(&[b, hkv, kl, d], 3),
            )
        } else {
            (
                pseudo::<f32>(&[b, hq, ql, d], 1),
                pseudo::<f32>(&[b, hkv, kl, d], 2),
                pseudo::<f32>(&[b, hkv, kl, d], 3),
            )
        };
        let reference = cpu_reference(dt, is_causal, &q, &k, &v)?;
        let scale = (d as f32).recip().sqrt();
        let metal = with_borrowed_metal_stream(|stream| {
            let qd = q.clone().into_device()?;
            let kd = k.clone().into_device()?;
            let vd = v.clone().into_device()?;
            let out = unsafe { DeviceTensor::uninitialized_dt(dt, &[b, hq, ql, d])? };
            dispatch_sdpa_vector_2pass(
                stream,
                dt,
                scale,
                is_causal && ql > 1,
                (b, hq, hkv, ql, kl, d),
                &qd,
                &kd,
                &vd,
                None,
                &out,
            )?;
            stream.wait_until_completed()?;
            Ok(out.to_host()?.into_tensor())
        })?;
        reference.close_enough(&metal, Approximation::Approximate).with_context(|| {
            format!("2pass dt={dt:?} b={b} hq={hq} hkv={hkv} ql={ql} kl={kl} d={d}")
        })
    }

    // Additive-mask path: the shape a causal LLM export actually emits, where the
    // mask is a separate [1, 1, qL, kL] input rather than the `is_causal` flag.
    #[allow(clippy::too_many_arguments)]
    fn run_masked_case(
        dt: DatumType,
        b: usize,
        hq: usize,
        hkv: usize,
        ql: usize,
        kl: usize,
        d: usize,
        mask_heads: usize,
    ) -> TractResult<()> {
        let (q, k, v) = if dt == f16::datum_type() {
            (
                pseudo::<f16>(&[b, hq, ql, d], 1),
                pseudo::<f16>(&[b, hkv, kl, d], 2),
                pseudo::<f16>(&[b, hkv, kl, d], 3),
            )
        } else {
            (
                pseudo::<f32>(&[b, hq, ql, d], 1),
                pseudo::<f32>(&[b, hkv, kl, d], 2),
                pseudo::<f32>(&[b, hkv, kl, d], 3),
            )
        };
        let mut m = vec![0f32; mask_heads * ql * kl];
        for h in 0..mask_heads {
            for i in 0..ql {
                for j in 0..kl {
                    if j > i + (kl - ql) {
                        m[(h * ql + i) * kl + j] = -1e30;
                    }
                }
            }
        }
        let mask = Tensor::from_shape(&[1, mask_heads, ql, kl], &m)?.cast_to_dt(dt)?.into_owned();
        let cpu = tract_transformers::ops::sdpa::Sdpa {
            scale: None,
            datum_type: dt,
            acc_datum_type: f32::datum_type(),
            is_causal: false,
        };
        let reference = cpu.eval(
            &EvalContext::pure(),
            tvec![
                q.clone().into_tvalue(),
                k.clone().into_tvalue(),
                v.clone().into_tvalue(),
                mask.clone().into_tvalue()
            ],
        )?[0]
            .clone()
            .into_tensor();
        let scale = (d as f32).recip().sqrt();
        let got = with_borrowed_metal_stream(|stream| {
            let qd = q.clone().into_device()?;
            let kd = k.clone().into_device()?;
            let vd = v.clone().into_device()?;
            let md = mask.clone().into_device()?;
            let out = unsafe { DeviceTensor::uninitialized_dt(dt, &[b, hq, ql, d])? };
            dispatch_mlx_sdpa(stream, scale, false, &qd, &kd, &vd, Some(&md), &out)?;
            stream.wait_until_completed()?;
            Ok(out.to_host()?.into_tensor())
        })?;
        reference.close_enough(&got, Approximation::Approximate).with_context(|| {
            format!("masked dt={dt:?} b={b} hq={hq} hkv={hkv} ql={ql} kl={kl} d={d}")
        })
    }

    #[test]
    fn masked_vector_f32() -> TractResult<()> {
        run_masked_case(f32::datum_type(), 1, 8, 8, 1, 128, 64, 1)
    }

    #[test]
    fn masked_vector_f16_gqa() -> TractResult<()> {
        run_masked_case(f16::datum_type(), 1, 8, 2, 1, 192, 128, 1)
    }

    #[test]
    fn masked_vector_per_head() -> TractResult<()> {
        run_masked_case(f32::datum_type(), 1, 4, 4, 2, 96, 64, 4)
    }

    #[test]
    fn masked_steel_f32() -> TractResult<()> {
        run_masked_case(f32::datum_type(), 1, 4, 4, 64, 64, 64, 1)
    }

    #[test]
    fn masked_steel_f16_gqa_unaligned() -> TractResult<()> {
        run_masked_case(f16::datum_type(), 1, 8, 2, 48, 70, 128, 1)
    }

    #[test]
    fn vector_1pass_f32() -> TractResult<()> {
        run_case(f32::datum_type(), 1, 8, 8, 1, 64, 64, false)
    }

    #[test]
    fn vector_1pass_f32_gqa() -> TractResult<()> {
        run_case(f32::datum_type(), 1, 8, 2, 4, 128, 128, false)
    }

    #[test]
    fn vector_1pass_f32_causal() -> TractResult<()> {
        run_case(f32::datum_type(), 1, 4, 4, 4, 64, 64, true)
    }

    #[test]
    fn vector_1pass_f16() -> TractResult<()> {
        run_case(f16::datum_type(), 1, 8, 8, 1, 256, 96, false).ok();
        run_case(f16::datum_type(), 1, 8, 8, 1, 256, 256, false)
    }

    #[test]
    fn vector_1pass_batched() -> TractResult<()> {
        run_case(f32::datum_type(), 3, 4, 2, 2, 65, 64, false)
    }

    #[test]
    fn vector_2pass_f32() -> TractResult<()> {
        run_two_pass_case(f32::datum_type(), 1, 8, 8, 1, 2048, 64, false)
    }

    #[test]
    fn vector_2pass_f32_gqa_causal() -> TractResult<()> {
        run_two_pass_case(f32::datum_type(), 1, 8, 2, 4, 1536, 128, true)
    }

    #[test]
    fn vector_2pass_f16() -> TractResult<()> {
        run_two_pass_case(f16::datum_type(), 2, 4, 4, 2, 1024, 64, false)
    }

    #[test]
    fn steel_f32_aligned() -> TractResult<()> {
        run_case(f32::datum_type(), 1, 4, 4, 64, 64, 64, false)
    }

    #[test]
    fn steel_f32_unaligned() -> TractResult<()> {
        run_case(f32::datum_type(), 1, 4, 4, 37, 53, 64, false)
    }

    #[test]
    fn steel_f32_gqa() -> TractResult<()> {
        run_case(f32::datum_type(), 1, 8, 2, 128, 128, 128, false)
    }

    #[test]
    fn steel_f32_causal() -> TractResult<()> {
        run_case(f32::datum_type(), 1, 4, 4, 96, 160, 64, true)
    }

    #[test]
    fn steel_f32_d80() -> TractResult<()> {
        run_case(f32::datum_type(), 1, 4, 4, 64, 64, 80, false)
    }

    #[test]
    fn steel_f16() -> TractResult<()> {
        run_case(f16::datum_type(), 1, 4, 4, 64, 96, 128, false)
    }

    #[test]
    fn steel_f16_gqa_causal_unaligned() -> TractResult<()> {
        run_case(f16::datum_type(), 2, 8, 4, 33, 47, 64, true)
    }

    // qL <= 8 at a steel-only dim must take the steel kernel (no vector inst).
    #[test]
    fn steel_decode_d80() -> TractResult<()> {
        run_case(f32::datum_type(), 1, 4, 4, 1, 100, 80, false)
    }

    // Which decode kernel wins at a given cache length, for tuning `use_two_pass`
    // on a new device.
    //   cargo test -p tract-metal bench_vector_passes -- --ignored --nocapture
    #[test]
    #[ignore]
    fn bench_vector_passes() -> TractResult<()> {
        use std::time::Instant;
        let (b, hq, hkv, ql, d) = (1usize, 8usize, 8usize, 1usize, 64usize);
        println!("  arch suffix {:?}", device_arch_suffix());
        with_borrowed_metal_stream(|stream| {
            for kl in [1024usize, 4096, 16384] {
                let q = Tensor::zero::<f32>(&[b, hq, ql, d])?.into_device()?;
                let k = Tensor::zero::<f32>(&[b, hkv, kl, d])?.into_device()?;
                let v = Tensor::zero::<f32>(&[b, hkv, kl, d])?.into_device()?;
                let o =
                    unsafe { DeviceTensor::uninitialized_dt(f32::datum_type(), &[b, hq, ql, d])? };
                let shape6 = (b, hq, hkv, ql, kl, d);
                let time = |two: bool| -> TractResult<f64> {
                    let run = || -> TractResult<()> {
                        let f = if two {
                            dispatch_sdpa_vector_2pass
                        } else {
                            dispatch_sdpa_vector_1pass
                        };
                        f(stream, f32::datum_type(), 0.125, false, shape6, &q, &k, &v, None, &o)
                    };
                    for _ in 0..5 {
                        run()?;
                    }
                    stream.wait_until_completed()?;
                    let mut best = f64::MAX;
                    for _ in 0..5 {
                        let t = Instant::now();
                        for _ in 0..50 {
                            run()?;
                        }
                        stream.wait_until_completed()?;
                        best = best.min(t.elapsed().as_secs_f64() / 50.0);
                    }
                    Ok(best)
                };
                let (one, two) = (time(false)?, time(true)?);
                println!(
                    "  kvL={kl:<6} 1-pass {:7.4} ms   2-pass {:7.4} ms   gate picks {}",
                    one * 1e3,
                    two * 1e3,
                    if use_two_pass(hq, hkv, kl) { "2-pass" } else { "1-pass" }
                );
            }
            Ok(())
        })
    }
}
