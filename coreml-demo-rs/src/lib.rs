//! C ABI for the macOS demo app. Swift owns the camera + UI + compositor;
//! we own model loading and per-frame inference. The bridge is intentionally
//! tiny — five functions described in the handout.

use std::cell::RefCell;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Result, anyhow};
use half::f16;
use objc2_core_ml::MLComputeUnits;

use tract_core::prelude::*;
use tract_core::transform::{ModelTransform, get_transform};
use tract_coreml::CoremlTransform;
use tract_metal::MetalTransform;
use tract_onnx::prelude::*;

const MODNET_PATHS: &[&str] = &[
    "~/coding/v7-webgl-relighting-worktree/models/onnx/modnet_fp16.onnx",
    "~/coding/v7-webgl-relighting-worktree/models/onnx/modnet.onnx",
];

const RVM_PATH: &str =
    "~/coding/dfn3-wasm-relaxed-simd/models/rvm_mobilenetv3_fp32_baked.onnx";

thread_local! {
    static LAST_ERROR: RefCell<String> = const { RefCell::new(String::new()) };
}

fn set_error(e: impl std::fmt::Display) {
    LAST_ERROR.with(|s| *s.borrow_mut() = format!("{e}"));
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ModelKind {
    Modnet,
    Rvm,
}

pub struct ModelHandle {
    runnable: Arc<TypedRunnableModel>,
    h: usize,
    w: usize,
    /// True if the runnable expects f16 input/output. False for the CPU path
    /// where we skip `f32_to_f16` and run the model in f32.
    is_f16: bool,
    kind: ModelKind,
}

#[unsafe(no_mangle)]
pub extern "C" fn tract_demo_init() {}

/// Returns 0 on success, nonzero on failure (call `tract_demo_get_last_error`).
///
/// `model_kind`: 0 = MODNet, 1 = RVM
/// `backend`: 0 = TractCpu, 1 = TractMetal, 2 = TractCoreML
/// `coreml_compute_units`: ignored unless backend == 2.
///   0 = CPUOnly, 1 = CPUAndGPU, 2 = CPUAndNeuralEngine, 3 = All
#[unsafe(no_mangle)]
pub extern "C" fn tract_demo_create_model(
    model_kind: u32,
    h: u32,
    w: u32,
    backend: u32,
    coreml_compute_units: u32,
    out_handle: *mut *mut ModelHandle,
) -> i32 {
    if out_handle.is_null() {
        set_error("out_handle is null");
        return -1;
    }
    match build_runnable(model_kind, h, w, backend, coreml_compute_units) {
        Ok(handle) => {
            unsafe { *out_handle = Box::into_raw(Box::new(handle)) };
            0
        }
        Err(e) => {
            set_error(format!("create_model: {e:#}"));
            unsafe { *out_handle = std::ptr::null_mut() };
            -1
        }
    }
}

/// Run one frame. `src_rgb_f16` is `[3 * H * W]` half-float bytes (CHW order).
/// `out_alpha_f16` is caller-allocated `[1 * H * W]`.
#[unsafe(no_mangle)]
pub extern "C" fn tract_demo_run_frame(
    handle: *mut ModelHandle,
    src_rgb_f16: *const u16,
    out_alpha_f16: *mut u16,
    out_ms_elapsed: *mut f64,
) -> i32 {
    if handle.is_null() || src_rgb_f16.is_null() || out_alpha_f16.is_null() {
        set_error("run_frame: null pointer");
        return -1;
    }
    let handle = unsafe { &mut *handle };
    let src_len = 3 * handle.h * handle.w;
    let out_len = handle.h * handle.w;
    let src = unsafe { std::slice::from_raw_parts(src_rgb_f16 as *const f16, src_len) };
    let dst = unsafe { std::slice::from_raw_parts_mut(out_alpha_f16 as *mut f16, out_len) };

    match run_frame_inner(handle, src, dst) {
        Ok(ms) => {
            if !out_ms_elapsed.is_null() {
                unsafe { *out_ms_elapsed = ms };
            }
            0
        }
        Err(e) => {
            set_error(format!("run_frame: {e:#}"));
            -1
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tract_demo_destroy_model(handle: *mut ModelHandle) {
    if !handle.is_null() {
        let _ = unsafe { Box::from_raw(handle) };
    }
}

/// Copies the last thread-local error into `buf`. Returns the byte length of
/// the error message. If `buf_len` is too small, copies what fits and still
/// returns the *full* length.
#[unsafe(no_mangle)]
pub extern "C" fn tract_demo_get_last_error(buf: *mut u8, buf_len: usize) -> usize {
    LAST_ERROR.with(|s| {
        let bytes = s.borrow().as_bytes().to_vec();
        if !buf.is_null() && buf_len > 0 {
            let n = bytes.len().min(buf_len);
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf, n) };
        }
        bytes.len()
    })
}

// ---- internals ----

fn resolve_path(p: &str) -> PathBuf {
    if let Some(s) = p.strip_prefix("~/") {
        let home = std::env::var("HOME").expect("HOME unset");
        PathBuf::from(home).join(s)
    } else {
        PathBuf::from(p)
    }
}

fn load_modnet(h: u32, w: u32, cast_to_f16: bool) -> Result<TypedModel> {
    let mut last_err: Option<anyhow::Error> = None;
    for raw in MODNET_PATHS {
        let p = resolve_path(raw);
        if !p.exists() {
            continue;
        }
        match load_onnx_simple(&p, h, w, cast_to_f16) {
            Ok(m) => return Ok(m),
            Err(e) => last_err = Some(e),
        }
    }
    Err(last_err
        .unwrap_or_else(|| anyhow!("no MODNet ONNX found at known paths: {MODNET_PATHS:?}")))
}

/// Single-input ONNX load (MODNet). Sets input slot 0 to `[1, 3, H, W]` F32,
/// declutters, optionally casts to F16.
fn load_onnx_simple(
    path: &PathBuf,
    h: u32,
    w: u32,
    cast_to_f16: bool,
) -> Result<TypedModel> {
    let mut inf = tract_onnx::onnx().model_for_path(path)?;
    inf.set_input_fact(
        0,
        InferenceFact::dt_shape(
            f32::datum_type(),
            tvec![1i64, 3, h as i64, w as i64],
        ),
    )?;
    inf.analyse(true)?;
    let model = inf.into_typed()?.into_decluttered()?;
    let mut model = model;
    if cast_to_f16 {
        get_transform("f32_to_f16")?
            .ok_or_else(|| anyhow!("f32_to_f16 transform missing"))?
            .transform(&mut model)?;
    }
    Ok(model)
}

/// RVM load. 5 inputs: `src` `[1, 3, H, W]` + 4 recurrent states each
/// `[1, 1, 1, 1]` (per `_baked.onnx` convention; recurrent layer initialises
/// fresh state for the all-ones shape).
fn load_rvm(h: u32, w: u32, cast_to_f16: bool) -> Result<TypedModel> {
    let p = resolve_path(RVM_PATH);
    if !p.exists() {
        return Err(anyhow!("RVM ONNX not found at {}", p.display()));
    }
    let mut inf = tract_onnx::onnx().model_for_path(&p)?;
    inf.set_input_fact(
        0,
        InferenceFact::dt_shape(
            f32::datum_type(),
            tvec![1i64, 3, h as i64, w as i64],
        ),
    )?;
    for slot in 1..=4 {
        inf.set_input_fact(
            slot,
            InferenceFact::dt_shape(f32::datum_type(), tvec![1i64, 1, 1, 1]),
        )?;
    }
    inf.analyse(true)?;
    let model = inf.into_typed()?.into_decluttered()?;
    let mut model = model;
    if cast_to_f16 {
        get_transform("f32_to_f16")?
            .ok_or_else(|| anyhow!("f32_to_f16 transform missing"))?
            .transform(&mut model)?;
    }
    Ok(model)
}

fn build_runnable(
    model_kind: u32,
    h: u32,
    w: u32,
    backend: u32,
    compute_units: u32,
) -> Result<ModelHandle> {
    let kind = match model_kind {
        0 => ModelKind::Modnet,
        1 => ModelKind::Rvm,
        other => return Err(anyhow!("invalid model_kind: {other}")),
    };

    // Only TRACT Metal hard-requires F16. CoreML can run F32 (ANE won't engage
    // but CPU/GPU paths work). CPU stays F32 (MODNet/RVM `Resize` is F32-only).
    let is_f16 = backend == 1;
    let mut model = match kind {
        ModelKind::Modnet => load_modnet(h, w, is_f16)?,
        ModelKind::Rvm => load_rvm(h, w, is_f16)?,
    };

    match backend {
        0 => { /* TractCpu */ }
        1 => MetalTransform::default().transform(&mut model)?,
        2 => {
            let cu = match compute_units {
                0 => MLComputeUnits::CPUOnly,
                1 => MLComputeUnits::CPUAndGPU,
                2 => MLComputeUnits::CPUAndNeuralEngine,
                3 => MLComputeUnits::All,
                _ => return Err(anyhow!("invalid coreml_compute_units: {compute_units}")),
            };
            CoremlTransform { compute_units: cu }.transform(&mut model)?;
        }
        _ => return Err(anyhow!("invalid backend: {backend}")),
    }

    let runnable = model.into_runnable()?;
    Ok(ModelHandle {
        runnable,
        h: h as usize,
        w: w as usize,
        is_f16,
        kind,
    })
}

fn run_frame_inner(handle: &mut ModelHandle, src: &[f16], dst: &mut [f16]) -> Result<f64> {
    match handle.kind {
        ModelKind::Modnet => run_modnet(handle, src, dst),
        ModelKind::Rvm => run_rvm_stateless(handle, src, dst),
    }
}

fn run_modnet(handle: &mut ModelHandle, src: &[f16], dst: &mut [f16]) -> Result<f64> {
    let shape = [1, 3, handle.h, handle.w];
    if handle.is_f16 {
        let input = Tensor::from_shape::<f16>(&shape, src)?;
        let start = Instant::now();
        let outputs = handle.runnable.run(tvec![input.into()])?;
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        copy_alpha::<f16>(&outputs[0], dst)?;
        Ok(ms)
    } else {
        let src_f32: Vec<f32> = src.iter().map(|&v| f32::from(v)).collect();
        let input = Tensor::from_shape::<f32>(&shape, &src_f32)?;
        let start = Instant::now();
        let outputs = handle.runnable.run(tvec![input.into()])?;
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        copy_alpha::<f32>(&outputs[0], dst)?;
        Ok(ms)
    }
}

/// M4 RVM: stateless. Recurrent inputs are `[1, 1, 1, 1]` zeros every frame —
/// no temporal continuity. M5 will replace this with cycled-state inference
/// (which requires removing the `set_input_fact` shape lock on slots 1..=4).
fn run_rvm_stateless(
    handle: &mut ModelHandle,
    src: &[f16],
    dst: &mut [f16],
) -> Result<f64> {
    let shape = [1, 3, handle.h, handle.w];
    if handle.is_f16 {
        let input = Tensor::from_shape::<f16>(&shape, src)?;
        let r_init = Tensor::from_shape::<f16>(&[1, 1, 1, 1], &[f16::ZERO])?;
        let start = Instant::now();
        let outputs = handle.runnable.run(tvec![
            input.into(),
            r_init.clone().into(),
            r_init.clone().into(),
            r_init.clone().into(),
            r_init.into(),
        ])?;
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        // outputs[0] = fgr (ignored), outputs[1] = pha, outputs[2..6] = r1o..r4o
        copy_alpha::<f16>(&outputs[1], dst)?;
        Ok(ms)
    } else {
        let src_f32: Vec<f32> = src.iter().map(|&v| f32::from(v)).collect();
        let input = Tensor::from_shape::<f32>(&shape, &src_f32)?;
        let r_init = Tensor::from_shape::<f32>(&[1, 1, 1, 1], &[0.0f32])?;
        let start = Instant::now();
        let outputs = handle.runnable.run(tvec![
            input.into(),
            r_init.clone().into(),
            r_init.clone().into(),
            r_init.clone().into(),
            r_init.into(),
        ])?;
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        copy_alpha::<f32>(&outputs[1], dst)?;
        Ok(ms)
    }
}

/// Generic alpha-copy: model output `T` (f16 or f32) → caller's f16 buffer.
/// Goes through f32 to avoid needing a `From<f32> for f16` impl (`half`'s
/// API exposes the conversion through `from_f32` only).
fn copy_alpha<T>(value: &TValue, dst: &mut [f16]) -> Result<()>
where
    T: Datum + Copy + Into<f32>,
{
    let src = unsafe { value.as_slice_unchecked::<T>() };
    if src.len() != dst.len() {
        return Err(anyhow!(
            "alpha output length {} != expected {}",
            src.len(),
            dst.len()
        ));
    }
    for (out, &v) in dst.iter_mut().zip(src.iter()) {
        *out = f16::from_f32(v.into());
    }
    Ok(())
}
