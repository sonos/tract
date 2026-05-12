//! Phase 4 canary #6: DeepFilterNet 3 (DFN3) decoder NNEF through `CoremlRuntime`.
//!
//! DFN3 is an audio noise-suppression model (the project's primary
//! deployment target — see `~/coding/dfn3-wasm-relaxed-simd/handout.md`).
//! This canary loads the **decoder** sub-model at chunk size T=100. The
//! file lives at `~/coding/dfn3-wasm-relaxed-simd/models/dfn3/df_dec_t100.nnef.tgz`
//! (a tract-NNEF archive, not ONNX).
//!
//! Decoder I/O (per `graph.nnef`):
//!
//!   inputs:
//!     emb : [1, 100, 512]      embedding
//!     c0  : [1, 64, 100, 96]   per-band convolution state
//!   output:
//!     coefs : filter coefficients
//!
//! Architecture:
//!   - Initial Linear (matmul → relu)
//!   - 3× Q/K/V-style projections feeding GRU input
//!   - 2× GRU layers via `tract_core_scan` (the recurrent body holds the
//!     per-timestep GRU cell math: matmul + sigmoid + tanh + add + mul)
//!   - Decoder convs producing the filter coefficients
//!
//! The interesting tract-coreml challenge: `tract_core_scan` itself isn't
//! translatable to MIL — there's no MIL "scan" op (Core ML uses recurrent
//! ops like `lstm` / `gru` directly, not generic scan). Two strategies:
//!   (a) Dispatch only the OUTSIDE-scan compute (initial Linear + 3 input
//!       projections + decoder convs). The scan stays on CPU. This is what
//!       `CoremlTransform` does today since `tract_core_scan` is unknown.
//!   (b) Future: recursively transform the scan body, emit a per-timestep
//!       MLPackage call inside the scan. Requires architectural extension.
//!
//! For this canary we measure (a) — the partial dispatch — and characterise
//! how much of DFN3 is reachable without scan-body support.
//!
//! ```sh
//! cargo test -p tract-coreml --test dfn3_dec_e2e -- --ignored --nocapture
//! ```

use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use half::f16;
use objc2_core_ml::MLComputeUnits;

use tract_core::internal::*;
use tract_core::transform::{ModelTransform, get_transform};
use tract_nnef::prelude::*;

use tract_coreml::{CoremlOp, CoremlTransform};

const MODEL_PATH: &str = "~/coding/dfn3-wasm-relaxed-simd/models/dfn3/df_dec_t100.nnef.tgz";
// Per the NNEF graph: emb is [1, 100, 512], c0 is [1, 64, 100, 96].
const EMB_SHAPE: [usize; 3] = [1, 100, 512];
const C0_SHAPE: [usize; 4] = [1, 64, 100, 96];

fn resolve_path(path: &str) -> PathBuf {
    if let Some(stripped) = path.strip_prefix("~/") {
        let home = std::env::var("HOME").expect("HOME unset");
        PathBuf::from(home).join(stripped)
    } else {
        PathBuf::from(path)
    }
}

fn load_dfn3_f16() -> Result<TypedModel> {
    let path = resolve_path(MODEL_PATH);
    let nnef = tract_nnef::nnef();
    let model = nnef.model_for_path(&path)?;
    println!("loaded {}: {} nodes", path.display(), model.nodes.len());

    let model = model.into_decluttered()?;
    println!("after into_decluttered: {} nodes", model.nodes.len());
    let mut model = model;
    let f16x = get_transform("f32_to_f16")?
        .ok_or_else(|| anyhow::anyhow!("f32_to_f16 transform not registered"))?;
    f16x.transform(&mut model)?;
    println!("after f32_to_f16: {} nodes", model.nodes.len());
    Ok(model)
}

#[test]
#[ignore = "loads external df_dec_t100.nnef.tgz; --ignored --nocapture"]
fn dfn3_coverage_probe() -> Result<()> {
    let cpu_model = load_dfn3_f16()?;

    // Pre-transform op histogram.
    let mut op_hist: std::collections::BTreeMap<String, usize> = Default::default();
    for n in &cpu_model.nodes {
        *op_hist.entry(n.op.name().to_string()).or_default() += 1;
    }
    println!("\n[op-type histogram, post-declutter+f16, {} nodes]", cpu_model.nodes.len());
    for (op, c) in &op_hist {
        println!("  {c:4}× {op}");
    }

    // Apply CoremlTransform.
    let mut coreml_model = cpu_model.clone();
    let t0 = Instant::now();
    match CoremlTransform::default().transform(&mut coreml_model) {
        Ok(()) => println!("\nCoremlTransform: {:?}", t0.elapsed()),
        Err(e) => {
            println!("\nCoremlTransform FAILED: {e}");
            return Ok(());
        }
    }

    let coreml_count =
        coreml_model.nodes.iter().filter(|n| n.op_as::<CoremlOp>().is_some()).count();
    println!("=== after CoremlTransform ===");
    println!("  total nodes: {}", coreml_model.nodes.len());
    println!("  CoremlOps:   {coreml_count}");

    let mut after_hist: std::collections::BTreeMap<String, usize> = Default::default();
    for n in &coreml_model.nodes {
        *after_hist.entry(n.op.name().to_string()).or_default() += 1;
    }
    println!("\n[post-transform op histogram (what's still on CPU)]");
    for (op, c) in &after_hist {
        println!("  {c:4}× {op}");
    }

    // Diagnostic: pre-transform topo-ordered node list (op + name) so we can
    // see where the fragmentation boundaries actually fall after transform.
    println!("\n[pre-transform topo-ordered node list — annotated by op type]");
    for n in &cpu_model.nodes {
        let in_facts: Vec<String> = n
            .inputs
            .iter()
            .map(|i| {
                let f = cpu_model.outlet_fact(*i).unwrap();
                format!("{:?}@{:?}", f.shape, f.datum_type)
            })
            .collect();
        let out_facts: Vec<String> = n
            .outputs
            .iter()
            .map(|o| format!("{:?}@{:?}", o.fact.shape, o.fact.datum_type))
            .collect();
        println!(
            "  #{:>3} {:<22} {} in=[{}] out=[{}]",
            n.id,
            n.op.name(),
            n.name,
            in_facts.join("; "),
            out_facts.join("; ")
        );
    }

    // Diagnostic: post-transform — list every node, mark CoremlOp nodes with
    // their input/output outlets so we can see which tract regions got fused
    // into each CoremlOp.
    println!("\n[post-transform topo-ordered node list]");
    for n in &coreml_model.nodes {
        let kind = n.op.name();
        let inputs_str: Vec<String> =
            n.inputs.iter().map(|i| format!("{}/{}", i.node, i.slot)).collect();
        let mark = if n.op_as::<CoremlOp>().is_some() { " ← CoremlOp" } else { "" };
        println!(
            "  #{:>3} {:<22} {} inputs=[{}]{}",
            n.id,
            kind,
            n.name,
            inputs_str.join(", "),
            mark
        );
    }

    // Try to build runnable + run one chunk.
    let coreml_runnable = match coreml_model.into_runnable() {
        Ok(r) => r,
        Err(e) => {
            println!("\n[NOTE] coreml_model.into_runnable() failed: {e}");
            return Ok(());
        }
    };

    let emb_n: usize = EMB_SHAPE.iter().product();
    let emb_data: Vec<f16> =
        (0..emb_n).map(|i| f16::from_f32(((i as f32) * 0.001).sin() * 0.5)).collect();
    let emb = Tensor::from_shape::<f16>(&EMB_SHAPE, &emb_data)?;
    let c0_n: usize = C0_SHAPE.iter().product();
    let c0_data: Vec<f16> =
        (0..c0_n).map(|i| f16::from_f32(((i as f32) * 0.0005).cos() * 0.3)).collect();
    let c0 = Tensor::from_shape::<f16>(&C0_SHAPE, &c0_data)?;

    let t0 = Instant::now();
    let out = coreml_runnable.run(tvec![emb.into(), c0.into()])?;
    println!("\n[dfn3] CoreML first run: {:?}", t0.elapsed());
    for (i, t) in out.iter().enumerate() {
        println!("  out[{i}]: shape {:?}", t.shape());
    }
    Ok(())
}

/// Per-chunk perf for DFN3 decoder. Each chunk is T=100 audio frames at
/// (per the project's audio config) ~10ms each = 1s of audio.
/// RTF = inference_time / 1s. RTF < 1 means real-time; lower is better.
#[test]
#[ignore = "loads external df_dec_t100; long run; --ignored --nocapture"]
fn dfn3_perf_coreml_only() -> Result<()> {
    const ITERATIONS: usize = 100;
    const WARMUP: usize = 5;
    // Audio chunk duration: T=100 frames × 10ms/frame = 1.0s.
    const CHUNK_SECS: f64 = 1.0;

    let mut model = load_dfn3_f16()?;
    CoremlTransform::default().transform(&mut model)?;
    let coreml_runnable = model.into_runnable()?;

    let emb_n: usize = EMB_SHAPE.iter().product();
    let emb_data: Vec<f16> =
        (0..emb_n).map(|i| f16::from_f32(((i as f32) * 0.001).sin() * 0.5)).collect();
    let emb = Tensor::from_shape::<f16>(&EMB_SHAPE, &emb_data)?;
    let c0_n: usize = C0_SHAPE.iter().product();
    let c0_data: Vec<f16> =
        (0..c0_n).map(|i| f16::from_f32(((i as f32) * 0.0005).cos() * 0.3)).collect();
    let c0 = Tensor::from_shape::<f16>(&C0_SHAPE, &c0_data)?;

    for _ in 0..WARMUP {
        let _ = coreml_runnable.run(tvec![emb.clone().into(), c0.clone().into()])?;
    }

    let t0 = Instant::now();
    for _ in 0..ITERATIONS {
        let _ = coreml_runnable.run(tvec![emb.clone().into(), c0.clone().into()])?;
    }
    let coreml_per = t0.elapsed() / ITERATIONS as u32;
    let coreml_ms = coreml_per.as_secs_f64() * 1000.0;
    let rtf = coreml_ms / (CHUNK_SECS * 1000.0);

    println!(
        "\n=== DFN3 decoder perf (n={ITERATIONS}, warmup={WARMUP}, chunk_secs={CHUNK_SECS}) ===",
    );
    println!("  CoreML mean: {coreml_per:?}");
    println!("  RTF:         {rtf:.4} (<1 = real-time)");
    Ok(())
}

/// CPU-vs-CoreML perf comparison on the F16 path.
///
/// DFN3 has no Resize (NNEF-imported GRU + Conv model), so the F16 CPU path
/// was already runnable pre-Resize-F16-patch. Reports tract-CPU mean, CoreML
/// mean, speedup, and RTF for a 1 s audio chunk (T=100 frames × 10ms).
#[test]
#[ignore = "loads external df_dec_t100; long run; --ignored --nocapture"]
fn dfn3_perf_cpu_vs_coreml() -> Result<()> {
    const ITERATIONS_COREML: usize = 100;
    const WARMUP_COREML: usize = 5;
    // CPU is much slower; fewer iterations.
    const ITERATIONS_CPU: usize = 10;
    const WARMUP_CPU: usize = 2;
    const CHUNK_SECS: f64 = 1.0;

    let cpu_model = load_dfn3_f16()?;
    let mut coreml_model = load_dfn3_f16()?;
    CoremlTransform::default().transform(&mut coreml_model)?;
    let cpu_runnable = cpu_model.into_runnable()?;
    let coreml_runnable = coreml_model.into_runnable()?;

    let emb_n: usize = EMB_SHAPE.iter().product();
    let emb_data: Vec<f16> =
        (0..emb_n).map(|i| f16::from_f32(((i as f32) * 0.001).sin() * 0.5)).collect();
    let emb = Tensor::from_shape::<f16>(&EMB_SHAPE, &emb_data)?;
    let c0_n: usize = C0_SHAPE.iter().product();
    let c0_data: Vec<f16> =
        (0..c0_n).map(|i| f16::from_f32(((i as f32) * 0.0005).cos() * 0.3)).collect();
    let c0 = Tensor::from_shape::<f16>(&C0_SHAPE, &c0_data)?;
    let inputs = || -> TVec<TValue> { tvec![emb.clone().into(), c0.clone().into()] };

    println!("\n[dfn3] warming CPU path (F16)");
    for _ in 0..WARMUP_CPU {
        let _ = cpu_runnable.run(inputs())?;
    }
    let t0 = Instant::now();
    for _ in 0..ITERATIONS_CPU {
        let _ = cpu_runnable.run(inputs())?;
    }
    let cpu_per = t0.elapsed() / ITERATIONS_CPU as u32;
    let cpu_ms = cpu_per.as_secs_f64() * 1000.0;

    println!("[dfn3] warming CoreML path");
    for _ in 0..WARMUP_COREML {
        let _ = coreml_runnable.run(inputs())?;
    }
    let t0 = Instant::now();
    for _ in 0..ITERATIONS_COREML {
        let _ = coreml_runnable.run(inputs())?;
    }
    let coreml_per = t0.elapsed() / ITERATIONS_COREML as u32;
    let coreml_ms = coreml_per.as_secs_f64() * 1000.0;

    let speedup = cpu_ms / coreml_ms;
    let cpu_rtf = cpu_ms / (CHUNK_SECS * 1000.0);
    let coreml_rtf = coreml_ms / (CHUNK_SECS * 1000.0);

    println!("\n=== DFN3 decoder CPU-vs-CoreML perf (chunk_secs={CHUNK_SECS}, F16) ===");
    println!("  Tract CPU mean (n={ITERATIONS_CPU}): {cpu_per:?}  = {cpu_ms:.2} ms");
    println!("  CoreML mean    (n={ITERATIONS_COREML}): {coreml_per:?}  = {coreml_ms:.2} ms");
    println!("  Speedup:        {speedup:.1}×");
    println!("  RTF (CPU):      {cpu_rtf:.4} (<1 = real-time)");
    println!("  RTF (CoreML):   {coreml_rtf:.4} (<1 = real-time)");

    Ok(())
}

/// Sustained loop for `powermetrics` ANE measurement.
#[test]
#[ignore = "loads external dfn3; long sustained run; pair with sudo powermetrics"]
fn dfn3_sustained_for_powermetrics() -> Result<()> {
    let mut model = load_dfn3_f16()?;
    let xform = CoremlTransform { compute_units: MLComputeUnits::CPUAndNeuralEngine };
    xform.transform(&mut model)?;

    let coreml_count = model.nodes.iter().filter(|n| n.op_as::<CoremlOp>().is_some()).count();
    println!("Compiled with CPUAndNeuralEngine; {coreml_count} CoremlOp nodes.");

    let runnable = model.into_runnable()?;

    let emb_n: usize = EMB_SHAPE.iter().product();
    let emb_data: Vec<f16> =
        (0..emb_n).map(|i| f16::from_f32(((i as f32) * 0.001).sin() * 0.5)).collect();
    let emb = Tensor::from_shape::<f16>(&EMB_SHAPE, &emb_data)?;
    let c0_n: usize = C0_SHAPE.iter().product();
    let c0_data: Vec<f16> =
        (0..c0_n).map(|i| f16::from_f32(((i as f32) * 0.0005).cos() * 0.3)).collect();
    let c0 = Tensor::from_shape::<f16>(&C0_SHAPE, &c0_data)?;

    for _ in 0..3 {
        let _ = runnable.run(tvec![emb.clone().into(), c0.clone().into()])?;
    }

    let target_secs = 30u64;
    let start = Instant::now();
    let mut iters = 0usize;
    while start.elapsed().as_secs() < target_secs {
        let _ = runnable.run(tvec![emb.clone().into(), c0.clone().into()])?;
        iters += 1;
    }
    let elapsed = start.elapsed();
    println!("Ran {iters} iterations in {elapsed:?}  (mean {:?})", elapsed / iters as u32);
    Ok(())
}
