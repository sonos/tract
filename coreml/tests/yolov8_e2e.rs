//! Coverage probe for YOLOv8n (Ultralytics nano object detector) ONNX through
//! `CoremlTransform`.
//!
//! Standard Ultralytics YOLOv8n export: input `[1, 3, 640, 640]` RGB float32,
//! output `[1, 84, 8400]` (4 bbox + 80 COCO class scores per anchor, raw —
//! NMS / TopK live in the Python wrapper, not the graph). This probe is purely
//! diagnostic: surface the op histogram + stranded-op list + Conv eligibility,
//! and call out any detection-specific ops (NonMaxSuppression, TopK, anchor
//! decode) that would block growing the CoreML subgraph.
//!
//! `#[ignore]` by default; loads model from outside the workspace. Run with:
//!
//! ```sh
//! cargo test -p tract-coreml --test yolov8_e2e -- --ignored --nocapture
//! ```

use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;

use tract_core::internal::*;
use tract_core::ops::cnn::Conv;
use tract_core::transform::{ModelTransform, get_transform};
use tract_onnx::prelude::*;

use tract_coreml::{CoremlOp, CoremlTransform};

const MODEL_PATH: &str = "~/coding/dfn3-wasm-relaxed-simd/models/yolov8n.onnx";
const SRC_SHAPE: [usize; 4] = [1, 3, 640, 640];

fn resolve_path(path: &str) -> PathBuf {
    if let Some(stripped) = path.strip_prefix("~/") {
        let home = std::env::var("HOME").expect("HOME unset");
        PathBuf::from(home).join(stripped)
    } else {
        PathBuf::from(path)
    }
}

/// Load yolov8n.onnx, bind the input shape (Ultralytics exports a fixed
/// [1,3,640,640] but we still call `set_input_fact` defensively, matching the
/// SAM2 canary's load shape), declutter, then cast to F16.
fn load_yolov8n_f16() -> Result<TypedModel> {
    let path = resolve_path(MODEL_PATH);
    let mut inf = tract_onnx::onnx().model_for_path(&path)?;

    inf.set_input_fact(
        0,
        InferenceFact::dt_shape(
            f32::datum_type(),
            tvec![
                SRC_SHAPE[0] as i64,
                SRC_SHAPE[1] as i64,
                SRC_SHAPE[2] as i64,
                SRC_SHAPE[3] as i64
            ],
        ),
    )?;

    inf.analyse(true)?;
    let model = inf.into_typed()?;
    println!("loaded {}: {} nodes", path.display(), model.nodes.len());

    let model = model.into_decluttered()?;
    println!("after into_decluttered: {} nodes", model.nodes.len());

    let mut model = model;
    let f16_xform = get_transform("f32_to_f16")?
        .ok_or_else(|| anyhow::anyhow!("f32_to_f16 transform not registered"))?;
    f16_xform.transform(&mut model)?;
    println!("after f32_to_f16: {} nodes", model.nodes.len());

    Ok(model)
}

#[test]
#[ignore = "loads external yolov8n.onnx; run with --ignored --nocapture"]
fn yolov8n_coverage_probe() -> Result<()> {
    let cpu_model = load_yolov8n_f16()?;

    // Pre-transform op histogram: tells us what ops sit between Convs.
    let mut op_hist: std::collections::BTreeMap<String, usize> = Default::default();
    for n in &cpu_model.nodes {
        *op_hist.entry(n.op.name().to_string()).or_default() += 1;
    }
    println!("\n[op-type histogram, post-declutter+f16, {} nodes]", cpu_model.nodes.len());
    for (op, c) in &op_hist {
        println!("  {c:4}× {op}");
    }

    // Per-Conv eligibility breakdown.
    use tract_coreml::ops::conv::{ConvAnalysis, analyse_conv};
    let conv_node_ids: Vec<usize> =
        cpu_model.nodes.iter().filter(|n| n.op_as::<Conv>().is_some()).map(|n| n.id).collect();
    println!("\n[per-Conv eligibility] {} Conv nodes:", conv_node_ids.len());
    let mut translatable = 0;
    let mut skip_counts: std::collections::BTreeMap<String, usize> =
        std::collections::BTreeMap::new();
    for id in &conv_node_ids {
        let n = &cpu_model.nodes[*id];
        match analyse_conv(&cpu_model, n)? {
            ConvAnalysis::Translatable(_) => {
                translatable += 1;
            }
            ConvAnalysis::Skip(reason) => {
                let bucket = reason.split(' ').take(3).collect::<Vec<_>>().join(" ");
                *skip_counts.entry(bucket).or_default() += 1;
            }
        }
    }
    println!("  → {translatable}/{} translatable", conv_node_ids.len());
    if !skip_counts.is_empty() {
        println!("[Conv skip-reason histogram]");
        for (reason, count) in &skip_counts {
            println!("  {count:3}× {reason}…");
        }
    }

    // Dump every Conv's input fact + kernel shape so when CoremlTransform fails
    // we can correlate the "Conv-N" temp-package name in the error message to
    // a specific node and shape.
    use tract_core::ops::cnn::Conv as ConvOp;
    println!("\n[Conv-by-Conv input/kernel shapes]");
    for (i, id) in conv_node_ids.iter().enumerate() {
        let n = &cpu_model.nodes[*id];
        let conv = n.op_as::<ConvOp>().unwrap();
        let in_fact = cpu_model.outlet_fact(n.inputs[0])?;
        let out_fact = cpu_model.outlet_fact(OutletId::new(n.id, 0))?;
        println!(
            "  #{i:2} node#{id:3} {} in={:?} out={:?} k={:?} stride={:?} pad={:?} group={} data_format={:?}",
            n.name,
            in_fact.shape,
            out_fact.shape,
            conv.pool_spec.kernel_shape,
            conv.pool_spec.strides,
            conv.pool_spec.padding,
            conv.group,
            conv.pool_spec.data_format,
        );
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
    let convs_after = coreml_model.nodes.iter().filter(|n| n.op_as::<Conv>().is_some()).count();
    println!("=== after CoremlTransform ===");
    println!("  total nodes:     {}", coreml_model.nodes.len());
    println!("  Conv nodes left: {convs_after}  (CPU fallback)");
    println!("  CoremlOp nodes:  {coreml_count}");

    // Op-type histogram of remaining (= what's still on CPU = stranded ops).
    let mut after_hist: std::collections::BTreeMap<String, usize> = Default::default();
    for n in &coreml_model.nodes {
        *after_hist.entry(n.op.name().to_string()).or_default() += 1;
    }
    println!("\n[post-transform op histogram (what's still on CPU + CoremlOp count)]");
    for (op, count) in &after_hist {
        println!("  {count:4}× {op}");
    }

    // Detection-specific call-out: list any nodes whose op name suggests NMS,
    // TopK, anchor decode, or other postprocess machinery. Raw Ultralytics
    // export is *supposed* to skip NMS, but custom exports sometimes bake it
    // in — we want explicit confirmation either way.
    println!("\n[detection-specific op scan]");
    let detection_keywords =
        ["NonMaxSuppression", "TopK", "ScatterND", "ScatterElements", "GatherND"];
    let mut detection_hits: Vec<(String, usize)> = Vec::new();
    for n in &coreml_model.nodes {
        let name = n.op.name().to_string();
        for kw in &detection_keywords {
            if name.contains(kw) {
                detection_hits.push((name.clone(), n.id));
            }
        }
    }
    if detection_hits.is_empty() {
        println!("  (none — no NMS/TopK/Scatter*/GatherND in graph; Ultralytics-style raw export)");
    } else {
        for (op, id) in &detection_hits {
            println!("  node #{id}: {op}");
        }
    }

    // Dump the FIRST instance of each remaining (non-Coreml) op kind with a
    // skip reason that's specific enough to act on.
    println!("\n[first-instance details for stranded ops]");
    let mut seen: std::collections::BTreeSet<String> = Default::default();
    for n in &coreml_model.nodes {
        if n.op_as::<CoremlOp>().is_some() {
            continue;
        }
        // Source / Const / inputs aren't actionable.
        let op_name = n.op.name().to_string();
        if matches!(op_name.as_str(), "Source" | "Const" | "TypedSource") {
            continue;
        }
        if !seen.insert(op_name.clone()) {
            continue;
        }
        let in_shapes: Vec<String> = n
            .inputs
            .iter()
            .map(|i| {
                let f = coreml_model.outlet_fact(*i).unwrap();
                format!("{:?}@{:?}", f.shape, f.datum_type)
            })
            .collect();
        println!("  {} (#{}, name={})", op_name, n.id, n.name);
        println!("    inputs: [{}]", in_shapes.join("; "));
    }

    // Try to build runnable. If a leftover op isn't supported the runnable
    // will fail to build — useful coverage signal either way.
    let coreml_runnable = match coreml_model.into_runnable() {
        Ok(r) => r,
        Err(e) => {
            println!("\n[NOTE] coreml_model.into_runnable() failed: {e}");
            println!(
                "[NOTE] this typically means a CPU-fallback op needs work; \
                     skipping end-to-end run"
            );
            return Ok(());
        }
    };
    // Use a generic zero-init f16 input (doesn't matter for coverage probe).
    let n: usize = SRC_SHAPE.iter().product();
    let data: Vec<half::f16> =
        (0..n).map(|i| half::f16::from_f32(((i as f32) * 0.001).sin() * 0.5)).collect();
    let input = Tensor::from_shape::<half::f16>(&SRC_SHAPE, &data)?;
    let t0 = Instant::now();
    match coreml_runnable.run(tvec![input.into()]) {
        Ok(out) => {
            println!("\nCoreML first run: {:?}", t0.elapsed());
            println!("  outputs: {}", out.len());
            for (i, o) in out.iter().enumerate() {
                println!("  output #{i} shape: {:?} dtype={:?}", o.shape(), o.datum_type());
            }
        }
        Err(e) => {
            println!("\n[NOTE] runnable.run() failed: {e}");
        }
    }

    Ok(())
}
