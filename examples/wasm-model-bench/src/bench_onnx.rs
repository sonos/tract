//! Bench an ONNX model.
//!
//! Usage: bench-onnx <model.onnx> [<shape_spec>|-] [warmup] [timed] [reps]
//!
//!   <shape_spec> can be a single per-input fact like "1,3,224,224,f32" or
//!   multi-input with ";" separators: "1,1,100,32,f32;1,2,100,96,f32".
//!   Use "-" or empty to skip override for that input.
//!
//! Optional env var TRACT_BENCH_SYMBOLS="S=100,BATCH=1" applies global
//! symbol concretization for models whose internal nodes still reference
//! symbols after input facts are set.

use anyhow::{Result, anyhow};
use tract_onnx::prelude::*;

fn parse_fact_spec(spec: &str) -> Result<InferenceFact> {
    let parts: Vec<&str> = spec.split(',').collect();
    if parts.len() < 2 {
        anyhow::bail!("shape spec needs at least 'shape,dtype': {spec}");
    }
    let dt_str = parts.last().unwrap();
    let dt = match *dt_str {
        "f32" => DatumType::F32,
        "f16" => DatumType::F16,
        "i64" => DatumType::I64,
        "i32" => DatumType::I32,
        "u8" => DatumType::U8,
        s => return Err(anyhow!("unsupported dtype: {s}")),
    };
    let shape: Vec<usize> = parts[..parts.len() - 1]
        .iter()
        .map(|s| s.parse::<usize>().map_err(anyhow::Error::from))
        .collect::<Result<_>>()?;
    Ok(InferenceFact::dt_shape(dt, shape))
}

fn main() -> Result<()> {
    let mut args = std::env::args();
    let _prog = args.next();
    let model_path = args.next().ok_or_else(|| {
        anyhow!("usage: bench-onnx <model.onnx> [<shape_spec>|-] [warmup] [timed] [reps]")
    })?;
    let shape_spec = args.next();
    let warmup_iters: usize = args.next().map(|s| s.parse()).transpose()?.unwrap_or(20);
    let timed_iters: usize = args.next().map(|s| s.parse()).transpose()?.unwrap_or(50);
    let repetitions: usize = args.next().map(|s| s.parse()).transpose()?.unwrap_or(10);

    let mut model = tract_onnx::onnx().model_for_path(&model_path)?;
    let symbols_env = std::env::var("TRACT_BENCH_SYMBOLS").ok().filter(|s| !s.is_empty());

    // When symbols are provided, the model is symbolic-shaped and we go straight
    // to TypedModel → set_symbols, ignoring shape_spec (input shapes will
    // be derived from the symbol substitution). When no symbols, we use the
    // shape_spec to pin input facts on the InferenceModel before into_typed.
    let typed = if let Some(symbols_str) = symbols_env {
        let typed = model.into_typed()?;
        let mut subs = std::collections::HashMap::new();
        for kv in symbols_str.split(',') {
            let (k, v) = kv.split_once('=').ok_or_else(|| anyhow!("bad symbol: {kv}"))?;
            let sym = typed.symbols.sym(k);
            subs.insert(sym, TDim::Val(v.parse::<i64>()?));
        }
        typed.set_symbols(&subs)?
    } else {
        if let Some(spec) = shape_spec.as_deref().filter(|s| *s != "-") {
            for (i, one) in spec.split(';').enumerate() {
                if one.is_empty() || one == "-" {
                    continue;
                }
                let fact = parse_fact_spec(one)?;
                model.set_input_fact(i, fact)?;
            }
        }
        model.into_typed()?
    };
    let model = typed.into_optimized()?.into_runnable()?;

    let inputs = wasm_model_bench::build_zero_inputs(&model)?;

    let samples =
        wasm_model_bench::run_bench(&model, &inputs, warmup_iters, timed_iters, repetitions)?;
    wasm_model_bench::print_stats(&model_path, &samples);
    if std::env::var("TRACT_BENCH_QUALITY").ok().as_deref() == Some("1") {
        wasm_model_bench::run_quality_check(&model, &inputs)?;
    }
    Ok(())
}
