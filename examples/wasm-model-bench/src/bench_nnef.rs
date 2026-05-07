//! Bench an NNEF model. Loads via tract-nnef, runs N timed reps, reports
//! min/median/max/spread.

use anyhow::Result;
use tract_nnef::prelude::*;

fn main() -> Result<()> {
    let mut args = std::env::args();
    let _prog = args.next();
    let model_path = args.next().ok_or_else(|| {
        anyhow::anyhow!("usage: bench-nnef <model.nnef.tgz> [warmup] [timed] [reps]")
    })?;
    let warmup_iters: usize = args.next().map(|s| s.parse()).transpose()?.unwrap_or(20);
    let timed_iters: usize = args.next().map(|s| s.parse()).transpose()?.unwrap_or(50);
    let repetitions: usize = args.next().map(|s| s.parse()).transpose()?.unwrap_or(10);

    let model =
        tract_nnef::nnef().model_for_path(&model_path)?.into_optimized()?.into_runnable()?;

    let inputs = wasm_model_bench::build_zero_inputs(&model)?;

    let samples =
        wasm_model_bench::run_bench(&model, &inputs, warmup_iters, timed_iters, repetitions)?;
    wasm_model_bench::print_stats(&model_path, &samples);
    if std::env::var("TRACT_BENCH_QUALITY").ok().as_deref() == Some("1") {
        wasm_model_bench::run_quality_check(&model, &inputs)?;
    }
    Ok(())
}
