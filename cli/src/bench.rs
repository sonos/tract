use crate::Parameters;
use std::time::Duration;
use tract_hir::internal::*;
use tract_libcli::profile::{BenchLimits, BenchResult, run_one_step};
use tract_libcli::tensor::get_or_make_inputs;
use tract_libcli::terminal;

pub fn criterion(
    params: &Parameters,
    _matches: &clap::ArgMatches,
    sub_matches: &clap::ArgMatches,
) -> TractResult<()> {
    let mut crit = criterion::Criterion::default();
    let mut group = crit.benchmark_group("net");

    let run_params = crate::tensor::run_params_from_subcommand(params, sub_matches)?;
    let inputs = get_or_make_inputs(&params.tract_model, &run_params)?;

    let runnable = params.req_runnable()?;
    let mut state = runnable.spawn()?;
    group.bench_function("run", move |b| b.iter(|| run_one_step(&runnable, &mut state, &inputs)));

    Ok(())
}

/// Measure the net and return its metrics, without printing. The `bench` subcommand
/// and the bench suite both go through this.
pub fn run(
    params: &Parameters,
    sub_matches: &clap::ArgMatches,
    limits: &BenchLimits,
) -> TractResult<BenchResult> {
    let run_params = crate::tensor::run_params_from_subcommand(params, sub_matches)?;
    let inputs = get_or_make_inputs(&params.tract_model, &run_params)?;

    limits.warmup(&params.req_runnable()?, &inputs)?;

    let (iters, dur) = {
        #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
        let _profiler =
            sub_matches.get_flag("cuda-gpu-trace").then(cudarc::driver::safe::Profiler::new);
        limits.bench(&params.req_runnable()?, &inputs)?
    };
    let evaltime = dur.div_f64(iters as _).as_secs_f64();

    let mut metrics = vec![("evaltime".to_string(), evaltime)];
    if let Some(pp) = sub_matches.get_one::<String>("pp") {
        let pp = pp.parse::<usize>()?;
        metrics.push((format!("pp{pp}"), pp as f64 / evaltime));
    }
    Ok(BenchResult { metrics, iters })
}

pub fn handle(
    params: &Parameters,
    sub_matches: &clap::ArgMatches,
    limits: &BenchLimits,
) -> TractResult<()> {
    let mut result = run(params, sub_matches, limits)?;
    if params.emit_jsonl {
        result.metrics.extend(tract_libcli::profile::stage_metrics_from_readings("readings.out"));
        result.emit_jsonl();
        return Ok(());
    }
    let evaltime = result.metrics.iter().find(|(k, _)| k == "evaltime").map_or(0.0, |(_, v)| *v);

    if params.machine_friendly {
        println!("real: {evaltime}");
    } else {
        println!(
            "Bench ran {} times, {}.",
            result.iters,
            terminal::dur_avg(Duration::from_secs_f64(evaltime))
        );
    }

    for (k, v) in &result.metrics {
        if let Some(pp) = k.strip_prefix("pp") {
            println!("PP{pp}: {v:.1} tokens/sec");
        }
    }

    Ok(())
}
