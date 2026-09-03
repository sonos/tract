use crate::Parameters;
use std::time::Duration;
use tract_hir::internal::*;
#[cfg(not(target_family = "wasm"))]
use tract_libcli::profile::run_one_step;
use tract_libcli::profile::{BenchLimits, BenchResult};
use tract_libcli::tensor::get_or_make_inputs;
use tract_libcli::terminal;

#[cfg(not(target_family = "wasm"))]
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

    let streams: usize =
        sub_matches.get_one::<String>("streams").map(|s| s.parse()).transpose()?.unwrap_or(1);
    #[cfg(not(target_family = "wasm"))]
    if streams > 1 {
        return bench_streams(params, &inputs, limits, streams);
    }

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

/// Several streams at once, each saturating a thread of its own. `evaltime` is
/// the mean turn as a stream sees it, which under saturation includes waiting
/// for the others, and `turns_per_s` is what the streams got served together --
/// the two numbers a laned runtime trades against each other.
#[cfg(not(target_family = "wasm"))]
fn bench_streams(
    params: &Parameters,
    inputs: &tract_libcli::tensor::RunTensors,
    limits: &BenchLimits,
    streams: usize,
) -> TractResult<BenchResult> {
    let runnable = params.req_runnable()?;
    let laned = runnable.downcast_ref::<tract_core::lanes::LanedRunnable>();
    if let Some(laned) = laned {
        ensure!(
            streams <= laned.max_lanes(),
            "{streams} streams over {} lanes: a stream holds a lane for its whole session",
            laned.max_lanes()
        );
    }
    let bench = limits.bench_streams(&runnable, inputs, streams)?;
    let mut metrics = vec![
        ("evaltime".to_string(), bench.mean_latency().as_secs_f64()),
        ("turns_per_s".to_string(), bench.turns_per_second()),
        ("latency_p50".to_string(), bench.latency_quantile(0.5).as_secs_f64()),
        ("latency_p95".to_string(), bench.latency_quantile(0.95).as_secs_f64()),
    ];
    if let Some(laned) = laned {
        let (turns, seats) = laned.turns_and_seats();
        metrics.push(("occupancy".to_string(), seats as f64 / turns as f64));
    }
    Ok(BenchResult { metrics, iters: bench.turns() })
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

    let metric = |name: &str| result.metrics.iter().find(|(k, _)| k == name).map(|(_, v)| *v);
    if let Some(turns_per_s) = metric("turns_per_s") {
        println!("{turns_per_s:.1} turns/sec over every stream together.");
        for (name, q) in [("p50", "latency_p50"), ("p95", "latency_p95")] {
            if let Some(latency) = metric(q) {
                println!("{name} turn {}.", terminal::dur_avg(Duration::from_secs_f64(latency)));
            }
        }
        if let Some(occupancy) = metric("occupancy") {
            println!("Mean occupancy {occupancy:.2}.");
        }
    }

    for (k, v) in &result.metrics {
        if let Some(pp) = k.strip_prefix("pp") {
            println!("PP{pp}: {v:.1} tokens/sec");
        }
    }

    Ok(())
}
