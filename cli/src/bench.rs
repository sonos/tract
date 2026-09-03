use crate::Parameters;
use std::time::Duration;
use tract_hir::internal::*;
#[cfg(not(target_family = "wasm"))]
use tract_libcli::profile::run_one_step;
use tract_libcli::profile::{BenchLimits, BenchResult};
#[cfg(not(target_family = "wasm"))]
use tract_libcli::profile::{PacedBench, Pacing};
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
    #[allow(unused_mut)]
    let mut run_params = crate::tensor::run_params_from_subcommand(params, sub_matches)?;
    // A stream feeds one row per turn, whatever the lanes the worker holds, so
    // the batch symbol is what an input of one seat carries.
    #[cfg(not(target_family = "wasm"))]
    if let Some(laned) = params.req_runnable()?.downcast_ref::<tract_core::lanes::LanedRunnable>() {
        run_params.symbols.set(laned.batch_symbol(), 1);
    }
    let inputs = get_or_make_inputs(&params.tract_model, &run_params)?;

    limits.warmup(&params.req_runnable()?, &inputs)?;

    let streams: usize =
        sub_matches.get_one::<String>("streams").map(|s| s.parse()).transpose()?.unwrap_or(1);
    #[cfg(not(target_family = "wasm"))]
    if let Some(pacing) = pacing(sub_matches)? {
        return bench_paced(params, &inputs, limits, &pacing, sub_matches);
    }
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

/// The real-time load the flags ask for, or `None` when the bench saturates
/// instead.
#[cfg(not(target_family = "wasm"))]
fn pacing(sub_matches: &clap::ArgMatches) -> TractResult<Option<Pacing>> {
    let millis = |name: &str| -> TractResult<Option<Duration>> {
        sub_matches
            .get_one::<String>(name)
            .map(|ms| -> TractResult<Duration> {
                let parsed: f64 = ms
                    .parse()
                    .with_context(|| format!("--{name} expects milliseconds, got {ms}"))?;
                Ok(Duration::from_secs_f64(parsed / 1e3))
            })
            .transpose()
    };
    let Some(period) = millis("turn-period")? else {
        ensure!(
            !sub_matches.get_flag("capacity"),
            "--capacity searches a real-time load: it wants --turn-period"
        );
        ensure!(
            millis("deadline")?.is_none(),
            "--deadline times a turn against the wall clock: it wants --turn-period"
        );
        return Ok(None);
    };
    Ok(Some(Pacing { period, deadline: millis("deadline")?.unwrap_or(period) }))
}

/// A real-time load: every stream owes a turn per `--turn-period`, and a turn
/// completing more than `--deadline` after its input arrived is late. Either
/// the fixed load `--streams` offers, or -- under `--capacity` -- the largest
/// load which holds, doubling until the deadline breaks and bisecting back.
///
/// The ceiling is `--streams`, or the lanes the runtime was given when it is
/// laned: a stream holds a lane for its whole session, so no search can go past
/// them.
#[cfg(not(target_family = "wasm"))]
fn bench_paced(
    params: &Parameters,
    inputs: &tract_libcli::tensor::RunTensors,
    limits: &BenchLimits,
    pacing: &Pacing,
    sub_matches: &clap::ArgMatches,
) -> TractResult<BenchResult> {
    let runnable = params.req_runnable()?;
    let laned = runnable.downcast_ref::<tract_core::lanes::LanedRunnable>();
    let asked: Option<usize> =
        sub_matches.get_one::<String>("streams").map(|s| s.parse()).transpose()?;
    let ceiling = match (asked, laned.map(|laned| laned.max_lanes())) {
        (Some(asked), Some(lanes)) => {
            ensure!(
                asked <= lanes,
                "{asked} streams over {lanes} lanes: a stream holds a lane for its whole session"
            );
            asked
        }
        (Some(asked), None) => asked,
        (None, Some(lanes)) => lanes,
        (None, None) => 1,
    };
    let quantile: f64 = sub_matches
        .get_one::<String>("deadline-quantile")
        .map(|q| q.parse())
        .transpose()?
        .unwrap_or(0.99);
    ensure!(
        (0. ..=1.).contains(&quantile),
        "--deadline-quantile reads the deadline at a q in 0..1, got {quantile}"
    );

    let trial = |streams: usize| -> TractResult<(PacedBench, Option<f64>)> {
        let before = laned.map(|laned| laned.turns_and_seats());
        let bench = limits.bench_streams_paced(&runnable, inputs, streams, pacing)?;
        let occupancy = laned.zip(before).map(|(laned, (turns, seats))| {
            let (worked, filled) = laned.turns_and_seats();
            (filled - seats) as f64 / (worked - turns).max(1) as f64
        });
        eprintln!(
            "{streams:4} streams: {:7.1} ms added at q{quantile}, {:5.1}% late{}",
            bench.added_quantile(quantile).as_secs_f64() * 1e3,
            bench.late_fraction() * 100.,
            occupancy.map(|occupancy| format!(", occupancy {occupancy:.2}")).unwrap_or_default(),
        );
        Ok((bench, occupancy))
    };

    if !sub_matches.get_flag("capacity") {
        let (bench, occupancy) = trial(ceiling)?;
        let metrics = paced_metrics(&bench, occupancy, None);
        return Ok(BenchResult { iters: bench.turns(), metrics });
    }

    let mut held: Option<(PacedBench, Option<f64>)> = None;
    let mut broken: Option<(PacedBench, Option<f64>)> = None;
    let mut floor = 0;
    let mut broke: Option<usize> = None;
    let mut streams = 1;
    while broke.is_none() {
        let (bench, occupancy) = trial(streams)?;
        if bench.holds(quantile) {
            floor = streams;
            held = Some((bench, occupancy));
            if streams == ceiling {
                break;
            }
            streams = (streams * 2).min(ceiling);
        } else {
            broke = Some(streams);
            broken = Some((bench, occupancy));
        }
    }
    while let Some(bad) = broke {
        if bad <= floor + 1 {
            break;
        }
        let streams = floor + (bad - floor) / 2;
        let (bench, occupancy) = trial(streams)?;
        if bench.holds(quantile) {
            floor = streams;
            held = Some((bench, occupancy));
        } else {
            broke = Some(streams);
            broken = Some((bench, occupancy));
        }
    }
    // The load which held, or -- when not even one stream did -- the one which
    // broke, so that the metrics say by how much.
    let (bench, occupancy) = held.or(broken).context("A capacity search ran no trial")?;
    let metrics = paced_metrics(&bench, occupancy, Some(floor));
    Ok(BenchResult { iters: bench.turns(), metrics })
}

/// What a paced trial answers with: the mean turn, what the streams got served,
/// how late the turns ran, and -- for a search -- the load which held.
#[cfg(not(target_family = "wasm"))]
fn paced_metrics(
    bench: &PacedBench,
    occupancy: Option<f64>,
    capacity: Option<usize>,
) -> Vec<(String, f64)> {
    let mut metrics = vec![
        ("evaltime".to_string(), bench.mean_service().as_secs_f64()),
        ("streams".to_string(), bench.streams as f64),
        ("turns_per_s".to_string(), bench.turns_per_second()),
        ("added_p50".to_string(), bench.added_quantile(0.5).as_secs_f64()),
        ("added_p95".to_string(), bench.added_quantile(0.95).as_secs_f64()),
        ("added_p99".to_string(), bench.added_quantile(0.99).as_secs_f64()),
        ("late_fraction".to_string(), bench.late_fraction()),
    ];
    metrics.extend(occupancy.map(|occupancy| ("occupancy".to_string(), occupancy)));
    metrics.extend(capacity.map(|capacity| ("capacity".to_string(), capacity as f64)));
    metrics
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
        for (name, q) in [("p50", "added_p50"), ("p95", "added_p95"), ("p99", "added_p99")] {
            if let Some(added) = metric(q) {
                println!("{name} added latency {:.1} ms.", added * 1e3);
            }
        }
        if let Some(late) = metric("late_fraction") {
            println!("{:.1}% of the turns completed past the deadline.", late * 100.);
        }
        if let Some(occupancy) = metric("occupancy") {
            println!("Mean occupancy {occupancy:.2}.");
        }
    }
    if let Some(capacity) = metric("capacity") {
        println!("Capacity {capacity:.0} streams in real time.");
    }

    for (k, v) in &result.metrics {
        if let Some(pp) = k.strip_prefix("pp") {
            println!("PP{pp}: {v:.1} tokens/sec");
        }
    }

    Ok(())
}
