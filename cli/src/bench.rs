use crate::Parameters;
use readings_probe::Probe;
use std::time::{Duration, Instant};
use tract_hir::internal::*;
use tract_libcli::profile::BenchLimits;
use tract_libcli::terminal;

pub fn criterion(
    params: &Parameters,
    _matches: &clap::ArgMatches,
    sub_matches: &clap::ArgMatches,
) -> TractResult<()> {
    let run_params = crate::tensor::run_params_from_subcommand(params, sub_matches)?;

    let model =
        params.tract_model.downcast_ref::<TypedModel>().context("Can only bench TypedModel")?;
    let plan = SimplePlan::new(model)?;
    let mut state = SimpleState::new(plan)?;

    let mut crit = criterion::Criterion::default();
    let mut group = crit.benchmark_group("net");
    let inputs = tract_libcli::tensor::retrieve_or_make_inputs(model, &run_params)?.remove(0);
    group.bench_function("run", move |b| b.iter(|| state.run(inputs.clone())));
    Ok(())
}

pub fn handle(
    params: &Parameters,
    _matches: &clap::ArgMatches,
    sub_matches: &clap::ArgMatches,
    limits: &BenchLimits,
    probe: Option<&Probe>,
) -> TractResult<()> {
    let run_params = crate::tensor::run_params_from_subcommand(params, sub_matches)?;
    let model =
        params.tract_model.downcast_ref::<TypedModel>().context("Can only bench TypedModel")?;
    let inputs = tract_libcli::tensor::retrieve_or_make_inputs(model, &run_params)?.remove(0);
    let plan = SimplePlan::new(model)?;

    limits.warmup(model, &inputs)?;

    let mut state = SimpleState::new(plan)?;
    let mut iters = 0;

    let progress = probe.and_then(|m| m.get_i64("progress"));
    info!("Starting bench itself");
    let start = Instant::now();
    while iters < limits.max_loops && start.elapsed() < limits.max_time {
        if let Some(mon) = probe {
            let _ = mon.log_event(&format!("loop_{iters}"));
        }
        if let Some(p) = &progress {
            p.store(iters as _, std::sync::atomic::Ordering::Relaxed);
        }
        state.run(inputs.clone())?;
        iters += 1;
    }
    let dur = start.elapsed();
    let dur = Duration::from_secs_f64(dur.as_secs_f64() / iters as f64);

    if params.machine_friendly {
        println!("real: {}", dur.as_secs_f64());
    } else {
        println!("Bench ran {} times, {}.", iters, terminal::dur_avg(dur));
    }

    Ok(())
}
