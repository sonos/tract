use crate::CliResult;
use crate::{terminal, BenchLimits, Parameters};
use readings_probe::Probe;
use std::time::{Duration, Instant};
use tract_hir::internal::*;

pub fn criterion(params: &Parameters, matches: &clap::ArgMatches) -> CliResult<()> {
    let model =
        params.tract_model.downcast_ref::<TypedModel>().context("Can only bench TypedModel")?;
    let plan = SimplePlan::new(model)?;
    let mut state = SimpleState::new(plan)?;

    let mut crit = criterion::Criterion::default();
    let mut group = crit.benchmark_group("net");
    let allow_random_input = matches.is_present("allow-random-input");
    let allow_f32_to_f16 = matches.is_present("allow-f32-to-f16");
    let inputs = crate::tensor::retrieve_or_make_inputs(
        model,
        params,
        allow_random_input,
        allow_f32_to_f16,
    )?
    .remove(0);
    group.bench_function("run", move |b| b.iter(|| state.run(inputs.clone())));
    Ok(())
}

pub fn handle(
    params: &Parameters,
    matches: &clap::ArgMatches,
    limits: &BenchLimits,
    probe: Option<&Probe>,
) -> CliResult<()> {
    let model =
        params.tract_model.downcast_ref::<TypedModel>().context("Can only bench TypedModel")?;
    let plan = SimplePlan::new(model)?;
    let mut state = SimpleState::new(plan)?;

    let progress = probe.and_then(|m| m.get_i64("progress"));
    info!("Starting bench itself");
    let mut iters = 0;
    let allow_random_input = matches.is_present("allow-random-input");
    let allow_f32_to_f16 = matches.is_present("allow-f32-to-f16");
    let inputs = crate::tensor::retrieve_or_make_inputs(
        model,
        params,
        allow_random_input,
        allow_f32_to_f16,
    )?
    .remove(0);
    let start = Instant::now();
    while iters < limits.max_iters && start.elapsed() < limits.max_time {
        if let Some(mon) = probe {
            let _ = mon.log_event(&format!("loop_{}", iters));
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
