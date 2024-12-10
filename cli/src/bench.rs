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
    let plan_options = crate::plan_options::plan_options_from_subcommand(sub_matches)?;
    let run_params = crate::tensor::run_params_from_subcommand(params, sub_matches)?;

    let model =
        params.tract_model.downcast_ref::<TypedModel>().context("Can only bench TypedModel")?;
    let plan = SimplePlan::new_with_options(model, &plan_options)?;
    let mut state = SimpleState::new(plan)?;

    let mut crit = criterion::Criterion::default();
    let mut group = crit.benchmark_group("net");
    let inputs = tract_libcli::tensor::retrieve_or_make_inputs(model, &run_params)?.remove(0);
    group.bench_function("run", move |b| b.iter(|| state.run(inputs.clone())));
    Ok(())
}

pub fn handle(
    params: &Parameters,
    matches: &clap::ArgMatches,
    sub_matches: &clap::ArgMatches,
    limits: &BenchLimits,
    probe: Option<&Probe>,
) -> TractResult<()> {
    let plan_options = crate::plan_options::plan_options_from_subcommand(sub_matches)?;
    let run_params = crate::tensor::run_params_from_subcommand(params, sub_matches)?;
    let model =
        params.tract_model.downcast_ref::<TypedModel>().context("Can only bench TypedModel")?;
    let inputs = tract_libcli::tensor::retrieve_or_make_inputs(model, &run_params)?.remove(0);

    limits.warmup(model, &inputs)?;

    let mut state = {
        if matches.is_present("metal"){
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            {
                let mut plan = SimplePlan::new_with_options(model, &plan_options)?;
                let state = TypedSimpleState::new_from_inputs(&plan, inputs.clone())?;

                let session_handler = tract_metal::MetalSessionHandler::from_plan(
                    &plan,
                    &state.session_state.resolved_symbols,
                )?;

                plan = plan.with_session_handler(session_handler);
                SimpleState::new(Arc::new(plan))?
            }
            #[cfg(not(any(target_os = "macos", target_os = "ios")))]
            {
                bail!("Metal bench called on non-Metal model");
            }
        }
        else
        {
            let plan = SimplePlan::new_with_options(model, &plan_options)?;
            SimpleState::new(Arc::new(plan))?
        }
    };
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
