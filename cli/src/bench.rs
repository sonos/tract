use crate::Parameters;
use readings_probe::Probe;
use std::time::{Duration, Instant};
use tract_hir::internal::*;
use tract_libcli::capture_gpu_trace;
use tract_libcli::model::Model;
use tract_libcli::profile::BenchLimits;
use tract_libcli::tensor::get_or_make_inputs;
use tract_libcli::tensor::RunTensors;
use tract_libcli::terminal;

fn profile_single_turn<'m>(
    state: &mut TypedSimpleState<&'m TypedModel, Arc<TypedRunnableModel<&'m TypedModel>>>,
    inputs: &RunTensors,
) -> TractResult<Duration> {
    if state.model().properties().contains_key("pulse.delay") {
        let start = Instant::now();
        state.run(inputs.sources[0].clone())?;
        Ok(start.elapsed())
    } else {
        state.init_states(&mut inputs.state_initializers.clone())?;
        let start = Instant::now();
        state.run(inputs.sources[0].clone())?;
        let elapsed = start.elapsed();
        state.reset_op_states()?;
        Ok(elapsed)
    }
}

pub fn criterion(
    params: &Parameters,
    matches: &clap::ArgMatches,
    sub_matches: &clap::ArgMatches,
) -> TractResult<()> {
    let model =
        params.tract_model.downcast_ref::<TypedModel>().context("Can only bench TypedModel")?;
    let mut state = make_state(params, matches, sub_matches)?;

    let mut crit = criterion::Criterion::default();
    let mut group = crit.benchmark_group("net");

    let run_params = crate::tensor::run_params_from_subcommand(params, sub_matches)?;
    let inputs = get_or_make_inputs(model, &run_params)?;

    group.bench_function("run", move |b| b.iter(|| profile_single_turn(&mut state, &inputs)));
    Ok(())
}

pub(crate) fn make_state<'m>(
    params: &'m Parameters,
    matches: &clap::ArgMatches,
    sub_matches: &clap::ArgMatches,
) -> TractResult<TypedSimpleState<&'m TypedModel, Arc<TypedRunnableModel<&'m TypedModel>>>> {
    #[allow(unused_mut)]
    let mut plan_options = crate::plan_options::plan_options_from_subcommand(sub_matches)?;
    let model =
        params.tract_model.downcast_ref::<TypedModel>().context("Can only bench TypedModel")?;
    if matches.is_present("metal") || matches.is_present("cuda") {
        #[cfg(not(any(target_os = "macos", target_os = "ios")))]
        {
            use tract_cuda::utils::is_culib_present;
            if !is_culib_present() {
                bail!("GPU bench called on non-GPU model");
            }
        }
        plan_options.skip_order_opt_ram = true;
        let mut plan = SimplePlan::new_with_options(model, &plan_options)?;
        let mut symbol_values = SymbolValues::default();
        if let Some(s) = model.symbols.get("S") {
            symbol_values.set(&s, 1024);
        }
        if let Some(p) = model.symbols.get("P") {
            symbol_values.set(&p, 0);
        }

        let session_handler =
            tract_gpu::session_handler::DeviceSessionHandler::from_plan(&plan, &symbol_values)?;

        plan = plan.with_session_handler(session_handler);
        Ok(SimpleState::new(Arc::new(plan))?)
    } else {
        let plan = SimplePlan::new_with_options(model, &plan_options)?;
        Ok(SimpleState::new(Arc::new(plan))?)
    }
}

pub(crate) fn bench<'m>(
    state: &mut TypedSimpleState<&'m TypedModel, Arc<TypedRunnableModel<&'m TypedModel>>>,
    sub_matches: &clap::ArgMatches,
    inputs: RunTensors,
    limits: &BenchLimits,
    probe: Option<&Probe>,
) -> TractResult<(usize, Duration)> {
    let mut iters = 0;
    let progress = probe.and_then(|m| m.get_i64("progress"));
    info!("Starting bench itself");
    let mut dur = Duration::default();
    capture_gpu_trace(sub_matches, || -> TractResult<()> {
        while iters < limits.max_loops && dur < limits.max_time {
            if let Some(mon) = probe {
                let _ = mon.log_event(&format!("loop_{iters}"));
            }
            if let Some(p) = &progress {
                p.store(iters as _, std::sync::atomic::Ordering::Relaxed);
            }

            dur += profile_single_turn(state, &inputs)?;

            iters += 1;
        }
        Ok(())
    })?;
    Ok((iters, Duration::from_secs_f64(dur.as_secs_f64() / iters as f64)))
}

pub fn handle(
    params: &Parameters,
    matches: &clap::ArgMatches,
    sub_matches: &clap::ArgMatches,
    limits: &BenchLimits,
    probe: Option<&Probe>,
) -> TractResult<()> {
    let run_params = crate::tensor::run_params_from_subcommand(params, sub_matches)?;
    let mut state = make_state(params, matches, sub_matches)?;

    let inputs = get_or_make_inputs(state.model(), &run_params)?;

    limits.warmup(state.model(), &inputs)?;
    let (iters, dur) = bench(&mut state, sub_matches, inputs, limits, probe)?;

    if params.machine_friendly {
        println!("real: {}", dur.as_secs_f64());
    } else {
        println!("Bench ran {} times, {}.", iters, terminal::dur_avg(dur));
    }

    if let Some(pp) = sub_matches.value_of("pp") {
        let pp = pp.parse::<usize>()?;
        let tokens = pp as f64 / dur.as_secs_f64();
        println!("PP{pp}: {tokens:.1} tokens/sec");
    }

    Ok(())
}
