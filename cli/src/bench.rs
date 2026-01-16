use crate::Parameters;
use readings_probe::Probe;
use std::time::{Duration, Instant};
use tract_hir::internal::*;
use tract_libcli::capture_gpu_trace;
use tract_libcli::model::Model;
use tract_libcli::profile::BenchLimits;
use tract_libcli::tensor::RunTensors;
use tract_libcli::tensor::get_or_make_inputs;
use tract_libcli::terminal;

fn profile(state: &mut TypedSimpleState, inputs: &RunTensors) -> TractResult<Duration> {
    if state.model().properties().contains_key("pulse.delay") {
        let start = Instant::now();
        for source in &inputs.sources {
            state.run(source.clone())?;
        }
        Ok(start.elapsed())
    } else {
        state.init_states(&inputs.state_initializers.clone())?;
        let start = Instant::now();
        for source in &inputs.sources {
            state.run(source.clone())?;
        }
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
    let model = params.req_typed_model();
    let mut state = make_state(&model, matches, sub_matches)?;

    let mut crit = criterion::Criterion::default();
    let mut group = crit.benchmark_group("net");

    let run_params = crate::tensor::run_params_from_subcommand(params, sub_matches)?;
    let inputs = get_or_make_inputs(&params.tract_model, &run_params)?;

    group.bench_function("run", move |b| b.iter(|| profile(&mut state, &inputs)));
    Ok(())
}

pub(crate) fn make_state(
    model: &Arc<TypedModel>,
    _matches: &clap::ArgMatches,
    sub_matches: &clap::ArgMatches,
) -> TractResult<TypedSimpleState> {
    #[allow(unused_mut)]
    let mut plan_options = crate::plan_options::plan_options_from_subcommand(sub_matches)?;
    let plan = SimplePlan::new_with_options(model.clone(), &plan_options)?;
    plan.spawn()
}

pub(crate) fn bench(
    state: &mut TypedSimpleState,
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

            dur += profile(state, &inputs)?;

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
    let model = params.req_typed_model();
    let mut state = make_state(&model, matches, sub_matches)?;

    let inputs = get_or_make_inputs(&params.tract_model, &run_params)?;

    limits.warmup(state.plan(), &inputs)?;
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
