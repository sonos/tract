use crate::Parameters;
use tract_hir::internal::*;
use tract_libcli::profile::{BenchLimits, run_one_step};
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

pub fn handle(
    params: &Parameters,
    sub_matches: &clap::ArgMatches,
    limits: &BenchLimits,
) -> TractResult<()> {
    let run_params = crate::tensor::run_params_from_subcommand(params, sub_matches)?;
    let inputs = get_or_make_inputs(&params.tract_model, &run_params)?;

    limits.warmup(&params.req_runnable()?, &inputs)?;
    let (iters, dur) = limits.bench(&params.req_runnable()?, &inputs)?;
    let dur = dur.div_f64(iters as _);

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
