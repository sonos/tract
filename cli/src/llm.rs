use crate::bench::bench;;
use crate::Parameters;
use float_ord::FloatOrd;
use readings_probe::Probe;
use std::time::{Duration, Instant};
use tract_core::num_traits::Zero;
use tract_core::tract_data::itertools::Itertools;
use tract_hir::internal::*;
use tract_libcli::profile::BenchLimits;
use tract_libcli::tensor::get_or_make_inputs;
use tract_transformers::figure_out_causal_llm_b_s_p;

pub fn handle(
    params: &Parameters,
    matches: &clap::ArgMatches,
    sub_matches: &clap::ArgMatches,
    limits: &BenchLimits,
    probe: Option<&Probe>,
) -> TractResult<()> {
    bench_pp(params, matches, sub_matches, limits, 512, probe)?;
    bench_tg(params, matches, sub_matches, limits, 128, probe)?;
    Ok(())
}

pub fn bench_pp(
    params: &Parameters,
    _matches: &clap::ArgMatches,
    sub_matches: &clap::ArgMatches,
    limits: &BenchLimits,
    pp: usize,
    probe: Option<&Probe>,
) -> TractResult<()> {
    let mut run_params = crate::tensor::run_params_from_subcommand(params, sub_matches)?;
    run_params.allow_random_input = true;
    let model = params.req_typed_model();
    let state = params.req_runnable()?.spawn()?;
    let mut state: Box<TypedSimpleState> = state.downcast().unwrap();

    let (b, s, p) = figure_out_causal_llm_b_s_p(&model)
        .context("Could not find out LLM symbolic parameters")?;
    if let Some(b) = b {
        run_params.symbols.set(&b, 1);
    }

    ensure!(s.is_some() && p.is_some(), "Could not find LLM symbols in model");
    // Warmup
    run_params.symbols.set(&p.unwrap(), 0);
    run_params.symbols.set(&s.unwrap(), pp as i64);
    let inputs = get_or_make_inputs(&params.tract_model, &run_params)?;
    limits.warmup(state.plan(), &inputs)?;

    let inputs = get_or_make_inputs(&params.tract_model, &run_params)?;

    let (_, dur) = bench(&mut state, sub_matches, inputs, limits, probe)?;
    let tokens = pp as f64 / dur.as_secs_f64();
    println!("PP{pp}: {tokens:.1} tokens/sec");
    Ok(())
}

pub fn bench_tg(
    params: &Parameters,
    _matches: &clap::ArgMatches,
    sub_matches: &clap::ArgMatches,
    limits: &BenchLimits,
    tg: usize,
    probe: Option<&Probe>,
) -> TractResult<()> {
    let mut run_params = crate::tensor::run_params_from_subcommand(params, sub_matches)?;
    run_params.allow_random_input = true;
    let model = params.req_typed_model();
    let state = params.req_runnable()?.spawn()?;
    let mut state: Box<TypedSimpleState> = state.downcast().unwrap();

    let (b, s, p) = figure_out_causal_llm_b_s_p(&model)
        .context("Could not find out LLM symbolic parameters")?;
    if let Some(b) = b {
        run_params.symbols.set(&b, 1);
    }

    ensure!(s.is_some() && p.is_some(), "Could not find LLM symbols in model");
    run_params.symbols.set(&s.unwrap(), 1);

    let p = p.unwrap();
    // Warmup
    if !limits.warmup_loops.is_zero() || !limits.warmup_time.is_zero() {
        let mut iters = 0;
        let max_loops =
            if limits.warmup_loops.is_zero() { usize::MAX } else { limits.warmup_loops };
        let max_time =
            if limits.warmup_time.is_zero() { Duration::MAX } else { limits.warmup_time };
        let start_warmup = Instant::now();
        info!("TG warming before profiling...");
        while iters < max_loops && start_warmup.elapsed() < max_time {
            for t in 0..tg {
                run_params.symbols.set(&p, t as i64);
                let mut inputs = get_or_make_inputs(&params.tract_model, &run_params)?;

                state.run(inputs.sources.remove(0))?;
            }
            state.reset_op_states()?;
            iters += 1;
        }
        info!("Done warming up.");
    }

    // Bench
    let mut tot_dur = Duration::default();
    for t in 0..tg {
        if let Some(p) = probe {
            p.log_event(&format!("Starting token {t}"))?;
        }

        run_params.symbols.set(&p, t as i64);
        let mut inputs = get_or_make_inputs(&params.tract_model, &run_params)?;

        let start = Instant::now();
        state.run(inputs.sources.remove(0))?;
        tot_dur += start.elapsed();
    }
    state.reset_op_states()?;
    let tokens = tg as f64 / tot_dur.as_secs_f64();
    println!("TG{tg}: {tokens:.1} tokens/sec");
    Ok(())
}

pub fn top_logits_levenshtein(test: &Tensor, reference: &Tensor, n: usize) -> TractResult<usize> {
    let (test, reference) = [test, reference]
        .into_iter()
        .map(|t| {
            t.cast_to::<f32>()
                .unwrap()
                .as_slice::<f32>()
                .unwrap()
                .iter()
                .copied()
                .enumerate()
                .sorted_by_key(|(_, f)| FloatOrd(-*f))
                .map(|p| p.0)
                .collect_vec()
        })
        .collect_tuple()
        .unwrap();

    Ok(levenshtein_diff::distance(&test[..n], &reference[..n]).0)
}
