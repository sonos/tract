use std::fmt::{Debug, Display};

use ansi_term::Colour::*;
use atty;
use pbr::ProgressBar;

use log::Level::Info;

use crate::display_graph::DisplayOptions;
use crate::errors::*;
use crate::{Model, Parameters, ProfilingMode};

use crate::format::*;
use crate::profile::ProfileData;
use crate::rusage::{Duration, Instant};
use crate::tensor::make_inputs;

use tract_core::internal::*;
use tract_core::infer::*;

pub fn handle_benching(params: &Parameters, profiling: ProfilingMode) -> CliResult<()> {
    dispatch_model!(params.tract_model, |m| handle_benching_t(m, &params, profiling))
}

pub fn make_inputs_for_model<TI, O>(model: &ModelImpl<TI, O>) -> CliResult<TVec<Tensor>>
where
    TI: Fact + Clone + 'static,
    O: AsRef<dyn Op> + AsMut<dyn Op> + Display + Debug + Clone + 'static,
{
    Ok(make_inputs(
        &*model
            .input_outlets()?
            .iter()
            .map(|&t| model.outlet_typedfact(t))
            .collect::<TractResult<Vec<TypedFact>>>()?,
    )?)
}

fn handle_benching_t<TI, O>(
    model: &ModelImpl<TI, O>,
    params: &Parameters,
    profiling: ProfilingMode,
) -> CliResult<()>
where
    TI: Fact + Clone + 'static,
    O: AsRef<dyn Op> + AsMut<dyn Op> + Display + Debug + Clone + 'static,
{
    let (max_iters, max_time) =
        if let ProfilingMode::RegularBenching { max_iters, max_time } = profiling {
            (max_iters, max_time)
        } else {
            bail!("Expecting bench profile mode")
        };

    let plan = SimplePlan::new(model)?;
    let mut state = SimpleState::new(plan)?;
    info!("Starting bench itself");
    let mut iters = 0;
    let start = Instant::now();
    while iters < max_iters && start.elapsed_real() < (max_time as f64 * 1e-3) {
        state.run(make_inputs_for_model(model)?)?;
        iters += 1;
    }
    let mut dur = Duration::since(&start);
    dur /= iters as f64;

    if params.machine_friendly {
        println!("real: {}", dur.avg_real());
        println!("user: {}", dur.avg_user());
        println!("sys: {}", dur.avg_sys());
    } else {
        println!("Bench ran {} times.\n{}", iters, dur_avg_multiline(dur));
    }

    Ok(())
}

pub fn handle(
    params: &Parameters,
    profiling: ProfilingMode,
    display_options: DisplayOptions,
) -> CliResult<()> {
    dispatch_model!(params.tract_model, |m| handle_t(m, &params, profiling, display_options))
}

/// Handles the `profile` subcommand when there are no streaming dimensions.
pub fn handle_t<TI, O>(
    model: &ModelImpl<TI, O>,
    params: &Parameters,
    profiling: ProfilingMode,
    mut display_options: DisplayOptions,
) -> CliResult<()>
where
    TI: Fact + Clone + 'static,
    O: AsRef<dyn Op> + AsMut<dyn Op> + Display + Debug + Clone + 'static,
    ModelImpl<TI, O>: Model,
{
    let (max_iters, max_time) = if let ProfilingMode::Regular { max_iters, max_time } = profiling {
        (max_iters, max_time)
    } else {
        bail!("Expecting regular profile mode")
    };

    info!("Running entire network");
    let plan = SimplePlan::new(model)?;
    let mut iters = 0;
    let start = Instant::now();
    while iters < max_iters && start.elapsed_real() < (max_time as f64 * 1e-3) {
        let _ = plan.run(make_inputs_for_model(model)?)?;
        iters += 1;
    }
    let mut entire = Duration::since(&start);
    entire /= iters as f64;

    info!("Running {} iterations max. for each node.", max_iters);
    info!("Running for {} ms max. for each node.", max_time);

    let mut profile = ProfileData::default();

    let mut queue: Vec<(&ModelImpl<TI, O>, TVec<usize>, f32)> = vec![(model, tvec!(), 1f32)];
    while let Some((model, prefix, multiplier)) = queue.pop() {
        let plan = SimplePlan::new(model)?;
        let mut state = SimpleState::new(&plan)?;
        let mut progress = ProgressBar::new(plan.order.len() as u64);
        let mut full_id: TVec<usize> = prefix.iter().cloned().collect();
        if prefix.len() != 0 {
            info!("Doing subnet {:?}", prefix);
        }
        full_id.push(0);

        state.set_inputs(make_inputs_for_model(model)?)?;

        for &n in &plan.order {
            let node = &model.nodes()[n];

            if atty::is(atty::Stream::Stdout) {
                progress.inc();
            }

            if node.op.as_ref().name() == "Source" {
                continue;
            }

            if !display_options.filter(model, &*prefix, node.id)? {
                continue;
            }
            state.compute_recursively(n)?;

            let mut iters = 0;
            let start = Instant::now();

            while iters < max_iters && start.elapsed_real() < (max_time as f64 * 1e-3) {
                state.compute_one(n)?;
                iters += 1;
            }

            let mut measure = Duration::since(&start);
            measure *= multiplier as f64 / iters as f64;
            full_id[prefix.len()] = n;
            profile.add(&*full_id, measure)?;
            if prefix.len() > 0 {
                profile.sub(&*prefix, measure)?;
            }

            let inputs: TVec<TypedFact> = model
                .node_input_facts(n)?
                .iter()
                .map(|&i| i.to_typed_fact())
                .collect::<TractResult<_>>()?;
            let ref_inputs: TVec<&TypedFact> = inputs.iter().collect();
            let nested_multis =
                model.node_op(n).as_typed().unwrap().nested_model_multipliers(&*ref_inputs);

            for (ix, (_name, m, _, _)) in model.node_op(n).nested_models().iter().enumerate() {
                if let Some(m) = m.downcast_ref::<ModelImpl<TI, O>>() {
                    let mut prefix: TVec<usize> = prefix.clone();
                    prefix.push(n);
                    queue.push((m, prefix, multiplier * nested_multis[ix].1));
                }
            }
        }

        if atty::is(atty::Stream::Stdout) {
            progress.finish_print("");
        }
    }

    if display_options == DisplayOptions::default() {
        display_options.node_ids = Some(profile.most_consuming_nodes()?);
    };

    let mut display_graph = crate::display_graph::DisplayGraph::from_model_and_options(
        model,
        Arc::new(display_options),
    )?
    .with_graph_def(&params.graph)?;

    let sum = profile.summed();
    for (ix, measure) in profile.nodes.iter() {
        display_graph.add_node_label(&ix, dur_avg_oneline_ratio(*measure, sum))?;
    }

    display_graph.render()?;
    println!();

    profile.print_most_consuming_ops(model)?;

    println!("Entire network performance: {}", dur_avg_oneline(entire));
    println!("Accounted by ops: {}", dur_avg_oneline_ratio(profile.summed(), entire));

    if log_enabled!(Info) {
        println!(
            "(Real: {} in total, with max_iters={:e} and max_time={:?}ms.)",
            White.paint(format!("{:.3} ms", profile.summed().total_real * 1e3)),
            max_iters as f32,
            max_time,
        );
    }

    Ok(())
}
