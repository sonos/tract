use ansi_term::Colour::*;
use atty;
use pbr::ProgressBar;

use log::Level::Info;

use crate::display_graph::DisplayOptions;
use crate::errors::*;
use crate::{Parameters, ProfilingMode, SomeModel};

use crate::format::*;
use crate::profile::ProfileData;
use crate::rusage::{Duration, Instant};
use crate::tensor::make_inputs;

use tract_core::internal::*;

pub fn handle_benching(params: Parameters, profiling: ProfilingMode) -> CliResult<()> {
    match &params.tract_model {
        SomeModel::Inference(m) => handle_benching_t(m, &params, profiling),
        SomeModel::Typed(m) => handle_benching_t(m, &params, profiling),
        SomeModel::Normalized(m) => handle_benching_t(m, &params, profiling),
        SomeModel::Pulsed(_, m) => handle_benching_t(m, &params, profiling),
    }
}

pub fn make_inputs_for_model<TI: TensorInfo>(model: &Model<TI>) -> CliResult<TVec<Tensor>> {
    Ok(make_inputs(
        &*model
            .input_outlets()?
            .iter()
            .map(|&t| Ok(model.outlet_fact(t)?.to_tensor_fact()))
            .collect::<TractResult<Vec<_>>>()?,
    )?)
}

fn handle_benching_t<TI: TensorInfo>(
    model: &Model<TI>,
    params: &Parameters,
    profiling: ProfilingMode,
) -> CliResult<()> {
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
    let dur = Duration::since(&start, iters);

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
    params: Parameters,
    profiling: ProfilingMode,
    display_options: DisplayOptions,
) -> CliResult<()> {
    match &params.tract_model {
        SomeModel::Inference(ref m) => handle_t(m, &params, profiling, display_options),
        SomeModel::Typed(ref m) => handle_t(m, &params, profiling, display_options),
        SomeModel::Normalized(ref m) => handle_t(m, &params, profiling, display_options),
        SomeModel::Pulsed(_, ref m) => handle_t(m, &params, profiling, display_options),
    }
}

/// Handles the `profile` subcommand when there are no streaming dimensions.
pub fn handle_t<TI: TensorInfo>(
    model: &Model<TI>,
    params: &Parameters,
    profiling: ProfilingMode,
    mut display_options: DisplayOptions,
) -> CliResult<()> {
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
    let entire = Duration::since(&start, iters);

    info!("Running {} iterations max. for each node.", max_iters);
    info!("Running for {} ms max. for each node.", max_time);

    let mut state = SimpleState::new(&plan)?;
    state.set_inputs(make_inputs_for_model(model)?)?;
    debug!("Using execution plan: {:?}", plan);

    let mut profile = ProfileData::default();
    let mut progress = ProgressBar::new(plan.order.len() as u64);

    if log_enabled!(Info) {
        println!();
        print_header(format!("Profiling for {}:", params.name), &White.normal());
    }

    // Then execute the plan while profiling each step.
    for &n in &plan.order {
        let node = &model.nodes()[n];

        if atty::is(atty::Stream::Stdout) {
            progress.inc();
        }

        if node.op.name() == "Source" {
            if log_enabled!(Info) {
                print_node(
                    &node,
                    &params.graph,
                    Some(&state),
                    &[Yellow.paint("SKIP").to_string()],
                    vec![],
                );
            }

            continue;
        }

        if !display_options.filter(model, node)? {
            continue
        }
        state.compute_recursively(n)?;

        let mut iters = 0;
        let start = Instant::now();

        while iters < max_iters && start.elapsed_real() < (max_time as f64 * 1e-3) {
            state.compute_one(n)?;
            iters += 1;
        }

        let measure = Duration::since(&start, iters);
        profile.add(&node, measure)?;
    }

    if display_options == DisplayOptions::default() {
        display_options.node_ids = Some(profile.most_consuming_nodes()?);
    };

    let mut display_graph =
        crate::display_graph::DisplayGraph::from_model_and_options(model, display_options)?
            .with_graph_def(&params.graph)?;

    let sum = profile.summed();
    for (ix, measure) in profile.nodes.iter() {
        display_graph.add_node_label(*ix, dur_avg_oneline_ratio(*measure, sum))?;
    }

    if atty::is(atty::Stream::Stdout) {
        progress.finish_print("");
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
