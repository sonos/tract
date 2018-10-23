use atty;
use pbr::ProgressBar;

use log::Level::Info;

use display_graph::DisplayOptions;
use errors::*;
use {Parameters, ProfilingMode};

use format::*;
use profile::ProfileData;
use rusage::{Duration, Instant};
use tensor::make_inputs;

use tfdeploy::plan::{SimplePlan, SimpleState};

pub fn handle_benching(params: Parameters, profiling: ProfilingMode) -> CliResult<()> {
    let (max_iters, max_time) = if let ProfilingMode::RegularBenching {
        max_iters,
        max_time,
    } = profiling
    {
        (max_iters, max_time)
    } else {
        bail!("Expecting bench profile mode")
    };

    let model = &params.tfd_model;
    let plan = SimplePlan::new(model)?;
    let mut state = SimpleState::new(plan)?;
    info!("Starting bench itself");
    let mut iters = 0;
    let start = Instant::now();
    while iters < max_iters && start.elapsed_real() < (max_time as f64 * 1e-3) {
        state.run(make_inputs(&[model.input_fact()?.clone()])?)?;
        iters += 1;
    }
    let dur = Duration::since(&start, iters);

    println!("Bench ran {} times in {}", iters, dur_avg_oneline(dur));
    Ok(())
}

/// Handles the `profile` subcommand when there are no streaming dimensions.
pub fn handle(
    params: Parameters,
    profiling: ProfilingMode,
    display_options: DisplayOptions,
) -> CliResult<()> {
    use colored::Colorize;

    let (max_iters, max_time) = if let ProfilingMode::Regular {
        max_iters,
        max_time,
    } = profiling
    {
        (max_iters, max_time)
    } else {
        bail!("Expecting regular profile mode")
    };

    let ref model = params.tfd_model;

    info!("Running entire network");
    let plan = SimplePlan::new(model)?;
    let mut iters = 0;
    let start = Instant::now();
    while iters < max_iters && start.elapsed_real() < (max_time as f64 * 1e-3) {
        let _ = plan.run(make_inputs(&[params.tfd_model.input_fact()?.clone()])?)?;
        iters += 1;
    }
    let entire = Duration::since(&start, iters);

    info!("Running {} iterations max. for each node.", max_iters);
    info!("Running for {} ms max. for each node.", max_time);

    let mut state = SimpleState::new(&plan)?;
    state.set_inputs(make_inputs(&[params.tfd_model.input_fact()?.clone()])?)?;
    debug!("Using execution plan: {:?}", plan);

    let mut profile = ProfileData::new(model);
    let mut progress = ProgressBar::new(plan.order.len() as u64);

    if log_enabled!(Info) {
        println!();
        print_header(format!("Profiling for {}:", params.name), "white");
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
                    &["SKIP".yellow().to_string()],
                    vec![],
                );
            }

            continue;
        }

        let mut iters = 0;
        let start = Instant::now();

        while iters < max_iters && start.elapsed_real() < (max_time as f64 * 1e-3) {
            state.compute_one(n)?;
            iters += 1;
        }

        let measure = Duration::since(&start, iters);

        // Print the results for the node.
        if log_enabled!(Info) {
            print_node(
                &node,
                &params.graph,
                Some(&state),
                &[format!("{:.3} ms/i", measure.avg_real() * 1e3)
                    .white()
                    .to_string()],
                vec![],
            );
        }

        profile.add(&node, measure)?;
    }

    if atty::is(atty::Stream::Stdout) {
        progress.finish_print("");
    }

    print_header(format!("Summary for {}:", params.name), "white");

    profile.print_most_consuming_nodes(&params.tfd_model, &params.graph, display_options)?;
    println!();

    profile.print_most_consuming_ops(&params.tfd_model)?;
    println!();

    println!("Entire network performance: {}", dur_avg_oneline(entire));
    println!(
        "Accounted by ops: {}",
        dur_avg_oneline_ratio(profile.summed(), entire)
    );

    if log_enabled!(Info) {
        println!(
            "(Real: {} in total, with max_iters={:e} and max_time={:?}ms.)",
            format!("{:.3} ms", profile.summed().total_real * 1e3).white(),
            max_iters as f32,
            max_time,
        );
    }

    Ok(())
}
