use atty;
use pbr::ProgressBar;
use simplelog::Level::Info;

use { Parameters, InputParameters, ProfilingMode, OutputParameters };
use errors::*;
use utils::random_tensor;

use profile::ProfileData;
use rusage::{ Instant, Duration };
use format::*;

use tfdeploy::ModelState;

fn make_state(params: &Parameters) -> Result<ModelState> {
    let ref model = params.tfd_model;
    let mut state = model.state();
    let shape:Vec<usize> = params.input.as_ref().unwrap().shape.iter().map(|s| s.unwrap()).collect(); // checked by clap and dispatcher
    // First fill the inputs with randomly generated values.
    for s in &params.input_node_ids {
        let given_input:&InputParameters = params.input.as_ref()
            .ok_or("Exactly one of <size> or <data> must be specified.")?;
        let data = if let Some(value) = given_input.data.as_ref() {
            value.clone()
        } else {
            random_tensor(shape.clone(), given_input.datatype)
        };

        state.set_value(*s, data)?;
    }
    Ok(state)
}

pub fn handle_benching(params: Parameters, profiling: ProfilingMode, _output_params:OutputParameters) -> Result<()> {
    let (max_iters, max_time) = if let ProfilingMode::RegularBenching { max_iters, max_time } = profiling {
        (max_iters, max_time)
    } else {
        bail!("Expecting bench profile mode")
    };

    let ref model = params.tfd_model;
    let state = make_state(&params)?;

    let plan = model.plan_for_one(params.output_node_id)?;
    info!("Starting bench itself");
    let mut iters = 0;
    let start = Instant::now();
    while iters < max_iters && start.elapsed_real() < (max_time as f64 * 1e-3) {
        let mut cloned = state.clone();
        let _ = plan.run(&mut cloned)?;
        iters += 1;
    }
    let dur = Duration::since(&start, iters);

    println!("Bench ran {} times in {}", iters, dur_avg_oneline(dur));
    Ok(())
}

/// Handles the `profile` subcommand when there are no streaming dimensions.
pub fn handle(params: Parameters, profiling: ProfilingMode, output_parameters: OutputParameters) -> Result<()> {
    use colored::Colorize;

    let (max_iters, max_time) = if let ProfilingMode::Regular { max_iters, max_time } = profiling {
        (max_iters, max_time)
    } else {
        bail!("Expecting regular profile mode")
    };

    let ref model = params.tfd_model;
    let output = model.get_node_by_id(params.output_node_id)?;
    let mut state = make_state(&params)?;

    let plan = model.plan_for_one(params.output_node_id)?;
    info!("Running entire network");
    let mut iters = 0;
    let start = Instant::now();
    while iters < max_iters && start.elapsed_real() < (max_time as f64 * 1e-3) {
        let mut cloned = state.clone();
        let _ = plan.run(&mut cloned)?;
        iters += 1;
    }
    let entire = Duration::since(&start, iters);

    info!("Running {} iterations max. for each node.", max_iters);
    info!("Running for {} ms max. for each node.", max_time);

    let plan = output.eval_order(&model)?;
    info!("Using execution plan: {:?}", plan);

    let mut profile = ProfileData::new(model);
    let mut progress = ProgressBar::new(plan.len() as u64);

    if log_enabled!(Info) {
        println!();
        print_header(format!("Profiling for {}:", params.name), "white");
    }

    // Then execute the plan while profiling each step.
    for n in plan {
        let node = model.get_node_by_id(n)?;

        if atty::is(atty::Stream::Stdout) {
            progress.inc();
        }

        if node.op_name == "Placeholder" {
            if log_enabled!(Info) {
                print_node(node, &params.graph, Some(&state), &["SKIP".yellow().to_string()], vec![]);
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
            print_node(node, &params.graph, Some(&state), &[
                format!("{:.3} ms/i", measure.avg_real() * 1e3).white().to_string()
            ], vec![]);
        }

        profile.add(&node, measure)?;
    }

    if atty::is(atty::Stream::Stdout) {
        progress.finish_print("");
    }

    print_header(format!("Summary for {}:", params.name), "white");

    profile.print_most_consuming_nodes(&params.tfd_model, &params.graph, &output_parameters)?;
    println!();

    profile.print_most_consuming_ops(&params.tfd_model)?;
    println!();

    println!("Entire network performance: {}", dur_avg_oneline(entire));
    println!("Accounted by ops: {}", dur_avg_oneline_ratio(profile.summed(), entire));


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

