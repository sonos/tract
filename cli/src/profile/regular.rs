use atty;
use pbr::ProgressBar;
use simplelog::Level::Info;

use { Parameters, InputParameters };
use errors::*;
use utils::random_tensor;

use profile::ProfileData;
use rusage::{ Instant, Duration };
use format::*;

/// Handles the `profile` subcommand when there are no streaming dimensions.
pub fn handle(params: Parameters, input: InputParameters, max_iters: u64, max_time: u64, shape: Vec<usize>) -> Result<()> {
    use colored::Colorize;

    let ref model = params.tfd_model;
    let output = model.get_node_by_id(params.output_node_id)?;
    let mut state = model.state();

    // First fill the inputs with randomly generated values.
    for s in &params.input_node_ids {
        let data = if input.data.is_some() {
            input.data.as_ref().unwrap().clone()
        } else {
            random_tensor(shape.clone(), input.datatype)
        };

        state.set_value(*s, data)?;
    }

    info!("Running {} iterations max. for each node.", max_iters);
    info!("Running for {} ms max. for each node.", max_time);

    let plan = output.eval_order(&model)?;
    info!("Using execution plan: {:?}", plan);

    let mut profile = ProfileData::new(&params.graph, model);
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
                print_node(node, &params.graph, Some(&state), vec!["SKIP".yellow().to_string()], vec![]);
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
            print_node(node, &params.graph, Some(&state), vec![
                format!("{:.3} ms/i", measure.avg_real * 1e3).white().to_string()
            ], vec![]);
        }

        profile.add(node.id, measure)?;
    }

    if atty::is(atty::Stream::Stdout) {
        progress.finish_print("");
    }

    println!();
    print_header(format!("Summary for {}:", params.name), "white");

    profile.print_most_consuming_nodes(Some(&state))?;
    println!();

    profile.print_most_consuming_ops();

    if log_enabled!(Info) {
        println!(
            "(Real: {} in total, with max_iters={:e} and max_time={:?}ms.)",
            format!("{:.3} ms", profile.global.total_real * 1e3).white(),
            max_iters as f32,
            max_time,
        );
    }

    Ok(())
}

