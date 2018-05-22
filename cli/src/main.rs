#[macro_use]
extern crate clap;
extern crate colored;
#[cfg(feature = "tensorflow")]
extern crate conform;
#[macro_use]
extern crate error_chain;
#[macro_use]
extern crate log;
extern crate ndarray;
#[macro_use]
extern crate prettytable;
extern crate rand;
extern crate simplelog;
extern crate textwrap;
extern crate tfdeploy;

use std::collections::HashMap;
use std::process;
use std::time::Instant;

use simplelog::{Config, LevelFilter, TermLogger};
use simplelog::Level::Info;
use tfdeploy::tfpb;
#[cfg(feature = "tensorflow")]
use tfdeploy::Matrix;
use tfpb::graph::GraphDef;
use tfpb::types::DataType;

use errors::*;
use format::print_header;
use format::print_node;
#[allow(unused_imports)]
use format::Row;
#[cfg(feature = "tensorflow")]
use utils::compare_outputs;
use utils::detect_inputs;
use utils::detect_output;
use utils::random_matrix;

mod errors;
mod format;
mod utils;

/// The default maximum for iterations and time.
const DEFAULT_MAX_ITERS: u32 = 10_000;
const DEFAULT_MAX_TIME: u32 = 10_000;

/// Structure holding the parsed parameters.
struct Parameters {
    name: String,
    graph: GraphDef,
    tfd_model: tfdeploy::Model,

    #[cfg(feature = "tensorflow")]
    tf_model: conform::tf::Tensorflow,

    inputs: Vec<usize>,
    output: usize,

    sizes: Vec<usize>,
    datatype: DataType,
}

/// Entrypoint for the command-line interface.
fn main() {
    let app = clap_app!(("tfdeploy-cli") =>
        (version: "1.0")
        (author: "Romain Liautaud <romain.liautaud@snips.ai>")
        (about: "A set of tools to compare tfdeploy with tensorflow.")

        (@setting UnifiedHelpMessage)
        (@setting SubcommandRequired)
        (@setting DeriveDisplayOrder)

        (@arg model: +required +takes_value
            "Sets the TensorFlow model to use (in Protobuf format).")

        (@arg inputs: -i --input ... [input]
            "Sets the input nodes names (auto-detects otherwise).")

        (@arg output: -o --output [output]
            "Sets the output node name (auto-detects otherwise).")

        (@arg size: -s --size <size>
            "Sets the input size, e.g. 32x64xf32.")

        (@arg verbosity: -v ... "Sets the level of verbosity.")

        (@subcommand compare =>
            (about: "Compares the output of tfdeploy and tensorflow on randomly generated input."))

        (@subcommand profile =>
            (about: "Benchmarks tfdeploy on randomly generated input.")
            (@arg max_iters: -n [max_iters]
                "Sets the maximum number of iterations for each node [default: 10_000].")
            (@arg max_time: -t [max_time]
                "Sets the maximum execution time for each node (in ns) [default: 10_000]."))
    );

    let matches = app.get_matches();

    // Configure the logging level.
    let level = match matches.occurrences_of("verbosity") {
        0 => LevelFilter::Warn,
        1 => LevelFilter::Info,
        2 => LevelFilter::Debug,
        _ => LevelFilter::Trace,
    };

    TermLogger::init(level, Config::default()).unwrap();

    if let Err(e) = handle(matches) {
        error!("{}", e.to_string());
        process::exit(1)
    }
}

/// Handles the command-line input.
fn handle(matches: clap::ArgMatches) -> Result<()> {
    let params = parse(&matches)?;

    match matches.subcommand() {
        ("compare", _) => handle_compare(params),

        ("profile", None) => handle_profile(params, DEFAULT_MAX_ITERS, DEFAULT_MAX_TIME),

        ("profile", Some(m)) => handle_profile(
            params,
            match m.value_of("max_iters") {
                None => DEFAULT_MAX_ITERS,
                Some(s) => s.parse::<u32>()?,
            },
            match m.value_of("max_time") {
                None => DEFAULT_MAX_TIME,
                Some(s) => s.parse::<u32>()?,
            },
        ),

        (s, _) => bail!("Unknown subcommand {}.", s),
    }
}

/// Parses the command-line arguments.
fn parse(matches: &clap::ArgMatches) -> Result<Parameters> {
    let name = matches.value_of("model").unwrap();
    let graph = tfdeploy::Model::graphdef_for_path(&name)?;
    let tfd_model = tfdeploy::for_path(&name)?;

    #[cfg(feature = "tensorflow")]
    let tf_model = conform::tf::for_path(&name)?;

    let splits: Vec<&str> = matches.value_of("size").unwrap().split("x").collect();

    if splits.len() < 1 {
        bail!("Size should be formatted as {size}x{...}x{type}.");
    }

    let (datatype, sizes) = splits.split_last().unwrap();
    let sizes: Vec<usize> = sizes.iter().map(|s| Ok(s.parse()?)).collect::<Result<_>>()?;
    let datatype = match datatype.to_lowercase().as_str() {
        "f64" => DataType::DT_DOUBLE,
        "f32" => DataType::DT_FLOAT,
        "i32" => DataType::DT_INT32,
        "i8" => DataType::DT_INT8,
        "u8" => DataType::DT_UINT8,
        _ => bail!("Type of the input should be f64, f32, i32, i8 or u8."),
    };

    let inputs = match matches.values_of("inputs") {
        Some(names) => names
            .map(|s| Ok(tfd_model.node_id_by_name(s)?))
            .collect::<Result<_>>()?,
        None => detect_inputs(&tfd_model)?
            .ok_or("Impossible to auto-detect input nodes: no placeholder.")?,
    };

    let output = match matches.value_of("output") {
        Some(name) => tfd_model.node_id_by_name(name)?,
        None => detect_output(&tfd_model)?.ok_or("Impossible to auto-detect output nodes.")?,
    };

    #[cfg(feature = "tensorflow")]
    return Ok(Parameters {
        name: name.to_string(),
        graph,
        tfd_model,
        tf_model,
        inputs,
        output,
        sizes,
        datatype,
    });

    #[cfg(not(feature = "tensorflow"))]
    return Ok(Parameters {
        name: name.to_string(),
        graph,
        tfd_model,
        inputs,
        output,
        sizes,
        datatype,
    });
}

/// Handles the `compare` subcommand.
#[cfg(not(feature = "tensorflow"))]
fn handle_compare(_params: Parameters) -> Result<()> {
    bail!("Comparison requires the `tensorflow` feature.")
}

#[cfg(feature = "tensorflow")]
fn handle_compare(params: Parameters) -> Result<()> {
    use colored::Colorize;

    let tfd = params.tfd_model;
    let mut tf = params.tf_model;

    let output = tfd.get_node_by_id(params.output)?;
    let mut state = tfd.state();

    // First generate random values for the inputs.
    let mut generated = Vec::new();
    for i in params.inputs {
        generated.push((
            tfd.get_node_by_id(i)?.name.as_str(),
            random_matrix(params.sizes.clone(), params.datatype),
        ));
    }

    // Execute the model on tensorflow first.
    info!("Running the model on tensorflow.");
    let mut tf_outputs = tf.run_get_all(generated.clone())?;

    // Execute the model step-by-step on tfdeploy.
    state.set_values(generated)?;
    let plan = output.eval_order(&tfd)?;
    info!("Using execution plan: {:?}", plan);

    if log_enabled!(Info) {
        print_header(format!("Detailed comparison for {}:", params.name), "white");
    }

    let mut failures = vec![];

    for n in plan {
        let node = tfd.get_node_by_id(n)?;

        if node.op_name == "Placeholder" {
            if log_enabled!(Info) {
                print_node(
                    node,
                    &params.graph,
                    &state,
                    "SKIP".yellow().to_string(),
                    vec![],
                );
            }

            continue;
        }

        let tf_output = tf_outputs.remove(&node.name.to_string()).expect(
            format!(
                "No node with name {} was computed by tensorflow.",
                node.name
            ).as_str(),
        );

        let (failure, status, mismatches) = match state.compute_one(n) {
            Err(e) => (
                true,
                "ERROR".red(),
                vec![Row::Simple(format!("Error message: {:?}", e))],
            ),

            _ => {
                let tfd_output = state.outputs[n].as_ref().unwrap();
                let views = tfd_output.iter().map(|m| &**m).collect::<Vec<&Matrix>>();

                match compare_outputs(&tf_output, &views) {
                    Err(_) => {
                        let mismatches: Vec<_> = tfd_output
                            .iter()
                            .enumerate()
                            .map(|(n, data)| {
                                let header = format!("{} (TFD):", format!("Output {}", n).bold(),);

                                let reason = if n >= tf_output.len() {
                                    "Too many outputs"
                                } else if tf_output[n].shape() != data.shape() {
                                    "Wrong shape"
                                } else if !tf_output[n].close_enough(data) {
                                    "Too far away"
                                } else {
                                    "Other error"
                                };

                                let infos = data.partial_dump(false).unwrap();

                                Row::Double(header, format!("{}\n{}", reason.to_string(), infos))
                            })
                            .collect();

                        (true, "MISM.".red(), mismatches)
                    }

                    _ => (false, "OK".green(), vec![]),
                }
            }
        };

        let outputs: Vec<_> = tf_output
            .iter()
            .enumerate()
            .map(|(n, data)| Row::Double(
                format!("{} (TF):", format!("Output {}", n).bold()),
                data.partial_dump(false).unwrap(),
            ))
            .collect();

        if log_enabled!(Info) {
            // Print the results for the node.
            print_node(
                node,
                &params.graph,
                &state,
                status.to_string(),
                vec![outputs.clone(), mismatches.clone()],
            );
        }

        if failure {
            failures.push((node, status, outputs, mismatches));
        }

        // Re-use the output from tensorflow to keep tfdeploy from drifting.
        state.set_outputs(node.id, tf_output)?;
    }

    if log_enabled!(Info) {
        println!();
    }

    if failures.len() > 0 {
        print_header(format!("There were {} errors:", failures.len()), "red");

        for (node, status, outputs, mismatches) in failures {
            print_node(
                node,
                &params.graph,
                &state,
                status.to_string(),
                vec![outputs, mismatches],
            );
        }
    } else {
        println!("{}", "Each node passed the comparison.".bold().green());
    }

    Ok(())
}

/// Handles the `profile` subcommand.
fn handle_profile(params: Parameters, max_iters: u32, max_time: u32) -> Result<()> {
    use colored::Colorize;

    // The number of nanoseconds since a start time as an u32.
    macro_rules! elapsed_ns {
        ($start:ident) => ({
            let duration = $start.elapsed();
            duration.as_secs() as u32 * 1_000_000 + duration.subsec_nanos()
        })
    }

    let model = params.tfd_model;
    let output = model.get_node_by_id(params.output)?;
    let mut state = model.state();

    // First fill the inputs with randomly generated values.
    for s in params.inputs {
        state.set_value(s, random_matrix(params.sizes.clone(), params.datatype))?;
    }

    let global_start = Instant::now();
    let plan = output.eval_order(&model)?;
    info!("Using execution plan: {:?}", plan);
    info!("Running {} iterations max. for each node.", max_iters);
    info!("Running for {} ms max. for each node.", max_time);

    if log_enabled!(Info) {
        print_header(format!("Detailed profiling for {}:", params.name), "white");
    }

    let mut total = 0.;
    let mut total_avg = 0.;
    let mut nodes = vec![];
    let mut operations = HashMap::new();

    // Then execute the plan while profiling each step.
    for n in plan {
        let node = model.get_node_by_id(n)?;

        if node.op_name == "Placeholder" {
            if log_enabled!(Info) {
                print_node(
                    node,
                    &params.graph,
                    &state,
                    "SKIP".yellow().to_string(),
                    vec![],
                );
            }

            continue;
        }

        let mut iters = 0;
        let start = Instant::now();

        while iters < max_iters && elapsed_ns!(start) < max_time {
            state.compute_one(n)?;
            iters += 1;
        }

        let elapsed = elapsed_ns!(start);
        let time = elapsed as f64 * 1e-6;
        let time_avg = elapsed as f64 / iters as f64 * 1e-6;
        let status = format!("{:.3} ms", time_avg).white();

        // Print the results for the node.
        if log_enabled!(Info) {
            print_node(
                node,
                &params.graph,
                &state,
                status.to_string(),
                vec![],
            );
        }

        total += time;
        total_avg += time_avg;
        nodes.push((node, time, status));
        let mut pair = operations
            .entry(node.op_name.as_str())
            .or_insert((0., 0));
        pair.0 += time;
        pair.1 += 1;
    }

    let global_elapsed = elapsed_ns!(global_start);
    let global_time = global_elapsed as f64 * 1e-6;

    if log_enabled!(Info) {
        println!();
    }

    print_header(format!("Summary for {}:", params.name), "white");

    println!();
    println!("Most time consuming nodes:");
    nodes.sort_by(
        |(_, a, _), (_, b, _)| a.partial_cmp(b).unwrap().reverse()
    );
    for (node, _, status) in nodes.iter().take(5) {
        print_node(
            node,
            &params.graph,
            &state,
            status.to_string(),
            vec![],
        );
    }

    println!();
    println!("Total execution time:");
    println!("{} (for {} nodes).", format!("{:.3} ms", global_time).yellow().bold(), nodes.len());

    println!();
    println!("Total node execution time:");
    println!("- {} in total.", format!("{:.3} ms", total).purple().bold());
    println!("- {} averaged.", format!("{:.3} ms", total_avg).purple().bold());

    // We don't display this when verbose logging is enabled, because the results
    // are skewed by the time it takes to call `print_node` for each node.
    if !log_enabled!(Info) {
        println!("- {} of total execution time.", format!("{:.0}%", total / global_time * 100.).purple().bold());
    }

    println!();
    println!("Most time consuming operations:");
    let mut operations: Vec<_> = operations
        .iter()
        .map(|(o, (t, c))| (o, t, c))
        .collect();
    operations.sort_by(
        |(_, a, _), (_, b, _)| a.partial_cmp(b).unwrap().reverse()
    );
    for (operation, time, count) in operations.iter().take(5) {
        println!("- {}: {:.3} ms (for {} nodes).", operation.blue().bold(), time, count);
    }

    Ok(())
}
