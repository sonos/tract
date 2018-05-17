#[macro_use]
extern crate clap;
extern crate colored;
#[cfg(feature = "tensorflow")]
extern crate conform;
#[macro_use]
extern crate error_chain;
#[macro_use]
extern crate log;
extern crate rand;
extern crate simplelog;
extern crate tfdeploy;
extern crate time;

use std::process;

use simplelog::{Config, LevelFilter, TermLogger};
use tfdeploy::tfpb;
#[cfg(feature = "tensorflow")]
use tfdeploy::Matrix;
use tfpb::graph::GraphDef;
use tfpb::types::DataType;
use time::PreciseTime as Time;

use errors::*;
use format::print_node;
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
const DEFAULT_MAX_ITERS: i64 = 10000;
const DEFAULT_MAX_TIME: i64 = 10;

/// Structure holding the parsed parameters.
struct Parameters {
    file: String,
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

        (@arg debug: -d ... "Sets the level of debugging information.")

        (@subcommand compare =>
            (about: "Compares the output of tfdeploy and tensorflow on randomly generated input."))

        (@subcommand profile =>
            (about: "Benchmarks tfdeploy on randomly generated input.")
            (@arg max_iters: -n [max_iters]
                "Sets the maximum number of iterations for each node [default: 10000].")
            (@arg max_time: -t [max_time]
                "Sets the maximum execution time for each node (in ms) [default: 10]."))
    );

    let matches = app.get_matches();

    // Configure the logging level.
    let level = match matches.occurrences_of("debug") {
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
                Some(s) => s.parse::<i64>()?,
            },
            match m.value_of("max_time") {
                None => DEFAULT_MAX_TIME,
                Some(s) => s.parse::<i64>()?,
            },
        ),

        (s, _) => bail!("Unknown subcommand {}.", s),
    }
}

/// Parses the command-line arguments.
fn parse(matches: &clap::ArgMatches) -> Result<Parameters> {
    let file = matches.value_of("model").unwrap();
    let graph = tfdeploy::Model::graphdef_for_path(&file)?;
    let tfd_model = tfdeploy::for_path(&file)?;

    #[cfg(feature = "tensorflow")]
    let tf_model = conform::tf::for_path(&file)?;

    let splits: Vec<&str> = matches.value_of("size").unwrap().split("x").collect();

    if splits.len() < 2 {
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
        file: file.to_string(),
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
        path: path.to_string(),
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

    println!();
    println!("Comparing the execution of {}:", params.file);

    for n in plan {
        let node = tfd.get_node_by_id(n)?;

        if node.op_name == "Placeholder" {
            print_node(
                node,
                &params.graph,
                &state,
                "SKIP".yellow().to_string(),
                vec![],
            );

            continue;
        }

        let tf_output = tf_outputs.remove(&node.name.to_string()).expect(
            format!(
                "No node with name {} was computed by tensorflow.",
                node.name
            ).as_str(),
        );

        let (status, mismatches) = match state.compute_one(n) {
            Err(e) => (
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

                        ("MISM.".red(), mismatches)
                    }

                    _ => ("OK".green(), vec![]),
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

        // Print the results for the node.
        print_node(
            node,
            &params.graph,
            &state,
            status.to_string(),
            vec![outputs, mismatches],
        );

        // Re-use the output from tensorflow to keep tfdeploy from drifting.
        state.set_outputs(node.id, tf_output)?;
    }

    println!();
    Ok(())
}

/// Handles the `profile` subcommand.
fn handle_profile(params: Parameters, max_iters: i64, max_time: i64) -> Result<()> {
    use colored::Colorize;

    let model = params.tfd_model;
    let output = model.get_node_by_id(params.output)?;
    let mut state = model.state();

    // First fill the inputs with randomly generated values.
    for s in params.inputs {
        state.set_value(s, random_matrix(params.sizes.clone(), params.datatype))?;
    }

    let plan = output.eval_order(&model)?;
    info!("Using execution plan: {:?}", plan);
    info!("Running {} iterations max. for each node.", max_iters);
    info!("Running for {} ms max. for each node.", max_time);

    println!();
    println!("Profiling the execution of {}:", params.file);

    // Then execute the plan while profiling each step.
    for n in plan {
        let node = model.get_node_by_id(n)?;

        if node.op_name == "Placeholder" {
            print_node(
                node,
                &params.graph,
                &state,
                "SKIP".yellow().to_string(),
                vec![],
            );

            continue;
        }

        let mut iters = 0;
        let start = Time::now();

        while iters < max_iters && start.to(Time::now()).num_milliseconds() < max_time {
            state.compute_one(n)?;
            iters += 1;
        }

        let time = start.to(Time::now()).num_milliseconds();

        // Print the results for the node.
        print_node(
            node,
            &params.graph,
            &state,
            format!("{:.*} ms", 6, time as f64 / iters as f64)
                .white()
                .to_string(),
            vec![],
        );
    }

    println!();

    Ok(())
}
