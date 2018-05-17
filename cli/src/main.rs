#[macro_use]
extern crate clap;
extern crate colored;
#[cfg(feature="tensorflow")]
extern crate conform;
#[macro_use]
extern crate error_chain;
#[macro_use]
extern crate log;
extern crate rand;
extern crate simplelog;
extern crate tfdeploy;
extern crate time;

use std::path;
use std::process;

use simplelog::{TermLogger, LevelFilter, Config};
#[cfg(feature="tensorflow")]
use tfdeploy::Matrix;
use tfdeploy::tfpb;
use tfpb::types::DataType;
use time::PreciseTime;

use errors::*;
#[cfg(feature="tensorflow")]
use utils::compare_outputs;
use utils::detect_inputs;
use utils::detect_output;
use utils::random_matrix;

mod format;
mod utils;
mod errors;


/// The default number of iterations for the profiler.
const DEFAULT_ITERS: usize = 10000;


/// Structure holding the parsed parameters.
struct Parameters {
    path: String,
    graph: tfpb::graph::GraphDef,
    tfd_model: tfdeploy::Model,

    #[cfg(feature="tensorflow")]
    tf_model: conform::tf::Tensorflow,

    inputs: Vec<String>,
    output: String,

    sizes: Vec<usize>,
    datatype: DataType
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
            (@arg iters: -n [iters]
                "Sets the number of iterations for the average [default: 10000]."))
    );

    let matches = app.get_matches();

    // Configure the logging level.
    let level = match matches.occurrences_of("debug") {
        0 => LevelFilter::Warn,
        1 => LevelFilter::Info,
        2 => LevelFilter::Debug,
        _ => LevelFilter::Trace
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
        ("compare", _) =>
            handle_compare(params),

        ("profile", None) =>
            handle_profile(params, DEFAULT_ITERS),

        ("profile", Some(m)) =>
            handle_profile(params, match m.value_of("iters") {
                None => DEFAULT_ITERS,
                Some(s) => s.parse::<usize>()?
            }),

        (s, _) => bail!("Unknown subcommand {}.", s)
    }
}


/// Parses the command-line arguments.
fn parse(matches: &clap::ArgMatches) -> Result<Parameters> {
    let path = matches.value_of("model").unwrap();
    let graph = tfdeploy::Model::graphdef_for_path(&path::Path::new(path))?;
    let tfd_model = tfdeploy::for_path(&path)?;

    #[cfg(feature="tensorflow")]
    let tf_model = conform::tf::for_path(&path)?;

    let splits: Vec<&str> = matches
        .value_of("size")
        .unwrap()
        .split("x")
        .collect();

    if splits.len() < 2 {
        bail!("Size should be formatted as {size}x{...}x{type}.");
    }

    let (datatype, sizes) = splits.split_last().unwrap();

    let sizes: Vec<usize> = sizes
        .iter()
        .map(|s| Ok(s.parse()?))
        .collect::<Result<_>>()?;

    let datatype = match datatype.to_lowercase().as_str() {
        "f64" => DataType::DT_DOUBLE,
        "f32" => DataType::DT_FLOAT,
        "i32" => DataType::DT_INT32,
        "i8" => DataType::DT_INT8,
        "u8" => DataType::DT_UINT8,
        _ => bail!("Type of the input should be f64, f32, i32, i8 or u8.")
    };

    let inputs = match matches.values_of("inputs") {
        Some(names) => names.map(|s| s.to_string()).collect(),
        None => detect_inputs(&tfd_model)?
            .ok_or("Impossible to auto-detect input nodes: no placeholder.")?
    };

    let output = match matches.value_of("output") {
        Some(name) => name.to_string(),
        None => detect_output(&tfd_model)?
            .ok_or("Impossible to auto-detect output nodes.")?
    };

    #[cfg(feature="tensorflow")]
    return Ok(Parameters {
        path: path.to_string(),
        graph, tfd_model, tf_model,
        inputs, output, sizes, datatype
    });

    #[cfg(not(feature="tensorflow"))]
    return Ok(Parameters {
        path: path.to_string(),
        graph, tfd_model,
        inputs, output, sizes, datatype
    });
}


/// Handles the `compare` subcommand.
#[cfg(not(feature="tensorflow"))]
fn handle_compare(_params: Parameters) -> Result<()> {
    bail!("Comparison requires the `tensorflow` feature.")
}

#[cfg(feature="tensorflow")]
fn handle_compare(params: Parameters) -> Result<()> {
    use colored::Colorize;

    let tfd = params.tfd_model;
    let mut tf = params.tf_model;

    let output = tfd.get_node(params.output.as_str())?;
    let mut state = tfd.state();

    // First generate random values for the inputs.
    let mut generated = Vec::new();
    for s in &params.inputs {
        generated.push((
            s.as_str(),
            random_matrix(params.sizes.clone(), params.datatype)
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
    println!("Comparing the execution of {}:", params.path);

    for n in plan {
        let node = tfd.get_node_by_id(n)?;

        if node.op_name == "Placeholder" {
            format::print_box(
                node.id.to_string(),
                node.op_name.to_string(),
                node.name.to_string(),
                "SKIP".yellow().to_string(),
                format::node_info(node, &params.graph, &state)?
            );

            continue;
        }

        let tf_output = tf_outputs
            .remove(&node.name.to_string())
            .expect(format!("No node with name {} was computed by tensorflow.", node.name).as_str());


        let (status, mismatches) = match state.compute_one(n) {
            Err(e) => ("ERROR".red(), vec![format!("Error message: {:?}", e)]),

            _ => {
                let tfd_output = state.outputs[n].as_ref().unwrap();
                let views = tfd_output.iter().map(|m| &**m).collect::<Vec<&Matrix>>();

                match compare_outputs(&tf_output, &views) {
                    Err(_) => {
                        let mut mismatches = vec![];

                        for (n, data) in tfd_output.iter().enumerate() {
                            let header = format!(
                                "{} (TFD):",
                                format!("Output {}", n).bold(),
                            );

                            let reason = if n >= tf_output.len() {
                                "Too many outputs"
                            } else if tf_output[n].shape() != data.shape() {
                                "Wrong shape"
                            } else if !tf_output[n].close_enough(data) {
                                "Too far away"
                            } else {
                                "Other error"
                            };

                            mismatches.extend(
                                format::with_header(header.clone(), reason.yellow().to_string(), 80)
                            );

                            mismatches.extend(
                                format::with_header(
                                    format!("{:1$}", "", header.len() - format::hidden_len(&header)),
                                    data.partial_dump(false).unwrap(),
                                    80
                                )
                            );
                        }

                        ("MISM.".red(), mismatches)
                    },

                    _ => ("OK".green(), vec![])
                }
            }
        };

        let mut information = format::node_info(node, &params.graph, &state)?;

        let mut outputs = Vec::new();
        for (ix, data) in tf_output.iter().enumerate() {
            outputs.extend(
                format::with_header(
                    format!(
                        "{} (TF):",
                        format!("Output {}", ix).bold(),
                    ),
                    data.partial_dump(false).unwrap(),
                    80
                )
            );
        }

        information.push(outputs);
        information.push(mismatches);

        // Print the results for the node.
        format::print_box(
            node.id.to_string(),
            node.op_name.to_string(),
            node.name.to_string(),
            status.to_string(),
            information
        );

        // Re-use the output from tensorflow to keep tfdeploy from drifting.
        state.set_outputs(node.id, tf_output)?;
    }

    println!();
    Ok(())
}


/// Handles the `profile` subcommand.
fn handle_profile(params: Parameters, iters: usize) -> Result<()> {
    use colored::Colorize;

    let model = params.tfd_model;
    let output = model.get_node(params.output.as_str())?;
    let mut state = model.state();

    // First fill the inputs with randomly generated values.
    for s in params.inputs {
        state.set_value(
            model.node_id_by_name(s.as_str())?,
            random_matrix(params.sizes.clone(), params.datatype)
        )?;
    }

    let plan = output.eval_order(&model)?;
    info!("Using execution plan: {:?}", plan);
    info!("Running {} iterations at each step", iters);

    println!();
    println!("Profiling the execution of {}:", params.path);

    // Then execute the plan while profiling each step.
    for n in plan {
        let node = model.get_node_by_id(n)?;

        if node.op_name == "Placeholder" {
            format::print_box(
                node.id.to_string(),
                node.op_name.to_string(),
                node.name.to_string(),
                "SKIP".yellow().to_string(),
                format::node_info(node, &params.graph, &state)?
            );

            continue;
        }

        let start = PreciseTime::now();
        for _ in 0..iters { state.compute_one(n)?; }
        let end = PreciseTime::now();

        // Print the results for the node.
        format::print_box(
            node.id.to_string(),
            node.op_name.to_string(),
            node.name.to_string(),
            format!(
                "{} ms",
                start.to(end).num_milliseconds() as f64 / iters as f64
            ).white().to_string(),
            format::node_info(node, &params.graph, &state)?
        );
    }

    println!();

    Ok(())
}