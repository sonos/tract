#[macro_use]
extern crate clap;
extern crate colored;
#[cfg(feature = "tensorflow")]
extern crate conform;
extern crate dot;
#[macro_use]
extern crate error_chain;
#[macro_use]
extern crate log;
extern crate ndarray;
#[macro_use]
extern crate prettytable;
extern crate rand;
extern crate simplelog;
extern crate terminal_size;
extern crate textwrap;
#[macro_use]
extern crate tfdeploy;
extern crate pbr;
extern crate atty;
extern crate libc;
#[macro_use]
extern crate rouille;
extern crate open;
extern crate serde;
#[macro_use]
extern crate serde_derive;
extern crate serde_json;
extern crate bincode;

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::process;
use std::thread;

use simplelog::Level::{Error, Info, Trace};
use simplelog::{Config, LevelFilter, TermLogger};
use tfdeploy::analyser::Analyser;
use tfdeploy::analyser::detect_inputs;
use tfdeploy::analyser::detect_output;
use tfdeploy::analyser::{TensorFact, ShapeFact, DimFact};
use tfdeploy::tfpb;
use tfdeploy::Tensor;
use tfpb::graph::GraphDef;
use tfpb::types::DataType;
use pbr::ProgressBar;
use ndarray::Axis;

use errors::*;
use format::print_header;
use format::print_node;
#[allow(unused_imports)]
use format::Row;
#[cfg(feature = "tensorflow")]
use utils::compare_outputs;
use utils::random_tensor;
use utils::generate_json;
use rusage::{ Duration, Instant };

mod errors;
mod format;
mod graphviz;
mod utils;
mod rusage;

/// The default maximum for iterations and time.
const DEFAULT_MAX_ITERS: u64 = 100_000;
const DEFAULT_MAX_TIME: u64 = 200;

/// Structure holding the input data.
struct InputData {
    data: Option<Tensor>,
    shape: Vec<Option<usize>>,
    datatype: DataType,
}

/// Structure holding the parsed parameters.
struct Parameters {
    name: String,
    graph: GraphDef,
    tfd_model: tfdeploy::Model,

    #[cfg(feature = "tensorflow")]
    tf_model: conform::tf::Tensorflow,

    input: Option<InputData>,
    inputs: Vec<usize>,
    output: usize,
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

        (@arg size: -s --size [size]
            "Generates random input of a given size, e.g. 32x64xf32.")

        (@arg data: -f --data [data]
            "Loads input data from a given file.")

        (@arg verbosity: -v ... "Sets the level of verbosity.")

        (@subcommand compare =>
            (about: "Compares the output of tfdeploy and tensorflow on randomly generated input."))

        (@subcommand dump =>
            (about: "Dumps the Tensorflow graph in human readable form.")
            (@arg web: --web
                "Displays the dump in a web interface."))

        (@subcommand profile =>
            (about: "Benchmarks tfdeploy on randomly generated input.")
            (@arg max_iters: -n [max_iters]
                "Sets the maximum number of iterations for each node [default: 10_000].")
            (@arg max_time: -t [max_time]
                "Sets the maximum execution time for each node (in ms) [default: 500]."))

        (@subcommand analyse =>
            (about: "Analyses the graph to infer properties about tensors (experimental).")
            (@arg prune: --prune
                "Prunes constant nodes and edges from the graph.")
            (@arg web: --web
                "Displays the results of the analysis in a web interface."))
    );

    let matches = app.get_matches();

    // Configure the logging level.
    let level = match matches.occurrences_of("verbosity") {
        0 => LevelFilter::Warn,
        1 => LevelFilter::Info,
        2 => LevelFilter::Debug,
        _ => LevelFilter::Trace,
    };

    TermLogger::init(
        level,
        Config {
            time: None,
            time_format: None,
            level: Some(Error),
            target: None,
            location: Some(Trace),
        },
    ).unwrap();

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

        ("dump", Some(m)) => handle_dump(
            params,
            m.is_present("web")
        ),

        ("profile", Some(m)) => handle_profile(
            params,
            match m.value_of("max_iters") {
                None => DEFAULT_MAX_ITERS,
                Some(s) => s.parse::<u64>()?,
            },
            match m.value_of("max_time") {
                None => DEFAULT_MAX_TIME,
                Some(s) => s.parse::<u64>()?,
            },
        ),

        ("analyse", Some(m)) => handle_analyse(
            params,
            m.is_present("prune"),
            m.is_present("web")
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

    let input = match (matches.value_of("size"), matches.value_of("data")) {
        (Some(size), None)     => Some(parse_size(size)?),
        (None, Some(filename)) => Some(parse_data(filename)?),
        _ => None
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
        input,
    });

    #[cfg(not(feature = "tensorflow"))]
    return Ok(Parameters {
        name: name.to_string(),
        graph,
        tfd_model,
        inputs,
        output,
        input,
    });
}

/// Parses the `size` command-line argument.
fn parse_size(size: &str) -> Result<InputData> {
    let splits = size.split("x").collect::<Vec<_>>();

    if splits.len() < 1 {
        bail!("The <size> argument should be formatted as {size}x{...}x{type}.");
    }

    let (datatype, shape) = splits.split_last().unwrap();

    let shape = shape
        .iter()
        .map(|s| match *s {
            "S" => Ok(None),            // Streaming dimension.
            _   => Ok(Some(s.parse()?)) // Regular dimension.
        })
        .collect::<Result<Vec<_>>>()?;

    if shape.iter().filter(|o| o.is_none()).count() > 1 {
        bail!("The <size> argument doesn't support more than one streaming dimension.");
    }

    let datatype = match datatype.to_lowercase().as_str() {
        "f64" => DataType::DT_DOUBLE,
        "f32" => DataType::DT_FLOAT,
        "i32" => DataType::DT_INT32,
        "i8" => DataType::DT_INT8,
        "u8" => DataType::DT_UINT8,
        _ => bail!("Type of the input should be f64, f32, i32, i8 or u8."),
    };

    Ok(InputData { data: None, shape, datatype })
}


/// Parses the `data` command-line argument.
fn parse_data(filename: &str) -> Result<InputData> {
    let mut file = File::open(filename)?;
    let mut data = String::new();
    file.read_to_string(&mut data)?;

    let mut lines = data.lines();
    let InputData { shape, datatype, .. } = parse_size(lines.next().unwrap())?;

    let values = lines
        .flat_map(|l| l.split_whitespace())
        .collect::<Vec<_>>();

    // We know there is at most one streaming dimension, so we can deduce the
    // missing value with a simple division.
    let product: usize =  shape.iter().map(|o| o.unwrap_or(1)).product();
    let missing = values.len() / product;
    let data_shape = shape.iter()
        .map(|o| o.unwrap_or(missing))
        .collect::<Vec<_>>();

    macro_rules! for_type {
        ($t:ty) => ({
            let array = ndarray::Array::from_iter(
                values.iter().map(|v| v.parse::<$t>().unwrap())
            );

            array.into_shape(data_shape)?
        });
    }

    let tensor = match datatype {
        DataType::DT_DOUBLE => for_type!(f64).into(),
        DataType::DT_FLOAT => for_type!(f32).into(),
        DataType::DT_INT32 => for_type!(i32).into(),
        DataType::DT_INT8 => for_type!(i8).into(),
        DataType::DT_UINT8 => for_type!(u8).into(),
        _ => unimplemented!("missing type"),
    };

    Ok(InputData { data: Some(tensor), shape, datatype })
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

    let input = params.input
        .ok_or("Exactly one of <size> or <data> must be specified.")?;

    let shape = input.shape.iter().cloned().collect::<Option<Vec<_>>>()
        .ok_or("The compare command doesn't support streaming dimensions.")?;

    // First generate random values for the inputs.
    let mut generated = Vec::new();
    for i in params.inputs {
        let data = if input.data.is_some() {
            input.data.as_ref().unwrap().clone()
        } else {
            random_tensor(shape.clone(), input.datatype)
        };

        generated.push((
            tfd.get_node_by_id(i)?.name.as_str(),
            data,
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
                    Some(&state),
                    vec!["SKIP".yellow().to_string()],
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
                let views = tfd_output.iter().map(|m| &**m).collect::<Vec<&Tensor>>();

                match compare_outputs(&tf_output, &views) {
                    Err(_) => {
                        let mismatches = tfd_output
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
                            .collect::<Vec<_>>();

                        (true, "MISM.".red(), mismatches)
                    }

                    _ => (false, "OK".green(), vec![]),
                }
            }
        };

        let outputs = tf_output
            .iter()
            .enumerate()
            .map(|(n, data)| {
                Row::Double(
                    format!("{} (TF):", format!("Output {}", n).bold()),
                    data.partial_dump(false).unwrap(),
                )
            })
            .collect::<Vec<_>>();

        if log_enabled!(Info) {
            // Print the results for the node.
            print_node(
                node,
                &params.graph,
                Some(&state),
                vec![status.to_string()],
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
                Some(&state),
                vec![status.to_string()],
                vec![outputs, mismatches],
            );
        }
    } else {
        println!("{}", "Each node passed the comparison.".bold().green());
    }

    Ok(())
}

fn handle_dump(params: Parameters, web: bool) -> Result<()> {
    let tfd = params.tfd_model;
    let output = tfd.get_node_by_id(params.output)?;
    let plan = output.eval_order(&tfd)?;

    if web {
        let data = generate_json(&tfd)?;
        open_web(data);
    } else {
        for n in plan {
            let node = tfd.get_node_by_id(n)?;
            print_node(
                node,
                &params.graph,
                None,
                vec![],
                vec![],
            );
        }
    }

    Ok(())
}

/// Handles the `profile` subcommand.
fn handle_profile(mut params: Parameters, max_iters: u64, max_time: u64) -> Result<()> {
    let input = params.input
        .take()
        .ok_or("Exactly one of <size> or <data> must be specified.")?;

    match input.shape.iter().cloned().collect::<Option<Vec<_>>>() {
        Some(shape) =>
            handle_profile_regular(params, input, max_iters, max_time, shape),
        None =>
            handle_profile_streaming(params, input, max_iters, max_time),
    }
}

/// Handles the `profile` subcommand when there are no streaming dimensions.
fn handle_profile_regular(params: Parameters, input: InputData, max_iters: u64, max_time: u64, shape: Vec<usize>) -> Result<()> {
    use colored::Colorize;

    let model = params.tfd_model;
    let output = model.get_node_by_id(params.output)?;
    let mut state = model.state();

    // First fill the inputs with randomly generated values.
    for s in params.inputs {
        let data = if input.data.is_some() {
            input.data.as_ref().unwrap().clone()
        } else {
            random_tensor(shape.clone(), input.datatype)
        };

        state.set_value(s, data)?;
    }

    info!("Running {} iterations max. for each node.", max_iters);
    info!("Running for {} ms max. for each node.", max_time);

    let mut global = Duration::new();
    let capacity = model.nodes().len();
    let mut nodes = Vec::with_capacity(capacity);
    let mut operations = HashMap::with_capacity(capacity);

    let plan = output.eval_order(&model)?;
    info!("Using execution plan: {:?}", plan);

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

        global += measure;
        nodes.push((node, measure));
        let mut pair = operations
            .entry(node.op_name.as_str())
            .or_insert((Duration::new(), 0));
        pair.0 += measure;
        pair.1 += 1;
    }

    if atty::is(atty::Stream::Stdout) {
        progress.finish_print("");
    }

    println!();
    print_header(format!("Summary for {}:", params.name), "white");

    println!();
    println!("Most time consuming nodes:");
    nodes.sort_by(|(_, a), (_, b)| a.avg_real.partial_cmp(&b.avg_real).unwrap().reverse());
    for (node, measure) in nodes.iter().take(5) {
        let status_real = format!(
            "Real: {} ({:.1?}%)",
            format!("{:.3} ms/i", measure.avg_real * 1e3).white(),
            measure.avg_real / global.avg_real * 100.
        );

        let status_user = format!(
            "User: {} ({:.1?}%)",
            format!("{:.3} ms/i", measure.avg_user * 1e3).white(),
            measure.avg_user / global.avg_user * 100.
        );

        let status_sys = format!(
            "Sys: {} ({:.1?}%)",
            format!("{:.3} ms/i", measure.avg_sys * 1e3).white(),
            measure.avg_sys / global.avg_sys * 100.
        );

        print_node(
            node,
            &params.graph,
            Some(&state),
            vec![status_real.to_string(), status_user.to_string(), status_sys.to_string()],
            vec![]
        );
    }

    println!();
    println!("Total execution time (for {} nodes):", nodes.len());
    println!("- Real: {}.", format!("{:.3} ms/i", global.avg_real * 1e3).yellow().bold());
    println!("- User: {}.", format!("{:.3} ms/i", global.avg_user * 1e3).yellow().bold());
    println!("- Sys: {}.", format!("{:.3} ms/i", global.avg_sys * 1e3).yellow().bold());

    if log_enabled!(Info) {
        println!(
            "(Real: {} in total, with max_iters={:e} and max_time={:?}ms.)",
            format!("{:.3} ms", global.total_real * 1e3).white(),
            max_iters as f32,
            max_time,
        );
    }

    println!();
    println!("Most time consuming operations:");
    let mut operations = operations.iter()
        .map(|(o, (measure, c))| (o, measure, c))
        .collect::<Vec<_>>();
    operations.sort_by(|(_, a, _), (_, b, _)| a.avg_real.partial_cmp(&b.avg_real).unwrap().reverse());
    for (operation, measure, count) in operations.iter().take(5) {
        println!(
            "- {} (for {} nodes):",
            operation.blue().bold(), count
        );

        println!(
            "    - Real: {} ({:.2?}%).",
            format!("{:.3} ms/i", measure.avg_real * 1e3).white().bold(),
            measure.avg_real / global.avg_real * 100.
        );

        println!(
            "    - User: {} ({:.2?}%).",
            format!("{:.3} ms/i", measure.avg_user * 1e3).white().bold(),
            measure.avg_user / global.avg_user * 100.
        );

        println!(
            "    - Sys: {} ({:.2?}%).",
            format!("{:.3} ms/i", measure.avg_sys * 1e3).white().bold(),
            measure.avg_sys / global.avg_sys * 100.
        );

        if log_enabled!(Info) {
            println!("    - {:.3} ms in total.", measure.total_real * 1e3);
        }
    }

    Ok(())
}

/// Handles the `profile` subcommand when there are streaming dimensions.
fn handle_profile_streaming(params: Parameters, input: InputData, _max_iters: u64, _max_time: u64) -> Result<()> {
    use Tensor::*;
    use colored::Colorize;

    use tfdeploy::StreamingInput;
    use tfdeploy::StreamingState;

    let model = params.tfd_model;
    let datatype = input.datatype;
    let shape = input.shape;

    let axis = shape.iter()
        .position(|&d| d == None)
        .unwrap();
    let inputs = params.inputs.iter()
        .map(|&s| (s, StreamingInput::Streamed(datatype, shape.clone())))
        .collect::<Vec<_>>();

    info!("Initializing the StreamingState.");
    let start = Instant::now();
    let state = StreamingState::start(
        model.clone(),
        inputs.clone(),
        Some(params.output)
    )?;

    let measure = Duration::since(&start, 1);
    let mut states = (0..100).map(|_| state.clone()).collect::<Vec<_>>();

    info!("Initialized the StreamingState in:");
    info!(
        "    - Real: {}.",
        format!("{:.3} ms/i", measure.avg_real * 1e3).white().bold(),
    );

    info!(
        "    - User: {}.",
        format!("{:.3} ms/i", measure.avg_user * 1e3).white().bold(),
    );

    info!(
        "    - Sys: {}.",
        format!("{:.3} ms/i", measure.avg_sys * 1e3).white().bold(),
    );

    if log_enabled!(Info) {
        println!();
        print_header(format!("Streaming profiling for {}:", params.name), "white");
    }

    // Either get the input data from the input file or generate it randomly.
    let random_shape = shape.iter()
        .map(|d| d.unwrap_or(20))
        .collect::<Vec<_>>();
    let data = input.data
        .unwrap_or_else(|| random_tensor(random_shape, datatype));

    // Split the input data into chunks along the streaming axis.
    macro_rules! split_inner {
        ($constr:ident, $array:expr) => ({
            $array.axis_iter(Axis(axis))
                .map(|v| $constr(v.insert_axis(Axis(axis)).to_owned()))
                .collect::<Vec<_>>()
        })
    }

    let chunks = match data {
        F64(m) => split_inner!(F64, m),
        F32(m) => split_inner!(F32, m),
        I32(m) => split_inner!(I32, m),
        I8(m) => split_inner!(I8, m),
        U8(m) => split_inner!(U8, m),
        String(m) => split_inner!(String, m),
    };

    for (step, chunk) in chunks.into_iter().enumerate() {
        for &input in &params.inputs {
            println!();
            println!("Starting step {:?} with input {:?}.", step, chunk);

            let mut input_chunks = vec![Some(chunk.clone()); 100];
            let mut outputs = Vec::with_capacity(100);
            let start = Instant::now();

            for i in 0..100 {
                outputs.push(states[i].step(input, input_chunks[i].take().unwrap())?);
            }

            let measure = Duration::since(&start, 100);
            println!("Completed step {:?} with output {:?} in:", step, outputs[0]);
            println!(
                "    - Real: {}.",
                format!("{:.3} ms/i", measure.avg_real * 1e3).white().bold(),
            );

            println!(
                "    - User: {}.",
                format!("{:.3} ms/i", measure.avg_user * 1e3).white().bold(),
            );

            println!(
                "    - Sys: {}.",
                format!("{:.3} ms/i", measure.avg_sys * 1e3).white().bold(),
            );

            thread::sleep(::std::time::Duration::from_secs(1));
        }
    }

    Ok(())
}

/// Handles the `analyse` subcommand.
fn handle_analyse(params: Parameters, prune: bool, web: bool) -> Result<()> {
    use std::fs::File;
    use std::io::prelude::*;

    let model = params.tfd_model;
    let output = model.get_node_by_id(params.output)?.id;

    info!("Starting the analysis.");

    let mut analyser = Analyser::new(model, output)?;

    // Add hints for the input nodes.
    if let Some(input) = params.input {
        let dims = input.shape.iter()
            .map(|d| match d {
                None    => DimFact::Streamed,
                Some(i) => DimFact::Only(*i),
            })
            .collect::<Vec<_>>();

        for &i in &params.inputs {
            analyser.hint(i, &TensorFact {
                datatype: typefact!(input.datatype),
                shape: ShapeFact::closed(dims.clone()),
                value: valuefact!(_),
            })?;
        }
    }

    analyser.run()?;

    // Prune constant nodes if needed.
    if prune {
        info!(
            "Size of the graph before pruning: approx. {:.2?} Ko for {:?} nodes.",
            bincode::serialize(&analyser.nodes)?.len() as f64 * 1e-3,
            analyser.nodes.len()
        );

        analyser.propagate_constants()?;
        analyser.prune_unused();

        info!(
            "Size of the graph after pruning: approx. {:.2?} Ko for {:?} nodes.",
            bincode::serialize(&analyser.nodes)?.len() as f64 * 1e-3,
            analyser.nodes.len()
        );
    }


    // Display an interactive view of the graph if needed.
    let data = serde_json::to_vec(&(&analyser.nodes, &analyser.edges)).unwrap();
    if web {
        open_web(data);
    } else {
        File::create("analyser.json")?.write_all(&data)?;
        println!("Wrote the result of the analysis to analyser.json.");
    }

    Ok(())
}

/// Starts a web server for TFVisualizer and opens its webroot in a browser.
fn open_web(data: Vec<u8>) -> () {
    use rouille::Response;

    println!("TFVisualizer is now running on http://127.0.0.1:8000/.");
    let _ = open::that("http://127.0.0.1:8000/");

    rouille::start_server("0.0.0.0:8000", move |request| {
        if request.remove_prefix("/dist").is_some() || request.remove_prefix("/public").is_some() {
            return rouille::match_assets(&request, "../visualizer");
        }

        return router!(request,
            (GET) (/) => {
                let index = File::open("../visualizer/index.html").unwrap();
                Response::from_file("text/html", index)
            },

            (GET) (/current) => {
                Response::from_data("application/json", data.clone())
            },

            _ => Response::empty_404(),
        );
    });
}

/// Handles the `prune` subcommand.
#[allow(dead_code)]
fn handle_prune(params: Parameters) -> Result<()> {
    let model = params.tfd_model;
    let output = model.get_node_by_id(params.output)?.id;

    info!("Starting the analysis.");

    let mut analyser = Analyser::new(model, output)?;

    info!(
        "Starting size of the graph: approx. {:?} bytes for {:?} nodes.",
        format!("{:?}", analyser.nodes).into_bytes().len(),
        analyser.nodes.len()
    );

    analyser.run()?;
    analyser.propagate_constants()?;
    analyser.prune_unused();

    info!(
        "Ending size of the graph: approx. {:?} bytes for {:?} nodes.",
        format!("{:?}", analyser.nodes).into_bytes().len(),
        analyser.nodes.len()
    );

    Ok(())
}
