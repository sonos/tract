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
use utils::random_tensor;
use rusage::{ Duration, Instant };

use profile::ProfileData;

mod compare;
mod dump;
mod errors;
mod format;
mod graphviz;
mod utils;
mod profile;
mod rusage;
mod web;

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
pub struct Parameters {
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
        ("compare", _) => compare::handle(params),

        ("dump", Some(m)) => dump::handle(
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
        _ => unimplemented!(),
    };

    Ok(InputData { data: Some(tensor), shape, datatype })
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

    let ref model = params.tfd_model;
    let output = model.get_node_by_id(params.output)?;
    let mut state = model.state();

    // First fill the inputs with randomly generated values.
    for s in &params.inputs {
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

/// Handles the `profile` subcommand when there are streaming dimensions.
fn handle_profile_streaming(params: Parameters, input: InputData, _max_iters: u64, _max_time: u64) -> Result<()> {
    use Tensor::*;

    use tfdeploy::StreamingInput;
    use tfdeploy::StreamingState;

    let model = params.tfd_model.clone();
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

    info!("Initialized the StreamingState in: {}", format::dur_avg_oneline(measure));

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

    let mut profile = ProfileData::new(&params.graph, &state.model());

    for (step, chunk) in chunks.into_iter().enumerate() {
        for &input in &params.inputs {
            trace!("Starting step {:?} with input {:?}.", step, chunk);

            let mut input_chunks = vec![Some(chunk.clone()); 100];
            let mut outputs = Vec::with_capacity(100);
            let start = Instant::now();

            for i in 0..100 {
                outputs.push(states[i].step_wrapping_ops(input, input_chunks[i].take().unwrap(),
                    |node, input, buffer| {
                        let start = Instant::now();
                        let r = node.op.step(input, buffer)?;
                        profile.add(node.id, Duration::since(&start, 1))?;
                        Ok(r)
                    }
                ));
            }

            let measure = Duration::since(&start, 100);
            println!("Completed step {:2} with output {:?} in: {}", step, outputs[0], format::dur_avg_oneline(measure));
            thread::sleep(::std::time::Duration::from_secs(1));
        }
    }

    println!();
    print_header(format!("Summary for {}:", params.name), "white");

    profile.print_most_consuming_nodes(None)?;
    println!();

    profile.print_most_consuming_ops();

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
        ::web::open_web(data);
    } else {
        File::create("analyser.json")?.write_all(&data)?;
        println!("Wrote the result of the analysis to analyser.json.");
    }

    Ok(())
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
