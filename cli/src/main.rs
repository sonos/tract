#[macro_use]
extern crate clap;
extern crate colored;
#[cfg(feature = "tensorflow")]
extern crate conform;
extern crate dot;
#[macro_use]
extern crate error_chain;
extern crate insideout;
extern crate itertools;
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
extern crate atty;
extern crate libc;
extern crate pbr;
extern crate tfdeploy_onnx;
extern crate tfdeploy_tf;
#[macro_use]
extern crate rouille;
extern crate open;
extern crate serde;
#[macro_use]
extern crate serde_derive;
extern crate serde_json;

use std::process;
use std::str::FromStr;

use insideout::InsideOut;
use simplelog::Level::{Error, Trace};
use simplelog::{Config, LevelFilter, TermLogger};
use tfdeploy::analyser::TensorFact;
use tfdeploy_tf::tfpb;
use tfpb::graph::GraphDef;

use errors::*;
#[allow(unused_imports)]
use format::Row;

mod analyse;
mod compare;
mod display_graph;
mod dump;
mod errors;
mod format;
// mod graphviz;
mod optimize_check;
mod profile;
mod prune;
mod run;
mod rusage;
mod stream_check;
mod tensor;
mod utils;
mod web;

/// The default maximum for iterations and time.
const DEFAULT_MAX_ITERS: u64 = 100_000;
const DEFAULT_MAX_TIME: u64 = 5000;

/// Entrypoint for the command-line interface.
fn main() {
    use clap::*;
    let mut app = clap_app!(("tfdeploy-cli") =>
        (version: "1.0")
        (author: "Romain Liautaud <romain.liautaud@snips.ai>")
        (about: "A set of tools to compare tfdeploy with tensorflow.")

        (@setting UnifiedHelpMessage)
        (@setting SubcommandRequired)
        (@setting DeriveDisplayOrder)

        (@arg model: +required +takes_value "Sets the model to use")

        (@arg format: +takes_value
            "Hint the model format ('onnx' or 'tf') instead of guess from extension.")

        (@arg input: -i --input +takes_value
            "Set input value (@file or 3x4xi32)")

        (@arg input_node: --("input-node") +takes_value
            "Override input nodes names (auto-detects otherwise).")

        (@arg output_node: --("output-node") +takes_value
            "Override output nodes name (auto-detects otherwise).")

        (@arg skip_analyse: --("skip-analyse") ... "Skip analyse after model build")

        (@arg verbosity: -v ... "Sets the level of verbosity.")
    );

    let compare = clap::SubCommand::with_name("compare")
        .help("Compares the output of tfdeploy and tensorflow on randomly generated input.");
    app = app.subcommand(output_options(compare));

    let dump = clap::SubCommand::with_name("dump")
        .help("Dumps the Tensorflow graph in human readable form.");
    app = app.subcommand(output_options(dump));

    let profile = clap::SubCommand::with_name("profile")
        .help("Benchmarks tfdeploy on randomly generated input.")
        .arg(
            Arg::with_name("bench")
                .long("bench")
                .help("Run as an overall bench"),
        )
        .arg(
            Arg::with_name("max_iters")
                .takes_value(true)
                .long("max-iters")
                .short("n")
                .help("Sets the maximum number of iterations for each node [default: 100_000]."),
        )
        .arg(
            Arg::with_name("max-time")
                .takes_value(true)
                .long("max-time")
                .help("Sets the maximum execution time for each node (in ms) [default: 5000]."),
        )
        .arg(
            Arg::with_name("buffering")
                .short("b")
                .help("Run the stream network without inner instrumentations"),
        );
    app = app.subcommand(output_options(profile));

    let run = clap::SubCommand::with_name("run")
        .help("Run the graph")
        .arg(
            Arg::with_name("assert-output")
                .takes_value(true)
                .long("assert-output")
                .help("Fact to check the ouput tensor against (@filename, or 3x4xf32)"),
        );
    app = app.subcommand(output_options(run));

    let analyse = clap::SubCommand::with_name("analyse")
        .help("Analyses the graph to infer properties about tensors (experimental).");
    app = app.subcommand(output_options(analyse));

    let optimize = clap::SubCommand::with_name("optimize").help("Optimize the graph");
    app = app.subcommand(output_options(optimize));

    let optimize_check = clap::SubCommand::with_name("optimize-check")
        .help("Compare output of optimized and un-optimized graph");
    app = app.subcommand(output_options(optimize_check));

    let stream_check = clap::SubCommand::with_name("stream-check")
        .help("Compare output of streamed and regular exec");
    app = app.subcommand(output_options(stream_check));

    let matches = app.get_matches();

    if let Err(e) = handle(matches) {
        error!("{}", e.to_string());
        process::exit(1)
    }
}

fn output_options<'a, 'b>(command: clap::App<'a, 'b>) -> clap::App<'a, 'b> {
    use clap::*;
    command
        .arg(
            Arg::with_name("web")
                .long("web")
                .help("Display int a web interface"),
        )
        .arg(
            Arg::with_name("json")
                .long("json")
                .takes_value(true)
                .help("output to a json file"),
        )
        .arg(
            Arg::with_name("quiet")
                .short("q")
                .long("quiet")
                .help("don't dump"),
        )
        .arg(
            Arg::with_name("node_id")
                .long("node-id")
                .takes_value(true)
                .help("Select a node to dump"),
        )
        .arg(
            Arg::with_name("successors")
                .long("successors")
                .takes_value(true)
                .help("Show successors of node"),
        )
        .arg(
            Arg::with_name("op_name")
                .long("op-name")
                .takes_value(true)
                .help("Select one op to dump"),
        )
        .arg(
            Arg::with_name("node_name")
                .long("node-name")
                .takes_value(true)
                .help("Select one node to dump"),
        )
        .arg(
            Arg::with_name("const")
                .long("const")
                .help("also display consts nodes"),
        )
}

#[derive(Debug)]
pub enum SomeGraphDef {
    Tf(GraphDef),
    Onnx(tfdeploy_onnx::pb::ModelProto),
}

/// Structure holding the parsed parameters.
#[derive(Debug)]
pub struct Parameters {
    name: String,
    graph: SomeGraphDef,
    tfd_model: tfdeploy::Model,

    #[cfg(feature = "tensorflow")]
    tf_model: Option<conform::tf::Tensorflow>,

    #[cfg(not(feature = "tensorflow"))]
    #[allow(dead_code)]
    tf_model: (),

    inputs: Option<Vec<Option<tfdeploy::Tensor>>>,
}

impl Parameters {
    /// Parses the command-line arguments.
    pub fn from_clap(matches: &clap::ArgMatches) -> CliResult<Parameters> {
        let name = matches.value_of("model").unwrap();
        let format = matches
            .value_of("format")
            .unwrap_or(if name.ends_with(".onnx") {
                "onnx"
            } else {
                "tf"
            });
        let (graph, mut tfd_model) = if format == "onnx" {
            let graph = tfdeploy_onnx::model::model_proto_for_path(&name)?;
            let tfd = tfdeploy_onnx::for_path(&name)?;
            (SomeGraphDef::Onnx(graph), tfd)
        } else {
            let graph = tfdeploy_tf::model::graphdef_for_path(&name)?;
            let tfd_model = tfdeploy_tf::for_path(&name)?;
            (SomeGraphDef::Tf(graph), tfd_model)
        };

        info!("Model {:?} loaded", name);

        #[cfg(feature = "tensorflow")]
        let tf_model = if format == "tf" {
            Some(conform::tf::for_path(&name)?)
        } else {
            None
        };

        #[cfg(not(feature = "tensorflow"))]
        let tf_model = ();

        if let Some(inputs) = matches.values_of("input_node") {
            tfd_model.set_inputs(inputs)?;
        };

        if let Some(outputs) = matches.values_of("output_node") {
            tfd_model.set_outputs(outputs)?;
        };

        let inputs = if let Some(inputs) = matches.values_of("input") {
            use tfdeploy::analyser::Fact;
            let mut vs = vec!();
            for (ix, v) in inputs.enumerate() {
                let t = tensor::for_string(v)?;
                // obliterate value in input (the analyser/optimizer would fold
                // the graph)
                let fact = TensorFact {
                    value: tfdeploy::analyser::GenericFact::Any, ..t
                };
                vs.push(t.value.concretize());
                let outlet = tfd_model.inputs()?[ix];
                tfd_model.set_fact(outlet, fact)?;
            }
            Some(vs)
        } else {
            None
        };

        if !matches.is_present("skip_analyse") {
            info!("Skipping analyse");
            tfd_model.analyse()?;
        }

        Ok(Parameters {
            name: name.to_string(),
            graph,
            tfd_model,
            tf_model,
            inputs
        })
    }
}

pub enum ProfilingMode {
    Regular { max_iters: u64, max_time: u64 },
    RegularBenching { max_iters: u64, max_time: u64 },
    StreamCruising,
    StreamBuffering,
    StreamBenching { max_iters: u64, max_time: u64 },
}

impl ProfilingMode {
    pub fn from_clap(matches: &clap::ArgMatches, streaming: bool) -> CliResult<ProfilingMode> {
        let max_iters = matches
            .value_of("max_iters")
            .map(u64::from_str)
            .inside_out()?
            .unwrap_or(DEFAULT_MAX_ITERS);
        let max_time = matches
            .value_of("max_time")
            .map(u64::from_str)
            .inside_out()?
            .unwrap_or(DEFAULT_MAX_TIME);
        let mode = if streaming {
            if matches.is_present("buffering") {
                ProfilingMode::StreamBuffering
            } else if matches.is_present("bench") {
                ProfilingMode::StreamBenching {
                    max_iters,
                    max_time,
                }
            } else {
                ProfilingMode::StreamCruising
            }
        } else {
            if matches.is_present("bench") {
                ProfilingMode::RegularBenching {
                    max_iters,
                    max_time,
                }
            } else {
                ProfilingMode::Regular {
                    max_iters,
                    max_time,
                }
            }
        };
        Ok(mode)
    }
}

pub struct OutputParameters {
    web: bool,
    konst: bool,
    quiet: bool,
    json: Option<String>,
    node_id: Option<usize>,
    op_name: Option<String>,
    node_name: Option<String>,
    successors: Option<usize>,
}

impl OutputParameters {
    pub fn from_clap(matches: &clap::ArgMatches) -> CliResult<OutputParameters> {
        Ok(OutputParameters {
            web: matches.is_present("web"),
            konst: matches.is_present("const"),
            quiet: matches.is_present("quiet"),
            json: matches.value_of("json").map(String::from),
            node_id: matches.value_of("node_id").map(|id| id.parse().unwrap()),
            node_name: matches.value_of("node_name").map(String::from),
            op_name: matches.value_of("op_name").map(String::from),
            successors: matches.value_of("successors").map(|id| id.parse().unwrap()),
        })
    }
}

/// Handles the command-line input.
fn handle(matches: clap::ArgMatches) -> CliResult<()> {
    // Configure the logging level.
    let level = match matches.occurrences_of("verbosity") {
        0 => LevelFilter::Warn,
        1 => LevelFilter::Info,
        2 => LevelFilter::Debug,
        _ => LevelFilter::Trace,
    };

    let log_config = Config {
        time: None,
        time_format: None,
        level: Some(Error),
        target: None,
        location: Some(Trace),
    };

    if TermLogger::init(level, log_config).is_err()
        && simplelog::SimpleLogger::init(level, log_config).is_err()
    {
        panic!("Could not initiatize logger")
    };

    let params = Parameters::from_clap(&matches)?;
    let streaming = params.tfd_model.input_fact()?.stream_info()?.is_some();

    match matches.subcommand() {
        ("compare", Some(m)) => compare::handle(params, OutputParameters::from_clap(m)?),

        ("run", Some(m)) => {
            let assert_outputs: Option<Vec<TensorFact>> = m
                .values_of("assert-output")
                .map(|vs| vs.map(|v| tensor::for_string(v).unwrap()).collect());
            run::handle(params, assert_outputs, OutputParameters::from_clap(m)?)
        }

        ("optimize-check", Some(m)) => {
            optimize_check::handle(params, OutputParameters::from_clap(m)?)
        }

        ("stream-check", Some(m)) => stream_check::handle(params, OutputParameters::from_clap(m)?),

        ("dump", Some(m)) => dump::handle(params, OutputParameters::from_clap(m)?),

        ("profile", Some(m)) => profile::handle(
            params,
            ProfilingMode::from_clap(&m, streaming)?,
            OutputParameters::from_clap(m)?,
        ),

        ("analyse", Some(m)) => analyse::handle(params, false, OutputParameters::from_clap(m)?),

        ("optimize", Some(m)) => analyse::handle(params, true, OutputParameters::from_clap(m)?),

        (s, _) => bail!("Unknown subcommand {}.", s),
    }
}
