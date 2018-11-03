extern crate box_drawing;
extern crate clap;
extern crate colored;
#[macro_use]
extern crate error_chain;
extern crate insideout;
extern crate itertools;
#[macro_use]
extern crate log;
extern crate ndarray;
#[macro_use]
extern crate prettytable;
extern crate atty;
extern crate env_logger;
extern crate libc;
extern crate pbr;
extern crate rand;
extern crate terminal_size;
extern crate textwrap;
#[macro_use]
extern crate tract_core;
extern crate tract_onnx;
extern crate tract_tensorflow;

use std::process;
use std::str::FromStr;

use insideout::InsideOut;
use tfpb::graph::GraphDef;
use tract_core::ops::prelude::*;
use tract_tensorflow::tfpb;

use display_graph::DisplayOptions;
use errors::*;

mod compare;
mod display_graph;
mod draw;
mod dump;
mod errors;
mod format;
// mod optimize_check;
mod profile;
mod run;
mod rusage;
// mod stream_check;
mod tensor;
mod utils;

/// The default maximum for iterations and time.
const DEFAULT_MAX_ITERS: u64 = 100_000;
const DEFAULT_MAX_TIME: u64 = 5000;

/// Entrypoint for the command-line interface.
fn main() {
    use clap::*;
    let mut app = clap_app!(("tract-cli") =>
        (version: "1.0")
        (author: "Romain Liautaud <romain.liautaud@snips.ai>")
        (about: "A set of tools to compare tract with tensorflow.")

        (@setting UnifiedHelpMessage)
        (@setting SubcommandRequired)
        (@setting DeriveDisplayOrder)

        (@arg model: +required +takes_value "Sets the model to use")

        (@arg format: +takes_value
            "Hint the model format ('onnx' or 'tf') instead of guess from extension.")

        (@arg input: -i --input +takes_value
            "Set input value (@file or 3x4xi32)")

        (@arg stream_axis: -s --("stream-axis") +takes_value
            "Set Axis number to stream upon (first is 0)")

        (@arg input_node: --("input-node") +takes_value
            "Override input nodes names (auto-detects otherwise).")

        (@arg output_node: --("output-node") +takes_value
            "Override output nodes name (auto-detects otherwise).")

        (@arg skip_analyse: --("skip-analyse") "Skip analyse after model build")
        (@arg optimize: -O --optimize "Optimize after model load")
        (@arg pulse: --pulse +takes_value "Translate to pulse network")

        (@arg verbosity: -v ... "Sets the level of verbosity.")
    );

    let compare = clap::SubCommand::with_name("compare")
        .help("Compares the output of tract and tensorflow on randomly generated input.");
    app = app.subcommand(output_options(compare));

    let dump = clap::SubCommand::with_name("dump")
        .help("Dumps the Tensorflow graph in human readable form.")
        .arg(
            Arg::with_name("assert-output")
                .takes_value(true)
                .long("assert-output")
                .help("Fact to check the ouput tensor against (@filename, or 3x4xf32)"),
        );
    app = app.subcommand(output_options(dump));

    let draw = clap::SubCommand::with_name("draw");
    app = app.subcommand(output_options(draw));

    let profile = clap::SubCommand::with_name("profile")
        .help("Benchmarks tract on randomly generated input.")
        .arg(
            Arg::with_name("bench")
                .long("bench")
                .help("Run as an overall bench"),
        ).arg(
            Arg::with_name("max_iters")
                .takes_value(true)
                .long("max-iters")
                .short("n")
                .help("Sets the maximum number of iterations for each node [default: 100_000]."),
        ).arg(
            Arg::with_name("max-time")
                .takes_value(true)
                .long("max-time")
                .help("Sets the maximum execution time for each node (in ms) [default: 5000]."),
        ).arg(
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

    if ::std::env::var("RUST_LOG").is_err() {
        let level = match matches.occurrences_of("verbosity") {
            0 => "cli=warn,tract=warn",
            1 => "cli=info,tract=info",
            2 => "cli=debug,tract=debug",
            _ => "cli=trace,tract=trace",
        };
        ::std::env::set_var("RUST_LOG", level);
    }

    let env = env_logger::Env::default().filter_or(env_logger::DEFAULT_FILTER_ENV, "warn");

    env_logger::Builder::from_env(env)
        .default_format_timestamp_nanos(true)
        .init();

    if let Err(e) = handle(matches) {
        error!("{}", e.to_string());
        process::exit(1)
    }
}

fn output_options<'a, 'b>(command: clap::App<'a, 'b>) -> clap::App<'a, 'b> {
    use clap::*;
    command
        .arg(
            Arg::with_name("quiet")
                .short("q")
                .long("quiet")
                .help("don't dump"),
        ).arg(
            Arg::with_name("debug-op")
                .long("debug-op")
                .help("show debug dump for each op"),
        ).arg(
            Arg::with_name("node_id")
                .long("node-id")
                .takes_value(true)
                .help("Select a node to dump"),
        ).arg(
            Arg::with_name("successors")
                .long("successors")
                .takes_value(true)
                .help("Show successors of node"),
        ).arg(
            Arg::with_name("op_name")
                .long("op-name")
                .takes_value(true)
                .help("Select one op to dump"),
        ).arg(
            Arg::with_name("node_name")
                .long("node-name")
                .takes_value(true)
                .help("Select one node to dump"),
        ).arg(
            Arg::with_name("const")
                .long("const")
                .help("also display consts nodes"),
        )
}

#[derive(Debug)]
pub enum SomeGraphDef {
    Tf(GraphDef),
    Onnx(tract_onnx::pb::ModelProto),
}

/// Structure holding the parsed parameters.
pub struct Parameters {
    name: String,
    graph: SomeGraphDef,
    tract_model: tract_core::Model,
    pulse_facts: Option<(PulsedTensorFact, PulsedTensorFact)>,

    #[cfg(feature = "conform")]
    tf_model: Option<tract_tensorflow::conform::tf::Tensorflow>,

    #[cfg(not(feature = "conform"))]
    #[allow(dead_code)]
    tf_model: (),

    inputs: Option<Vec<Option<tract_core::Tensor>>>,
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
        let (graph, mut tract_model) = if format == "onnx" {
            let graph = tract_onnx::model::model_proto_for_path(&name)?;
            let tract = tract_onnx::for_path(&name)?;
            (SomeGraphDef::Onnx(graph), tract)
        } else {
            let graph = tract_tensorflow::model::graphdef_for_path(&name)?;
            let tract_model = tract_tensorflow::for_path(&name)?;
            (SomeGraphDef::Tf(graph), tract_model)
        };

        info!("Model {:?} loaded", name);

        #[cfg(feature = "conform")]
        let tf_model = if format == "tf" {
            Some(tract_tensorflow::conform::tf::for_path(&name)?)
        } else {
            None
        };

        #[cfg(not(feature = "conform"))]
        let tf_model = ();

        if let Some(inputs) = matches.values_of("input_node") {
            tract_model.set_inputs(inputs)?;
        };

        if let Some(outputs) = matches.values_of("output_node") {
            tract_model.set_outputs(outputs)?;
        };

        let inputs = if let Some(inputs) = matches.values_of("input") {
            let mut vs = vec![];
            for (ix, v) in inputs.enumerate() {
                let t = tensor::for_string(v)?;
                // obliterate value in input (the analyser/optimizer would fold
                // the graph)
                let mut fact = TensorFact {
                    value: Default::default(),
                    ..t
                };
                if let Some(axis) = matches.value_of("stream_axis") {
                    let axis = axis.parse::<usize>().unwrap();
                    let shape = ShapeFact::closed(
                        fact.shape
                            .dims()
                            .enumerate()
                            .map(|(ix, &d)| {
                                if ix == axis {
                                    GenericFact::Only(::tract_core::TDim::s())
                                } else {
                                    d
                                }
                            }).collect()
                    );
                    fact.shape = shape;
                }
                vs.push(t.value.concretize());
                let outlet = tract_model.inputs()?[ix];
                tract_model.set_fact(outlet, fact)?;
            }
            Some(vs)
        } else {
            None
        };

        if !matches.is_present("skip_analyse") {
            info!("Running analyse");
            tract_model.analyse()?;
        } else {
            info!("Skipping analyse");
        }

        let pulse: Option<usize> = matches.value_of("pulse").map(|s| s.parse()).inside_out()?;

        if matches.is_present("optimize") || pulse.is_some() {
            info!("Optimize");
            if format == "tf" {
                tract_model = ::tract_tensorflow::model::optimize(tract_model)?;
            }
            tract_model = tract_model.into_optimized()?;
        }

        let pulse_facts = if let Some(pulse) = pulse {
            info!("Pulsify {}", pulse);
            let (model, ifact, ofact) = ::tract_core::pulse::pulsify(&tract_model, pulse)?;
            if matches.is_present("optimize") {
                info!("Optimize pulsing network");
                tract_model = model.into_optimized()?;
            } else {
                tract_model = model;
            };
            Some((ifact, ofact))
        } else {
            None
        };

        Ok(Parameters {
            name: name.to_string(),
            graph,
            tract_model,
            tf_model,
            pulse_facts,
            inputs,
        })
    }
}

pub enum ProfilingMode {
    Regular { max_iters: u64, max_time: u64 },
    RegularBenching { max_iters: u64, max_time: u64 },
}

impl ProfilingMode {
    pub fn from_clap(matches: &clap::ArgMatches) -> CliResult<ProfilingMode> {
        let max_iters = matches
            .value_of("max_iters")
            .map(u64::from_str)
            .inside_out()?
            .unwrap_or(DEFAULT_MAX_ITERS);
        let max_time = matches
            .value_of("max-time")
            .map(u64::from_str)
            .inside_out()?
            .unwrap_or(DEFAULT_MAX_TIME);
        let mode = if matches.is_present("bench") {
            ProfilingMode::RegularBenching {
                max_iters,
                max_time,
            }
        } else {
            ProfilingMode::Regular {
                max_iters,
                max_time,
            }
        };
        Ok(mode)
    }
}

pub fn display_options_from_clap(matches: &clap::ArgMatches) -> CliResult<DisplayOptions> {
    Ok(DisplayOptions {
        konst: matches.is_present("const"),
        quiet: matches.is_present("quiet"),
        debug_op: matches.is_present("debug-op"),
        node_ids: matches
            .values_of("node_id")
            .map(|id| id.map(|id| id.parse().unwrap()).collect()),
        node_name: matches.value_of("node_name").map(String::from),
        op_name: matches.value_of("op_name").map(String::from),
        successors: matches.value_of("successors").map(|id| id.parse().unwrap()),
    })
}

/// Handles the command-line input.
fn handle(matches: clap::ArgMatches) -> CliResult<()> {
    let params = Parameters::from_clap(&matches)?;

    match matches.subcommand() {
        ("compare", Some(m)) => compare::handle(params, display_options_from_clap(m)?),

        ("run", Some(m)) => {
            let assert_outputs: Option<Vec<TensorFact>> = m
                .values_of("assert-output")
                .map(|vs| vs.map(|v| tensor::for_string(v).unwrap()).collect());
            run::handle(params, assert_outputs)
        }

        /*
        ("optimize-check", Some(m)) => {
            optimize_check::handle(params, display_options_from_clap(m)?)
        }

        ("stream-check", Some(m)) => stream_check::handle(params, display_options_from_clap(m)?),
        */
        ("draw", _) => ::draw::render(&params.tract_model),

        ("dump", Some(m)) => {
            let assert_outputs: Option<Vec<TensorFact>> = m
                .values_of("assert-output")
                .map(|vs| vs.map(|v| tensor::for_string(v).unwrap()).collect());
            dump::handle(params, assert_outputs, display_options_from_clap(m)?)
        }

        ("profile", Some(m)) => profile::handle(
            params,
            ProfilingMode::from_clap(&m)?,
            display_options_from_clap(m)?,
        ),

        (s, _) => bail!("Unknown subcommand {}.", s),
    }
}
