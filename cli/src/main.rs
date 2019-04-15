extern crate ansi_term;
extern crate box_drawing;
extern crate clap;
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
#[cfg(feature = "onnx")]
extern crate tract_onnx;
#[cfg(feature = "tf")]
extern crate tract_tensorflow;

use itertools::Itertools;
use std::process;
use std::str::FromStr;

#[cfg(feature = "tf")]
use crate::tfpb::graph::GraphDef;
use insideout::InsideOut;
use tract_core::internal::*;
use tract_core::model::{InferenceModel, NormalizedModel, TypedModel};
#[cfg(feature = "tf")]
use tract_tensorflow::tfpb;

use crate::display_graph::DisplayOptions;
use crate::errors::*;

#[cfg(feature = "conform")]
mod compare;
mod cost;
mod display_graph;
mod draw;
mod dump;
mod errors;
mod format;
mod optimize_check;
mod profile;
mod run;
mod rusage;
mod stream_check;
mod tensor;
mod utils;

/// The default maximum for iterations and time.
const DEFAULT_MAX_ITERS: u64 = 100_000;
const DEFAULT_MAX_TIME: u64 = 5000;

/// Entrypoint for the command-line interface.
fn main() {
    use clap::*;
    let mut app = clap_app!(("tract") =>
        (version: "1.0")
        (author: "Romain Liautaud <romain.liautaud@snips.ai>")
        (author: "Mathieu Poumeyrol <mathieu.poumeyrol@snips.ai>")
        (about: "Tract command line interface")

        (@setting UnifiedHelpMessage)
        (@setting DeriveDisplayOrder)

        (@arg model: +takes_value "Sets the model to use")

        (@arg format: +takes_value
            "Hint the model format ('onnx' or 'tf') instead of guess from extension.")

        (@arg input: -i --input +takes_value +multiple number_of_values(1)
            "Set input value (@file or 3x4xi32)")

        (@arg stream_axis: -s --("stream-axis") +takes_value
            "Set Axis number to stream upon (first is 0)")

        (@arg input_node: --("input-node") +takes_value +multiple number_of_values(1)
            "Override input nodes names (auto-detects otherwise).")

        (@arg output_node: --("output-node") +takes_value
            "Override output nodes name (auto-detects otherwise).")

        (@arg skip_analyse: --("skip-analyse") "Skip analyse after model build")
        (@arg skip_type: --("skip-type") "Analyse as much as possible, but do not enforce full typing")

        (@arg declutter: --declutter "Declutter model after load")
        (@arg optimize: -O --optimize "Optimize after model load")
        (@arg pulse: --pulse +takes_value "Translate to pulse network")

        (@arg verbosity: -v ... "Sets the level of verbosity.")

        (@arg machine_friendly: --("machine-friendly") "Machine friendly output")

        (@arg list_ops: --("list-ops") "List all known operators")
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
        )
        .arg(
            Arg::with_name("assert-output-fact")
                .takes_value(true)
                .long("assert-output-fact")
                .help("Infered shape and datum type must match exactly this"),
        );
    app = app.subcommand(output_options(dump));

    let draw = clap::SubCommand::with_name("draw");
    app = app.subcommand(output_options(draw));

    let profile =
        clap::SubCommand::with_name("profile")
            .help("Benchmarks tract on randomly generated input.")
            .arg(Arg::with_name("bench").long("bench").help("Run as an overall bench"))
            .arg(
                Arg::with_name("max_iters").takes_value(true).long("max-iters").short("n").help(
                    "Sets the maximum number of iterations for each node [default: 100_000].",
                ),
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
        )
        .arg(
            Arg::with_name("assert-output-fact")
                .takes_value(true)
                .long("assert-output-fact")
                .help("Infered shape and datum type must match exactly this"),
        );
    app = app.subcommand(output_options(run));

    let cost = clap::SubCommand::with_name("cost").help("Compute a cost on (some) operations.");
    app = app.subcommand(output_options(cost));

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

    env_logger::Builder::from_env(env).default_format_timestamp_nanos(true).init();

    if let Err(e) = handle(matches) {
        error!("{}", e.to_string());
        process::exit(1)
    }
}

fn output_options<'a, 'b>(command: clap::App<'a, 'b>) -> clap::App<'a, 'b> {
    use clap::*;
    command
        .arg(Arg::with_name("quiet").short("q").long("quiet").help("don't dump"))
        .arg(Arg::with_name("debug-op").long("debug-op").help("show debug dump for each op"))
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
        .arg(Arg::with_name("const").long("const").help("also display consts nodes"))
}

#[derive(Debug)]
pub enum SomeGraphDef {
    #[cfg(feature = "tf")]
    Tf(GraphDef),
    #[cfg(feature = "onnx")]
    Onnx(tract_onnx::pb::ModelProto),
    _NoGraph, // here to avoid "irrefutable patterns" in match statements
}

#[derive(Debug)]
pub enum SomeModel {
    Inference(InferenceModel),
    Typed(TypedModel),
    Normalized(NormalizedModel),
    Pulsed(NormalizedModel, PulsedModel),
}

/// Structure holding the parsed parameters.
pub struct Parameters {
    name: String,
    graph: SomeGraphDef,
    unoptimized_model: Option<TypedModel>,
    tract_model: SomeModel,

    #[cfg(feature = "conform")]
    tf_model: Option<tract_tensorflow::conform::tf::Tensorflow>,

    #[cfg(not(feature = "conform"))]
    #[allow(dead_code)]
    tf_model: (),

    inputs: Option<Vec<Option<tract_core::internal::SharedTensor>>>,

    assertions: Option<Assertions>,

    machine_friendly: bool,
}

impl Parameters {
    /// Parses the command-line arguments.
    pub fn from_clap(matches: &clap::ArgMatches) -> CliResult<Parameters> {
        let name = matches.value_of("model").unwrap();
        let format = matches.value_of("format").unwrap_or(if name.ends_with(".onnx") {
            "onnx"
        } else {
            "tf"
        });
        let (graph, mut raw_model) = if format == "onnx" {
            #[cfg(not(feature = "onnx"))]
            {
                panic!("Tract compiled without onnx feature");
            }
            #[cfg(feature = "onnx")]
            {
                let onnx = tract_onnx::onnx();
                let graph = onnx.proto_model_for_path(&name)?;
                let tract = onnx.model_for_proto_model(&graph)?;
                (SomeGraphDef::Onnx(graph), tract)
            }
        } else {
            #[cfg(not(feature = "tf"))]
            {
                panic!("Tract compiled without tensorflow feature");
            }
            #[cfg(feature = "tf")]
            {
                let tf = tract_tensorflow::tensorflow();
                let graph = tf.proto_model_for_path(&name)?;
                let tract = tf.model_for_proto_model(&graph)?;
                (SomeGraphDef::Tf(graph), tract)
            }
        };

        info!("Model {:?} loaded", name);

        #[cfg(feature = "conform")]
        let tf_model = if format == "tf" {
            info!("Tensorflow version: {}", tract_tensorflow::conform::tf::version());
            Some(tract_tensorflow::conform::tf::for_path(&name)?)
        } else {
            None
        };

        #[cfg(not(feature = "conform"))]
        let tf_model = ();

        if let Some(inputs) = matches.values_of("input_node") {
            raw_model.set_input_names(inputs)?;
        };

        if let Some(outputs) = matches.values_of("output_node") {
            raw_model.set_output_names(outputs)?;
        };

        let machine_friendly = matches.is_present("machine_friendly");

        let inputs = if let Some(inputs) = matches.values_of("input") {
            let mut vs = vec![];
            for (ix, v) in inputs.enumerate() {
                let t = tensor::for_string(v)?;
                let outlet = raw_model.input_outlets()?[ix];
                vs.push(t.value.concretize());
                raw_model.set_outlet_fact(outlet, t)?;
            }
            Some(vs)
        } else {
            None
        };

        let pulse: Option<usize> = matches.value_of("pulse").map(|s| s.parse()).inside_out()?;

        //        println!("{:?}", raw_model);

        let mut unoptimized_model = None;
        let mut tract_model = if !matches.is_present("skip_analyse") {
            info!("Running analyse");
            if let Err(e) = raw_model.analyse(true) {
                // do not stop on mere analyse error
                error!("{}", e);
            }
            if matches.is_present("skip_type") {
                SomeModel::Inference(raw_model)
            } else {
                let typed = raw_model.into_typed()?;
                unoptimized_model = Some(typed.clone());
                SomeModel::Typed(typed)
            }
        } else {
            info!("Skipping analyse");
            SomeModel::Inference(raw_model)
        };

        if matches.is_present("optimize")
            || matches.is_present("declutter")
            || pulse.is_some()
            || matches.subcommand().0 == "optimize-check"
        {
            if let SomeModel::Typed(typed) = tract_model {
                info!("Declutter");
                tract_model = SomeModel::Typed(typed.declutter()?);
            } else {
                bail!("Can not run optimize without analyse")
            }
        }

        if let (Some(pulse), &SomeModel::Typed(ref model)) = (pulse, &tract_model) {
            info!("Convert to normalized net");
            let normalized = model.clone().into_normalized()?;
            info!("Pulsify {}", pulse);
            let pulsed = ::tract_core::pulse::PulsedModel::new(&normalized, pulse)?;
            tract_model = SomeModel::Pulsed(normalized, pulsed);
        };

        if matches.is_present("optimize") {
            if let SomeModel::Typed(typed) = tract_model {
                tract_model = SomeModel::Typed(typed.codegen()?);
            } else if let SomeModel::Pulsed(_, pulsed) = tract_model {
                tract_model = SomeModel::Typed(pulsed.into_typed()?.codegen()?);
            }
        }

        info!("Model ready");

        Ok(Parameters {
            name: name.to_string(),
            graph,
            unoptimized_model,
            tract_model,
            tf_model,
            inputs,
            assertions: None,
            machine_friendly,
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
            ProfilingMode::RegularBenching { max_iters, max_time }
        } else {
            ProfilingMode::Regular { max_iters, max_time }
        };
        Ok(mode)
    }
}

pub fn display_options_from_clap(matches: &clap::ArgMatches) -> CliResult<DisplayOptions> {
    Ok(DisplayOptions {
        konst: matches.is_present("const"),
        quiet: matches.is_present("quiet"),
        debug_op: matches.is_present("debug-op"),
        node_ids: matches.values_of("node_id").map(|id| id.map(|id| id.parse().unwrap()).collect()),
        node_name: matches.value_of("node_name").map(String::from),
        op_name: matches.value_of("op_name").map(String::from),
        successors: matches.value_of("successors").map(|id| id.parse().unwrap()),
    })
}

pub struct Assertions {
    assert_outputs: Option<Vec<TensorFact>>,
    assert_output_facts: Option<Vec<TensorFact>>,
}

impl Assertions {
    fn from_clap(sub_matches: &clap::ArgMatches) -> CliResult<Assertions> {
        let assert_outputs: Option<Vec<TensorFact>> = sub_matches
            .values_of("assert-output")
            .map(|vs| vs.map(|v| tensor::for_string(v).unwrap()).collect());
        let assert_output_facts: Option<Vec<TensorFact>> = sub_matches
            .values_of("assert-output-fact")
            .map(|vs| vs.map(|v| tensor::for_string(v).unwrap()).collect());
        Ok(Assertions { assert_outputs, assert_output_facts })
    }
}

/// Handles the command-line input.
fn handle(matches: clap::ArgMatches) -> CliResult<()> {
    if matches.is_present("list_ops") {
        #[cfg(feature = "onnx")]
        {
            let onnx = tract_onnx::onnx();
            let names = onnx.op_register.names().sorted().into_iter().join(", ");
            println!("Onnx:\n");
            println!("{}", names);
            println!("\n");
        }
        #[cfg(feature = "tf")]
        {
            let tf = tract_tensorflow::tensorflow();
            let names = tf.op_register.names().sorted().into_iter().join(", ");
            println!("Tensorflow:\n");
            println!("{}", names);
            println!("\n");
        }
        return Ok(());
    }

    let mut params = Parameters::from_clap(&matches)?;

    match matches.subcommand() {
        #[cfg(feature = "conform")]
        ("compare", Some(m)) => compare::handle(params, display_options_from_clap(m)?),
        #[cfg(not(feature = "conform"))]
        ("compare", _) => bail!("Need conform feature to be able to run comparison"),

        ("run", Some(m)) => {
            params.assertions = Some(Assertions::from_clap(m)?);
            run::handle(params)
        }

        ("optimize-check", Some(m)) => {
            optimize_check::handle(params, display_options_from_clap(m)?)
        }

        ("stream-check", Some(m)) => stream_check::handle(params, display_options_from_clap(m)?),

        ("cost", Some(m)) => crate::cost::handle(params, display_options_from_clap(m)?),

        ("draw", Some(m)) => {
            crate::draw::render(&params.tract_model, display_options_from_clap(m)?)
        }

        ("dump", Some(m)) => {
            params.assertions = Some(Assertions::from_clap(m)?);
            dump::handle(params, display_options_from_clap(m)?)
        }

        ("profile", Some(m)) => {
            profile::handle(params, ProfilingMode::from_clap(&m)?, display_options_from_clap(m)?)
        }

        (s, _) => bail!("Unknown subcommand {}.", s),
    }
}
