extern crate ansi_term;
extern crate box_drawing;
extern crate clap;
#[macro_use]
extern crate error_chain;
#[macro_use]
extern crate log;
extern crate atty;
extern crate env_logger;
extern crate pbr;
#[macro_use]
extern crate serde_derive;
#[macro_use]
extern crate tract_core;
#[cfg(feature = "onnx")]
extern crate tract_onnx;
#[cfg(feature = "tf")]
extern crate tract_tensorflow;

#[macro_use]
mod macros;

use std::process;
use std::str::FromStr;
#[allow(unused_imports)]
use tract_itertools::Itertools;

use tract_core::internal::*;
use tract_core::model::{NormalizedModel, TypedModel};
use tract_hir::internal::*;
#[cfg(feature = "tf")]
use tract_tensorflow::tfpb::tensorflow::GraphDef;

use crate::display_params::DisplayParams;
use crate::errors::*;

use readings_probe::*;

mod annotations;
mod bench;
mod compare;
mod cost;
mod display_params;
mod draw;
mod dump;
mod errors;
mod export;
mod optimize_check;
mod profile;
mod run;
// mod rusage;
mod stream_check;
mod tensor;
mod terminal;
mod utils;

readings_probe::instrumented_allocator!();

fn info_usage(stage: &str, probe: Option<&Probe>) {
    if let Some(mon) = probe {
        let _ = mon.log_event(stage);
    }
    if log::log_enabled!(log::Level::Info) {
        let usage = readings_probe::get_os_readings().unwrap();
        info!(
            "Resource usage {}: vsz:{} rsz:{} rszmax:{}",
            stage, usage.virtual_size, usage.resident_size, usage.resident_size_max
        );
    }
}

/// Entrypoint for the command-line interface.
fn main() {
    use clap::*;
    let mut app = clap_app!(("tract") =>
    (author: "Romain Liautaud <romain.liautaud@snips.ai>")
    (author: "Mathieu Poumeyrol <mathieu.poumeyrol@snips.ai>")
    (version: crate_version!())
    (about: "Tract command line interface")

    (@setting DeriveDisplayOrder)
    (@setting AllowLeadingHyphen)

    (@arg readings: --readings "Start readings instrumentation")
    (@arg readings_heartbeat: --("readings-heartbeat") +takes_value
     default_value("5") "Set heartbeat (ms)")

    (@arg model: +takes_value "Sets the model to use")

    (@arg format: -f +takes_value
     "Hint the model format ('kaldi', 'onnx' or 'tf') instead of guess from extension.")

    (@arg input: -i --input +takes_value +multiple number_of_values(1)
     "Set input shape and type (@file.pb or @file.npz:thing.npy or 3x4xi32).")

    (@arg const_input: --("const-input") +takes_value +multiple number_of_values(1)
     "Treat input as a Const (by name), retaining its value.")

    (@arg input_bundle: --("input-bundle") +takes_value +multiple number_of_values(1)
     "Path to an input container (.npz)")

    (@arg stream_axis: -s --("stream-axis") +takes_value
     "Set Axis number to stream upon (first is 0)")

    (@arg kaldi_adjust_final_offset: --("kaldi-adjust-final-offset") +takes_value
     "Adjust value of final offset in network (for reproducibility)")

    (@arg kaldi_downsample: --("kaldi-downsample") +takes_value
     "Add a subsampling to output on axis 0")

    (@arg kaldi_left_context: --("kaldi-left-context") +takes_value
     "Add lines of left context to input (dupping first time frame)")

    (@arg kaldi_right_context: --("kaldi-right-context") +takes_value
     "Add lines of right context to input (dupping last time frame)")

    (@arg input_node: --("input-node") +takes_value +multiple number_of_values(1)
     "Override input nodes names (auto-detects otherwise).")

    (@arg tf_initializer_output_node: --("tf-initializer-output-node") +takes_value +multiple number_of_values(1)
     "Set an initializer node")

    (@arg output_node: --("output-node") +takes_value +multiple number_of_values(1)
     "Override output nodes name (auto-detects otherwise).")

    (@arg override_fact: --("override-fact") +takes_value +multiple number_of_values(1)
     "Override a fact.")

    (@arg analyse_fail_fast: --("analyse-fail-fast") "Stop analyse at first error.")
    (@arg recursive: --recursive "Apply to sub graphes")

    (@arg proto: --proto "Keep proto model around after parse")
    (@arg determinize: --determinize "Enforce a seed in random operator")

    (@arg partial: --partial "Before analyse, eliminate dead branches")

    (@arg pass: --pass +takes_value
     possible_values(&["load", "analyse", "incorporate", "type", "declutter",
                     "pulse-normalized", "pulse", "pulse-to-type", "pulse-declutter",
                     "optimize"])
     "Pass to stop preprocessing after.")

    (@arg optimize: -O --optimize "Optimize before running")
    (@arg pulse: --pulse +takes_value "Translate to pulse network")

    (@arg verbosity: -v ... "Sets the level of verbosity.")

    (@arg machine_friendly: --("machine-friendly") "Machine friendly output")

    (@arg list_ops: --("list-ops") "List all known operators")
    );

    let compare = clap::SubCommand::with_name("compare")
        .long_about("Compares the output of tract and tensorflow on randomly generated input.")
        .arg(
            Arg::with_name("cumulative")
                .long("cumulative")
                .takes_value(false)
                .help("Do not reset with reference values at each node"),
        )
        .arg(
            Arg::with_name("resilient")
                .long("resilient")
                .takes_value(false)
                .help("Try nodes one per one to mitigate crashes"),
        );
    app = app.subcommand(output_options(compare));

    let compare_npz = clap::SubCommand::with_name("compare-npz")
        .long_about("Compares the output of tract to a refrence npz file.")
        .arg(
            Arg::with_name("cumulative")
                .long("cumulative")
                .takes_value(false)
                .help("Do not reset with reference values at each node"),
        )
        .arg(Arg::with_name("npz").takes_value(true).required(true).help("Npz filename"));
    app = app.subcommand(output_options(compare_npz));

    let compare_pbdir = clap::SubCommand::with_name("compare-pbdir")
        .long_about(
            "Compares the output of tract to a refrence directory of onnx protobufs tensors files.",
        )
        .arg(
            Arg::with_name("cumulative")
                .long("cumulative")
                .takes_value(false)
                .help("Do not reset with reference values at each node"),
        )
        .arg(Arg::with_name("pbdir").takes_value(true).required(true).help("protobuf dir"));
    app = app.subcommand(output_options(compare_pbdir));

    let bench = clap::SubCommand::with_name("bench")
        .long_about("Benchmarks tract on randomly generated input.");
    let bench = output_options(bench);
    let bench = benchlimits_options(bench);
    app = app.subcommand(bench);

    let dump = clap::SubCommand::with_name("dump")
        .long_about("Dumps the Tensorflow graph in human readable form.")
        .arg(Arg::with_name("cost").long("cost").help("Include const information"))
        .arg(Arg::with_name("profile").long("profile").help("Include results for profile run"))
        .arg(
            Arg::with_name("assert-cost")
            .takes_value(true)
            .long("assert-cost")
            .help("Checks computed against the provided value (form: \"FMA(F32)=2060448 DIV(F32)=24576\")")
            )
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
            )
        .arg(
            Arg::with_name("inner")
            .takes_value(true)
            .number_of_values(1)
            .multiple(true)
            .long("inner")
            .help("Navigate to a sub-model"),
            );
    let dump = output_options(dump);
    let dump = benchlimits_options(dump);
    app = app.subcommand(dump);

    let run = clap::SubCommand::with_name("run")
        .long_about("Run the graph")
        .arg(Arg::with_name("dump").long("dump").help("Show output"))
        .arg(
            Arg::with_name("assert-output-bundle")
                .takes_value(true)
                .long("assert-output-bundle")
                .help("Checks values against these tensor (.npz)"),
        )
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

    let optimize = clap::SubCommand::with_name("optimize").help("Optimize the graph");
    app = app.subcommand(output_options(optimize));

    let optimize_check = clap::SubCommand::with_name("optimize-check")
        .long_about("Compare output of optimized and un-optimized graph");
    app = app.subcommand(output_options(optimize_check));

    let stream_check = clap::SubCommand::with_name("stream-check")
        .long_about("Compare output of streamed and regular exec");
    app = app.subcommand(output_options(stream_check));

    let matches = app.get_matches();

    let probe = if matches.is_present("readings") {
        let file = std::fs::File::create("readings.out").unwrap();
        let mut probe = Probe::new(file).unwrap();
        probe.register_i64("progress").unwrap();
        let heartbeat = matches.value_of("readings_heartbeat").unwrap().parse::<f32>().unwrap();
        probe.spawn_heartbeat(std::time::Duration::from_secs_f32(heartbeat / 1000.0)).unwrap();
        Some(probe)
    } else {
        None
    };

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

    env_logger::Builder::from_env(env).format_timestamp_nanos().init();
    info_usage("init", probe.as_ref());

    if let Err(e) = handle(matches, probe.as_ref()) {
        use error_chain::ChainedError;
        error!("{}", e);
        eprintln!("{}", e.display_chain());
        process::exit(1)
    }

    info_usage("done", probe.as_ref());
}

fn benchlimits_options<'a, 'b>(command: clap::App<'a, 'b>) -> clap::App<'a, 'b> {
    use clap::*;
    command
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
}

fn output_options<'a, 'b>(command: clap::App<'a, 'b>) -> clap::App<'a, 'b> {
    use clap::*;
    command
        .arg(
            Arg::with_name("natural-order")
                .long("natural-order")
                .help("dump nodes in id order instead of evaluation order"),
        )
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
        .arg(Arg::with_name("info").long("info").help("show op inner information"))
        .arg(Arg::with_name("io-long").long("io-long").help("show full i/o information"))
        .arg(Arg::with_name("io-none").long("io-none").help("hide i/o information"))
        .arg(Arg::with_name("json").long("json").help("dump performance info as json"))
        .arg(Arg::with_name("outlet-labels").long("outlet-labels").help("display outlet labels"))
        .arg(
            Arg::with_name("invariants")
                .takes_value(false)
                .long("invariants")
                .help("Display operators invariants"),
        )
}

#[derive(Debug)]
pub enum SomeGraphDef {
    NoGraphDef,
    #[cfg(feature = "kaldi")]
    Kaldi(tract_kaldi::KaldiProtoModel),
    #[cfg(feature = "onnx")]
    Onnx(tract_onnx::pb::ModelProto, tract_onnx::model::ParseResult),
    #[cfg(feature = "tf")]
    Tf(GraphDef),
}

/// Structure holding the parsed parameters.
pub struct Parameters {
    analyse_error: Option<TractError>,
    graph: SomeGraphDef,
    typed_model: Option<TypedModel>,
    normalized_model: Option<NormalizedModel>,
    tract_model: Box<dyn Model>,

    output_names: Vec<String>,

    #[cfg(feature = "conform")]
    tf_model: Option<tract_tensorflow::conform::tf::Tensorflow>,

    #[cfg(not(feature = "conform"))]
    #[allow(dead_code)]
    tf_model: (),

    input_values: Vec<Option<Arc<Tensor>>>,

    assertions: Option<Assertions>,

    machine_friendly: bool,
}

#[cfg(feature = "tf")]
type TfExt = tract_tensorflow::model::TfModelExtensions;
#[cfg(not(feature = "tf"))]
type TfExt = ();

impl Parameters {
    #[allow(unused_variables)]
    /// Parses the command-line arguments.
    pub fn from_clap(matches: &clap::ArgMatches, probe: Option<&Probe>) -> CliResult<Parameters> {
        let name = matches.value_of("model").ok_or("Model argument required")?;
        let format = matches.value_of("format").unwrap_or(if name.ends_with(".onnx") {
            "onnx"
        } else {
            "tf"
        });

        let need_graph =
            matches.is_present("proto") || matches.subcommand_name() == Some("compare-pbdir");

        let (mut graph, mut raw_model, tf_model_extensions) = match format {
            #[cfg(feature = "kaldi")]
            "kaldi" => {
                let kaldi = tract_kaldi::kaldi();
                info_usage("loaded framework (kaldi)", probe);
                let mut graph = kaldi.proto_model_for_path(&name)?;
                info_usage("proto model loaded", probe);
                if let Some(i) = matches.value_of("kaldi_adjust_final_offset") {
                    graph.adjust_final_offset = i.parse()?;
                }
                let parsed = kaldi.model_for_proto_model(&graph)?;
                if need_graph {
                    (SomeGraphDef::Kaldi(graph), parsed, Option::<TfExt>::None)
                } else {
                    (SomeGraphDef::NoGraphDef, parsed, Option::<TfExt>::None)
                }
            }
            #[cfg(feature = "onnx")]
            "onnx" => {
                let onnx = tract_onnx::onnx();
                info_usage("loaded framework (onnx)", probe);
                let graph = onnx.proto_model_for_path(&name)?;
                info_usage("proto model loaded", probe);
                let parsed = onnx.parse(&graph)?;
                if need_graph {
                    (SomeGraphDef::Onnx(graph, parsed.clone()), parsed.model, Option::<TfExt>::None)
                } else {
                    (SomeGraphDef::NoGraphDef, parsed.model, Option::<TfExt>::None)
                }
            }
            #[cfg(feature = "tf")]
            "tf" => {
                let tf = tract_tensorflow::tensorflow();
                info_usage("loaded framework (tf)", probe);
                let mut graph = tf.proto_model_for_path(&name)?;
                info_usage("proto model loaded", probe);
                if matches.is_present("determinize") {
                    tract_tensorflow::Tensorflow::determinize(&mut graph)?;
                }
                let mut model_and_ext = tf.parse_graph(&graph)?;
                model_and_ext.1.initializing_nodes = matches
                    .values_of("tf_initializer_output_node")
                    .map(|values| {
                        values
                            .map(|name| model_and_ext.0.node_id_by_name(name))
                            .collect::<TractResult<Vec<usize>>>()
                    })
                    .transpose()?
                    .unwrap_or(vec![]);
                if need_graph {
                    (SomeGraphDef::Tf(graph), model_and_ext.0, Some(model_and_ext.1))
                } else {
                    (SomeGraphDef::NoGraphDef, model_and_ext.0, Some(model_and_ext.1))
                }
            }
            _ => bail!(
                "Format {} not supported. You may need to recompile tract with the right features.",
                format
            ),
        };

        info!("Model {:?} loaded", name);
        info_usage("model loaded", probe);

        #[cfg(feature = "conform")]
        let tf_model = if format == "tf" {
            info!("Tensorflow version: {}", tract_tensorflow::conform::tf::version());
            if matches.is_present("determinize") {
                if let SomeGraphDef::Tf(ref graph) = graph {
                    let graph = graph.write_to_bytes().unwrap();
                    Some(tract_tensorflow::conform::tf::for_slice(&graph)?)
                } else {
                    unreachable!()
                }
            } else {
                Some(tract_tensorflow::conform::tf::for_path(&name)?)
            }
        } else {
            None
        };

        if !matches.is_present("proto") && matches.subcommand_name() != Some("compare-pbdir") {
            graph = SomeGraphDef::NoGraphDef;
        }

        #[cfg(not(feature = "conform"))]
        let tf_model = ();

        if let Some(inputs) = matches.values_of("input") {
            let names = inputs
                .map(|t| Ok(tensor::for_string(t)?.0))
                .collect::<CliResult<Vec<Option<String>>>>()?;
            if names.iter().all(|s| s.is_some() && s.as_ref().unwrap().len() > 0) {
                let names: Vec<String> = names.into_iter().map(|s| s.unwrap()).collect();
                raw_model.set_input_names(names)?;
            }
        }

        if let Some(inputs) = matches.values_of("input_node") {
            raw_model.set_input_names(inputs)?;
        };

        if let Some(outputs) = matches.values_of("output_node") {
            raw_model.set_output_names(outputs)?;
        };

        if let Some(override_facts) = matches.values_of("override_fact") {
            for fact in override_facts {
                let (name, fact) = tensor::for_string(fact)?;
                let node = raw_model.node_by_name(name.unwrap())?.id;
                raw_model.set_outlet_fact(OutletId::new(node, 0), fact)?;
            }
        };

        let output_names = raw_model
            .output_outlets()?
            .iter()
            .map(|o| raw_model.node(o.node).name.to_string())
            .collect();

        if let Some(sub) = matches.value_of("kaldi_downsample") {
            let period = sub.parse::<isize>()?;
            if period != 1 {
                let mut outputs = raw_model.output_outlets()?.to_vec();
                let output_name = raw_model.node(outputs[0].node).name.clone();
                raw_model.node_mut(outputs[0].node).name = format!("{}-old", output_name);
                let id = raw_model.add_node(
                    output_name,
                    tract_core::ops::Downsample::new(0, period as _, 0),
                    tvec!(InferenceFact::default()),
                )?;
                raw_model.add_edge(outputs[0], InletId::new(id, 0))?;
                outputs[0].node = id;
                raw_model.set_output_outlets(&*outputs)?;
            }
        }

        if matches.value_of("kaldi_left_context").is_some()
            || matches.value_of("kaldi_right_context").is_some()
        {
            let left = matches.value_of("kaldi_left_context").unwrap_or("0").parse()?;
            let right = matches.value_of("kaldi_right_context").unwrap_or("0").parse()?;
            let op = tract_core::ops::array::Pad::new(
                vec![(left, right), (0, 0)],
                tract_core::ops::array::PadMode::Edge,
            );
            let mut patch = InferenceModelPatch::default();
            for input in raw_model.input_outlets()? {
                let tap = patch.tap_model(&raw_model, *input)?;
                let pad = patch.wire_node(
                    format!("{}-pad", raw_model.node(input.node).name),
                    op.clone(),
                    &[tap],
                )?[0];
                patch.shunt_outside(&raw_model, *input, pad)?;
            }
            patch.apply(&mut raw_model)?;
        }

        let machine_friendly = matches.is_present("machine_friendly");

        let mut input_values = vec![];

        if let Some(inputs) = matches.values_of("input") {
            for (ix, v) in inputs.enumerate() {
                let (name, mut t) = tensor::for_string(v)?;
                let outlet = if let Some(name) = name.filter(|s| s.len() > 0) {
                    let node = raw_model.node_by_name(&*name)?;
                    OutletId::new(node.id, 0)
                } else {
                    raw_model.input_outlets()?[ix]
                };
                while input_values.len() < ix {
                    input_values.push(None);
                }
                if let Some(t) = t.value.concretize() {
                    input_values.push(Some(t));
                }
                for input in raw_model.node(outlet.node).inputs.clone() {
                    raw_model.node_mut(input.node).outputs[input.slot]
                        .successors
                        .retain(|s| s.node != outlet.node);
                }
                raw_model.node_mut(outlet.node).inputs.clear();
                raw_model.node_mut(outlet.node).op =
                    Box::new(tract_hir::ops::source::Source::new());
                if let Some(s) = matches.value_of("stream_axis") {
                    t.shape.set_dim(s.parse()?, TDim::s());
                }
                info!("Input #{}: {:?}", ix, t);
                raw_model.set_outlet_fact(outlet, t)?;
            }
        }

        if let Some(bundle) = matches.values_of("input_bundle") {
            for input in bundle {
                let mut npz = ndarray_npy::NpzReader::new(std::fs::File::open(input)?)?;
                for name in npz.names()? {
                    match tensor::for_npz(&mut npz, &*name) {
                        Ok(t) => debug!("{} contains {}: {:?}", input, name, t),
                        Err(r) => warn!("Could not read {} from {} ({})", name, input, r),
                    }
                }
                let input_outlets = raw_model.input_outlets()?.to_vec();
                for (ix, input) in input_outlets.iter().enumerate() {
                    let name = format!("{}.npy", raw_model.node(input.node).name);
                    if let Ok(t) = tensor::for_npz(&mut npz, &name) {
                        let shape = t.shape().to_vec();
                        let mut fact = InferenceFact::dt_shape(t.datum_type(), shape);
                        if let Some(s) = matches.value_of("stream_axis") {
                            fact.shape.set_dim(s.parse()?, TDim::s());
                        }
                        raw_model.set_input_fact(ix, fact)?;
                        while input_values.len() <= ix {
                            input_values.push(None);
                        }
                        input_values[ix] = Some(t.into_arc_tensor());
                    }
                }
            }
        }

        for i in (0..raw_model.inputs.len()).rev() {
            let input = raw_model.inputs[i];
            let const_inputs =
                matches.values_of("const_input").map(|cis| cis.collect()).unwrap_or(vec![]);
            if const_inputs.contains(&raw_model.node_name(input.node)) {
                let t = raw_model.outlet_fact(input.node.into())?.value.concretize().unwrap();
                raw_model.node_mut(input.node).op = Box::new(tract_core::ops::konst::Const::new(t));
                raw_model.inputs.remove(i);
            }
        }

        if matches.is_present("partial") {
            raw_model = raw_model.eliminate_dead_branches()?;
        }

        let pulse: Option<usize> = matches.value_of("pulse").map(|s| s.parse()).transpose()?;
        let mut typed_model = None;
        let normalized_model: Option<NormalizedModel> = None;

        let mut analyse_error = None;

        let tract_model: Box<dyn Model> = {
            let stop_at = matches.value_of("pass").unwrap_or(if matches.is_present("optimize") {
                "optimize"
            } else if pulse.is_some() {
                "pulse-declutter"
            } else {
                "declutter"
            });
            info!("Will stop at {}", stop_at);

            (|| -> CliResult<Box<dyn Model>> {
                info!("Running 'load'");
                if stop_at == "load" {
                    return Ok(Box::new(raw_model) as _);
                }
                info_usage("after load", probe);
                info!("Running 'analyse'");
                let r = raw_model.analyse(!matches.is_present("analyse_fail_fast"));
                if let Err(e) = r {
                    analyse_error = Some(e);
                    return Ok(Box::new(raw_model) as _);
                }
                if stop_at == "analyse" || r.is_err() {
                    return Ok(Box::new(raw_model) as _);
                }
                info_usage("after analyse", probe);
                #[cfg(feature = "tf")]
                {
                    if let Some(ext) = tf_model_extensions {
                        info!("Running 'tf-preproc'");
                        raw_model = ext.preproc(raw_model)?;
                        if stop_at == "tf-preproc" {
                            return Ok(Box::new(raw_model) as _);
                        }
                        info_usage("after tf-preproc", probe);
                    }
                }
                info!("Running 'incorporate'");
                let model = raw_model.incorporate()?;
                if stop_at == "incorporate" {
                    return Ok(Box::new(model) as _);
                }
                info_usage("after incorporate", probe);
                info!("Running 'type'");
                let model = match model.clone().into_typed() {
                    Ok(typed) => {
                        typed_model = Some(typed.clone());
                        typed
                    }
                    Err(e) => {
                        error!("{:?}", e);
                        return Ok(Box::new(model) as _);
                    }
                };
                if stop_at == "type" {
                    return Ok(Box::new(model) as _);
                }
                info_usage("after type", probe);
                info!("Running 'declutter'");
                let mut model = model.declutter()?;
                typed_model = Some(model.clone());
                if stop_at == "declutter" {
                    return Ok(Box::new(model) as _);
                }
                info_usage("after declutter", probe);
                if let Some(pulse) = pulse {
                    info!("Running 'pulse-normalize'");
                    let normalized_model = model.into_normalized()?;
                    if stop_at == "pulse-normalize" {
                        return Ok(Box::new(normalized_model) as _);
                    }
                    info_usage("after pulse-normalize", probe);
                    info!("Running 'pulse' ({})", pulse);
                    let pulsed = ::tract_core::pulse::PulsedModel::new(&normalized_model, pulse)?;
                    if stop_at == "pulse" {
                        return Ok(Box::new(pulsed) as _);
                    }
                    info_usage("after pulse", probe);
                    info!("Running 'pulse-to-type'",);
                    model = pulsed.into_typed()?;
                    if stop_at == "pulse-to-type" {
                        return Ok(Box::new(model) as _);
                    }
                    info_usage("after pulse-to-type", probe);
                    info!("Running 'pulse-declutter'");
                    model = model.declutter()?;
                    if stop_at == "pulse-declutter" {
                        return Ok(Box::new(model) as _);
                    }
                    info_usage("after pulse-declutter", probe);
                }
                info!("Running 'optimize'");
                model = model.codegen()?;
                info_usage("after optimize", probe);
                Ok(Box::new(model) as _)
            })()?
        };

        info!("Model ready");
        info_usage("model ready", probe);

        Ok(Parameters {
            analyse_error,
            graph,
            typed_model,
            normalized_model,
            tract_model,
            tf_model,
            input_values,
            output_names,
            assertions: None,
            machine_friendly,
        })
    }
}

pub struct BenchLimits {
    max_iters: usize,
    max_time: std::time::Duration,
}

impl BenchLimits {
    pub fn from_clap(matches: &clap::ArgMatches) -> CliResult<BenchLimits> {
        let max_iters =
            matches.value_of("max_iters").map(usize::from_str).transpose()?.unwrap_or(100_000);
        let max_time = matches
            .value_of("max-time")
            .map(u64::from_str)
            .transpose()?
            .map(std::time::Duration::from_millis)
            .unwrap_or(std::time::Duration::from_secs(5));
        Ok(BenchLimits { max_iters, max_time })
    }
}

pub fn display_params_from_clap(
    root_matches: &clap::ArgMatches,
    matches: &clap::ArgMatches,
) -> CliResult<DisplayParams> {
    Ok(DisplayParams {
        konst: matches.is_present("const"),
        cost: matches.is_present("cost"),
        profile: matches.is_present("profile"),
        left_column_width: 0,
        invariants: matches.is_present("invariants"),
        quiet: matches.is_present("quiet"),
        natural_order: matches.is_present("natural-order"),
        debug_op: matches.is_present("debug-op"),
        node_ids: matches.values_of("node_id").map(|values| {
            values.map(|id| tvec!((id.parse::<usize>().unwrap(), "".to_string()))).collect()
        }),
        node_name: matches.value_of("node_name").map(String::from),
        op_name: matches.value_of("op_name").map(String::from),
        //        successors: matches.value_of("successors").map(|id| id.parse().unwrap()),
        expect_canonic: root_matches.value_of("pass").unwrap_or("declutter") == "declutter"
            && !root_matches.is_present("optimize"),
        outlet_labels: matches.is_present("outlet-labels"),
        io: if matches.is_present("io-long") {
            display_params::Io::Long
        } else if matches.is_present("io-none") {
            display_params::Io::None
        } else {
            display_params::Io::Short
        },
        info: matches.is_present("info"),
        json: matches.is_present("json"),
    })
}

pub struct Assertions {
    assert_outputs: Option<Vec<Option<Arc<Tensor>>>>,
    assert_output_facts: Option<Vec<InferenceFact>>,
}

impl Assertions {
    fn from_clap(sub_matches: &clap::ArgMatches, output_names: &[String]) -> CliResult<Assertions> {
        let mut assert_outputs: Option<Vec<Option<Arc<Tensor>>>> = sub_matches
            .values_of("assert-output")
            .map(|vs| vs.map(|v| tensor::for_string(v).unwrap().1.value.concretize()).collect());

        if assert_outputs.is_none() {
            if sub_matches.values_of("assert-output-bundle").is_some() {
                let values = output_names
                    .iter()
                    .map(move |name| {
                        let npy_name = format!("{}.npy", name);
                        for output_bundle in sub_matches.values_of("assert-output-bundle").unwrap()
                        {
                            let mut npz =
                                ndarray_npy::NpzReader::new(std::fs::File::open(output_bundle)?)?;
                            if let Ok(t) = tensor::for_npz(&mut npz, &npy_name) {
                                return Ok(Some(t.into_arc_tensor()));
                            }
                        }
                        return Ok(None);
                    })
                    .collect::<CliResult<_>>()?;
                assert_outputs = Some(values)
            }
        }

        let assert_output_facts: Option<Vec<InferenceFact>> = sub_matches
            .values_of("assert-output-fact")
            .map(|vs| vs.map(|v| tensor::for_string(v).unwrap().1).collect());
        Ok(Assertions { assert_outputs, assert_output_facts })
    }
}

/// Handles the command-line input.
fn handle(matches: clap::ArgMatches, probe: Option<&Probe>) -> CliResult<()> {
    if matches.is_present("list_ops") {
        #[cfg(feature = "onnx")]
        {
            let onnx = tract_onnx::onnx();
            let names = onnx.op_register.0.keys().sorted().into_iter().join(", ");
            println!("Onnx:\n");
            println!("{}", names);
            println!("\n");
        }
        #[cfg(feature = "tf")]
        {
            let tf = tract_tensorflow::tensorflow();
            let names = tf.op_register.0.keys().sorted().into_iter().join(", ");
            println!("Tensorflow:\n");
            println!("{}", names);
            println!("\n");
        }
        return Ok(());
    }

    let mut params = Parameters::from_clap(&matches, probe)?;

    let mut need_optimisations = false;

    match matches.subcommand() {
        #[cfg(feature = "conform")]
        ("compare", Some(m)) => compare::handle_tensorflow(
            m.is_present("cumulative"),
            m.is_present("resilient"),
            &mut params,
            display_params_from_clap(&matches, m)?,
        ),
        #[cfg(not(feature = "conform"))]
        ("compare", _) => bail!("Need conform feature to be able to run comparison"),

        ("compare-npz", Some(m)) => compare::handle_npz(
            m.is_present("cumulative"),
            m.value_of("npz").unwrap(),
            &params,
            display_params_from_clap(&matches, m)?,
        ),

        #[cfg(feature = "onnx")]
        ("compare-pbdir", Some(m)) => compare::handle_pbdir(
            m.is_present("cumulative"),
            m.value_of("pbdir").unwrap(),
            &params,
            display_params_from_clap(&matches, m)?,
        ),

        ("run", Some(m)) => {
            params.assertions = Some(Assertions::from_clap(m, &*params.output_names)?);
            run::handle(&params, m.is_present("dump"))
        }

        ("optimize-check", Some(m)) => {
            optimize_check::handle(&params, display_params_from_clap(&matches, m)?)
        }

        ("stream-check", Some(m)) => {
            stream_check::handle(&params, &display_params_from_clap(&matches, m)?)
        }

        ("", None) => dump::handle(
            &params,
            &display_params_from_clap(&matches, &clap::ArgMatches::default())?,
            &clap::ArgMatches::default(),
            &BenchLimits::from_clap(&clap::ArgMatches::default())?,
            vec![],
        ),

        ("dump", Some(m)) => {
            params.assertions = Some(Assertions::from_clap(m, &*params.output_names)?);
            need_optimisations = m.is_present("profile");
            let inner = m
                .values_of("inner")
                .map(|ss| ss.map(|s| s.to_string()).collect())
                .unwrap_or(vec![]);
            dump::handle(
                &params,
                &display_params_from_clap(&matches, m)?,
                m,
                &BenchLimits::from_clap(&m)?,
                inner,
            )
        }

        ("bench", Some(m)) => {
            need_optimisations = true;
            bench::handle(&params, &BenchLimits::from_clap(&m)?, probe)
        }

        (s, _) => bail!("Unknown subcommand {}.", s),
    }?;

    if need_optimisations {
        let style = ansi_term::Style::new().fg(ansi_term::Color::Red).bold();
        if !matches.is_present("optimize") {
            warn!("{}", style.paint("Profiling an un-optimized network. Consider adding -O."));
        }
        if cfg!(debug_assertions) {
            warn!("{}", style.paint("Profiling a debug build of tract!"));
        }
    }

    if let Some(e) = params.analyse_error {
        Err(e)?
    }
    Ok(())
}
