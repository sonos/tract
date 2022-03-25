#[macro_use]
extern crate log;
#[macro_use]
extern crate serde_derive;

#[macro_use]
mod macros;

#[allow(unused_imports)]
use tract_itertools::Itertools;

use tract_core::internal::*;
use tract_hir::internal::*;

use crate::model::Model;

use readings_probe::*;

mod annotations;
mod bench;
mod compare;
mod cost;
mod display_params;
mod draw;
mod dump;
mod errors {}
mod export;
mod model;
mod params;
mod profile;
mod run;
#[cfg(feature = "pulse")]
mod stream_check;
mod tensor;
mod terminal;
mod utils;

use params::*;

type CliResult<T> = tract_core::anyhow::Result<T>;

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

pub const STAGES: &'static [&'static str] = &[
    "load",
    "analyse",
    "incorporate",
    "type",
    "declutter",
    "concretize-stream-dim",
    "concretize-stream-dim-declutter",
    "pulse",
    "pulse-to-type",
    "pulse-declutter",
    "nnef-cycle",
    "nnef-cycle-declutter",
    "before-optimize",
    "optimize",
];

/// Entrypoint for the command-line interface.
fn main() -> tract_core::anyhow::Result<()> {
    use clap::*;
    let mut app = command!()
        .setting(AppSettings::DeriveDisplayOrder)
        .allow_hyphen_values(true)
        .arg(arg!(--readings "Start readings instrumentation"))
        .arg(arg!(--"readings-heartbeat" [MS] "Heartbeat for readings background collector").default_value("5"))
        .arg(arg!(verbose: -v ... "Sets the level of verbosity."))
        .arg(arg!([model] "Sets the model to use"))
        .arg(arg!(-f --format [format]
                  "Hint the model format ('kaldi', 'onnx', 'nnef' or 'tf') instead of guess from extension."))

        .arg(arg!(-i --input [input] ...
                  "Set input shape and type (@file.pb or @file.npz:thing.npy or 3x4xi32)."))

        .arg(arg!(--"const-input" [const_input] ... "Treat input as a Const (by name), retaining its value."))
        .arg(arg!(--"input-bundle" [input_bundle] "Path to an input container (.npz)"))

        .arg(arg!(--"kaldi-adjust-final-offset" [frames] "Adjust value of final offset in network (for reproducibility)"))
        .arg(arg!(--"kaldi-downsample" [frames] "Add a subsampling to output on axis 0"))
        .arg(arg!(--"kaldi-left-context" [frames] "Add lines of left context to input (dupping first time frame)"))
        .arg(arg!(--"kaldi-right-context" [frames] "Add lines of right context to input (dupping last time frame)"))

        .arg(arg!(--"onnx-test-data-set" [data_set] "Use onnx-test data-set as input (expect test_data_set_N dir with input_X.pb, etc. inside)"))

        .arg(arg!(--"input-node" [node] ... "Override input nodes names (auto-detects otherwise)."))
        .arg(arg!(--"output-node" [node] ... "Override output nodes name (auto-detects otherwise)."))

        .arg(arg!(--"tf-initializer-output-node" [node] "Set an initializer node"))

        .arg(arg!(--"override-fact" [fact] "Override a fact."))

        .arg(arg!(--"analyse-fail-fast" "Stop analyse at first error."))
        .arg(arg!(--recursive "Apply to sub graphes"))
        .arg(arg!(--proto "Keep proto model around after parse"))
        .arg(arg!(--determinize "Enforce a seed in random operator"))
        .arg(arg!(--multiturn "Switch run and compare to multiturn, session state preserving"))
        .arg(arg!(--partial "Before analyse, eliminate dead branches"))

        .arg(arg!(--pass [STAGE] "Pass to stop preprocessing after.").possible_values(STAGES))
        .arg(arg!(--"declutter-step" [STEP] "Stop decluttering process after application of patch number N"))
        .arg(arg!(--"optimize-step" [STEP] "Stop optimizing process after application of patch number N"))
        .arg(arg!(--"extract-decluttered-sub" [SUB] "Zoom on a subgraph after decluttering by parent node name"))

        .arg(arg!(--"nnef-cycle" "Perform NNEF dump and reload before optimizing"))

        .arg(arg!(--"nnef-tract-core" "Allow usage of tract-core extension in NNEF dump and load"))
        .arg(arg!(--"nnef-tract-onnx" "Allow usage of tract-onnx extension in NNEF dump and load"))
        .arg(arg!(--"nnef-tract-pulse" "Allow usage of tract-pulse extension in NNEF dump and load"))

        .arg(arg!(-O --optimize "Optimize before running"))
        .arg(arg!(--pulse [PULSE] "Translate to pulse network"))
        .arg(arg!(--"concretize-stream-dim" [VALUE] "Replace streaming dim by a concrete value"))

        .arg(arg!(--"machine-friendly" "Machine friendly output"))

        .arg(arg!(--"list-ops" "List all known operators"));

    let compare = clap::Command::new("compare")
        .long_about("Compares the output of tract and tensorflow on randomly generated input.")
        .arg(
            Arg::new("stage")
                .long("stage")
                .takes_value(true)
                .possible_values(STAGES)
                .help("Loading pipeline stage to compare with"),
        )
        .arg(Arg::new("tf").long("tf").takes_value(false).help("Compare against tensorflow"))
        .arg(Arg::new("twice").long("twice").takes_value(false).help("Run twice and compare"))
        .arg(Arg::new("npz").long("npz").takes_value(true).help("NPZ file to compare against"))
        .arg(
            Arg::new("pbdir")
                .long("pbdir")
                .takes_value(true)
                .help("protobuf directory file to compare against (like ONNX tests)"),
        )
        .group(
            ArgGroup::new("reference")
                .args(&["npz", "pbdir", "stage", "tf", "twice"])
                .required(true),
        )
        .arg(
            Arg::new("cumulative")
                .long("cumulative")
                .takes_value(false)
                .help("Do not reset with reference values at each node"),
        )
        .arg(
            Arg::new("resilient")
                .long("resilient")
                .takes_value(false)
                .help("Try nodes one per one to mitigate crashes"),
        );
    app = app.subcommand(output_options(compare));

    let bench =
        clap::Command::new("bench").long_about("Benchmarks tract on randomly generated input.");
    let bench = output_options(bench);
    let bench = benchlimits_options(bench);
    app = app.subcommand(bench);

    let criterion = clap::Command::new("criterion")
        .long_about("Benchmarks tract on randomly generated input using criterion.");
    app = app.subcommand(criterion);

    app = app.subcommand(dump_subcommand());

    let run = clap::Command::new("run")
        .long_about("Run the graph")
        .arg(Arg::new("dump").long("dump").help("Show output"))
        .arg(Arg::new("steps").long("steps").help("Show all inputs and outputs"))
        .arg(
            Arg::new("set")
                .long("set")
                .takes_value(true)
                .multiple_occurrences(true)
                .number_of_values(1)
                .help("Set a symbol value (--set S=12)"),
        )
        .arg(
            Arg::new("save-steps")
                .long("save-steps")
                .takes_value(true)
                .help("Save intermediary values"),
        )
        .arg(
            Arg::new("assert-sane-floats")
                .long("assert-sane-floats")
                .help("Check float for NaN and infinites at each step"),
        );
    let run = output_options(run);
    let run = assertions_options(run);
    app = app.subcommand(run);

    let optimize = clap::Command::new("optimize").about("Optimize the graph");
    app = app.subcommand(output_options(optimize));

    let stream_check = clap::Command::new("stream-check")
        .long_about("Compare output of streamed and regular exec");
    app = app.subcommand(output_options(stream_check));
    let matches = app.get_matches();

    let probe = if matches.is_present("readings") {
        let file = std::fs::File::create("readings.out").unwrap();
        let mut probe = Probe::new(file).unwrap();
        probe.register_i64("progress").unwrap();
        let heartbeat = matches.value_of("readings-heartbeat").unwrap().parse::<f32>().unwrap();
        probe.spawn_heartbeat(std::time::Duration::from_secs_f32(heartbeat / 1000.0)).unwrap();
        Some(probe)
    } else {
        None
    };

    if ::std::env::var("TRACT_LOG").is_err() {
        let level = match matches.occurrences_of("verbose") {
            0 => "cli=warn,tract=warn",
            1 => "cli=info,tract=info",
            2 => "cli=debug,tract=debug",
            _ => "cli=trace,tract=trace",
        };
        ::std::env::set_var("TRACT_LOG", level);
    }

    let env = env_logger::Env::default()
        .filter_or("TRACT_LOG", "warn");

    env_logger::Builder::from_env(env).format_timestamp_nanos().init();
    info_usage("init", probe.as_ref());

    if let Err(e) = handle(matches, probe.as_ref()) {
        error!("{:?}", e);
        std::process::exit(1);
    }

    info_usage("done", probe.as_ref());
    Ok(())
}

fn dump_subcommand<'a>() -> clap::Command<'a> {
    use clap::*;
    let dump = clap::Command::new("dump")
        .long_about("Dumps the Tensorflow graph in human readable form.")
        .arg(
            Arg::new("assert-cost")
            .takes_value(true)
            .long("assert-cost")
            .help("Checks computed against the provided value (form: \"FMA(F32)=2060448 DIV(F32)=24576\")")
            )
        .arg(
            Arg::new("nnef-override-output-name")
            .takes_value(true)
            .number_of_values(1)
            .long("nnef-override-output-name")
            .help("Rename output before dumping network")
            )
        .arg(
            Arg::new("nnef-dir")
            .takes_value(true)
            .long("nnef-dir")
            .help("Dump the network in NNEF format (as a directory)"),
            )
        .arg(
            Arg::new("nnef-tar")
            .takes_value(true)
            .long("nnef-tar")
            .help("Dump the network in NNEF format (as a tar file)"),
            )
        .arg(
            Arg::new("nnef")
            .takes_value(true)
            .long("nnef")
            .help("Dump the network in NNEF format (as a tar.gz file)"),
            )
        .arg(
            Arg::new("nnef-graph")
            .takes_value(true)
            .long("nnef-graph")
            .help("Dump the network definition (without the weights) as a graph.nnef-like file"),
            )
        .arg(
            Arg::new("inner")
            .takes_value(true)
            .number_of_values(1)
            .multiple_occurrences(true)
            .long("inner")
            .help("Navigate to a sub-model"),
            );
    let dump = output_options(dump);
    let dump = assertions_options(dump);
    let dump = benchlimits_options(dump);
    dump
}

fn assertions_options(command: clap::Command) -> clap::Command {
    use clap::*;
    command
        .arg(
            Arg::new("assert-output")
                .takes_value(true)
                .multiple_occurrences(true)
                .number_of_values(1)
                .long("assert-output")
                .help("Fact to check the ouput tensor against (@filename, or 3x4xf32)"),
        )
        .arg(
            Arg::new("assert-output-bundle")
                .takes_value(true)
                .long("assert-output-bundle")
                .help("Checks values against these tensor (.npz)"),
        )
        .arg(
            Arg::new("assert-output-fact")
            .takes_value(true)
            .long("assert-output-fact")
            .help("Infered shape and datum type must match exactly this"),
            )
        .arg(
            Arg::new("assert-output-count")
            .takes_value(true)
            .long("assert-output-count")
            .help("Check the number of outputs found."),
            )
        .arg(
            Arg::new("assert-op-count")
            .takes_value(true)
            .forbid_empty_values(true)
            .number_of_values(2)
            .value_names(&["operator", "count"])
            .multiple_occurrences(true)
            .long("assert-op-count")
            .help("Specified operator must appear exactly the specified number of times. This argument can appear multiple times."),
            )
}

fn benchlimits_options(command: clap::Command) -> clap::Command {
    use clap::*;
    command.args(&[
                     arg!(-n --"max-iters" [max_iters] "Sets the maximum number of iterations for each node [default: 100_000]."),
                     arg!(--"max-time" [max_time] "Sets the maximum execution time for each node (in ms) [default: 5000].") ])
}

fn output_options(command: clap::Command) -> clap::Command {
    use clap::*;
    command
        .args(&[
            arg!(--"natural-order" "dump nodes in id order instead of evaluation order"),
            arg!(-q --quiet "don't dump"),
        ])
        .arg(Arg::new("debug-op").long("debug-op").help("show debug dump for each op"))
        .arg(Arg::new("node-id").long("node-id").takes_value(true).help("Select a node to dump"))
        .arg(
            Arg::new("successors")
                .long("successors")
                .takes_value(true)
                .help("Show successors of node"),
        )
        .arg(Arg::new("op-name").long("op-name").takes_value(true).help("Select one op to dump"))
        .arg(
            Arg::new("node-name")
                .long("node-name")
                .takes_value(true)
                .help("Select one node to dump"),
        )
        .arg(Arg::new("const").long("const").help("also display consts nodes"))
        .arg(Arg::new("info").long("info").help("show op inner information"))
        .arg(Arg::new("io-long").long("io-long").help("show full i/o information"))
        .arg(Arg::new("io-none").long("io-none").help("hide i/o information"))
        .arg(Arg::new("json").long("json").help("dump performance info as json"))
        .arg(Arg::new("outlet-labels").long("outlet-labels").help("display outlet labels"))
        .arg(Arg::new("cost").long("cost").help("Include const information"))
        .arg(Arg::new("profile").long("profile").help("Include results for profile run"))
        .arg(
            Arg::new("invariants")
                .takes_value(false)
                .long("invariants")
                .help("Display operators invariants"),
        )
}

/// Handles the command-line input.
fn handle(matches: clap::ArgMatches, probe: Option<&Probe>) -> CliResult<()> {
    if matches.is_present("list-ops") {
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

    let builder_result = Parameters::from_clap(&matches, probe);
    #[allow(unused_mut)]
    let mut params = match builder_result {
        Ok(params) => params,
        Err(e) => {
            if let Some(params::ModelBuildingError(ref broken_model, _)) = e.downcast_ref() {
                let mut broken_model: Box<dyn Model> =
                    tract_hir::tract_core::dyn_clone::clone(broken_model);
                let annotations =
                    crate::annotations::Annotations::from_model(broken_model.as_ref())?;
                let display_params = if let Some(("dump", sm)) = matches.subcommand() {
                    display_params_from_clap(&matches, &sm)?
                } else {
                    crate::display_params::DisplayParams::default()
                };

                if broken_model.output_outlets().len() == 0 {
                    broken_model.auto_outputs()?;
                }
                terminal::render(broken_model.as_ref(), &annotations, &display_params)?;
            }
            Err(e)?
        }
    };

    let mut need_optimisations = false;

    match matches.subcommand() {
        Some(("bench", m)) => {
            need_optimisations = true;
            bench::handle(&params, &BenchLimits::from_clap(&m)?, probe)
        }

        Some(("criterion", _)) => {
            need_optimisations = true;
            bench::criterion(&params)
        }

        Some(("compare", m)) => {
            compare::handle(&mut params, &m, display_params_from_clap(&matches, m)?)
        }

        Some(("run", m)) => run::handle(&params, m),

        #[cfg(feature = "pulse")]
        Some(("stream-check", m)) => {
            stream_check::handle(&params, &display_params_from_clap(&matches, m)?)
        }

        None => dump::handle(
            &params,
            &crate::display_params::DisplayParams::default(),
            &matches,
            &dump_subcommand().get_matches_from::<_, &'static str>([]),
            &params::BenchLimits::default(),
            vec![],
        ),

        Some(("dump", m)) => {
            need_optimisations = m.is_present("profile");
            let inner = m
                .values_of("inner")
                .map(|ss| ss.map(|s| s.to_string()).collect())
                .unwrap_or(vec![]);
            dump::handle(
                &params,
                &display_params_from_clap(&matches, m)?,
                &matches,
                m,
                &BenchLimits::from_clap(&m)?,
                inner,
            )
        }

        Some((s, _)) => bail!("Unknown subcommand {}.", s),
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
    Ok(())
}

fn nnef(matches: &clap::ArgMatches) -> tract_nnef::internal::Nnef {
    let mut fw = tract_nnef::nnef();
    if matches.is_present("nnef-tract-onnx") {
        #[cfg(feature = "onnx")]
        {
            use tract_onnx::WithOnnx;
            fw = fw.with_onnx();
        }
        #[cfg(not(feature = "onnx"))]
        {
            panic!("tract is build without ONNX support")
        }
    }
    if matches.is_present("nnef-tract-pulse") {
        #[cfg(feature = "pulse-opl")]
        {
            use tract_pulse::WithPulse;
            fw = fw.with_pulse();
        }
        #[cfg(not(feature = "pulse-opl"))]
        {
            panic!("tract is build without pulse-opl support")
        }
    }
    if matches.is_present("nnef-tract-core") {
        fw = fw.with_tract_core();
    }
    fw
}
