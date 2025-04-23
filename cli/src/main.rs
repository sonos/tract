#![allow(clippy::len_zero)]
#![allow(clippy::redundant_closure_call)]
#[macro_use]
extern crate log;

#[macro_use]
mod macros;

#[allow(unused_imports)]
use tract_itertools::Itertools;

use tract_core::internal::*;
use tract_hir::internal::*;

use nu_ansi_term::Color::*;
use tract_libcli::annotations::Annotations;
use tract_libcli::display_params::DisplayParams;
use tract_libcli::model::Model;
use tract_libcli::profile::BenchLimits;

use fs_err as fs;
use readings_probe::*;

mod bench;
mod compare;
mod cost;
mod dump;
mod llm;
#[cfg(any(target_os = "macos", target_os = "ios"))]
mod memory_arena;
mod params;
mod plan_options;
mod run;
#[cfg(feature = "pulse")]
mod stream_check;
mod tensor;
mod utils;

use params::*;
use tract_linalg::block_quant::Q4_0;
use tract_linalg::WeightType;

readings_probe::instrumented_allocator!();

pub const QUALITY_COLORS: [nu_ansi_term::Color; 5] = [LightGreen, Green, White, Yellow, LightRed];

fn info_usage(stage: &str, probe: Option<&Probe>) {
    if let Some(mon) = probe {
        let _ = mon.log_event(stage);
    }
    if log::log_enabled!(log::Level::Info) {
        let usage = readings_probe::get_os_readings().unwrap();
        let allocated = readings_probe::alloc::ALLOCATED.load(std::sync::atomic::Ordering::Relaxed);
        let freeed = readings_probe::alloc::FREEED.load(std::sync::atomic::Ordering::Relaxed);
        info!(
            "Resource usage {}: vsz:{} rsz:{} rszmax:{} alloc:{}",
            stage,
            usage.virtual_size,
            usage.resident_size,
            usage.resident_size_max,
            allocated.saturating_sub(freeed),
        );
    }
}

pub const STAGES: &[&str] = &[
    "load",
    "analyse",
    "incorporate",
    "type",
    "declutter",
    "pulse",
    "pulse-to-type",
    "pulse-declutter",
    "set",
    "set-declutter",
    "nnef-cycle",
    "nnef-cycle-declutter",
    "tflite-cycle-predump",
    "tflite-cycle",
    "tflite-cycle-declutter",
    "before-optimize",
    "optimize",
];

/// Entrypoint for the command-line interface.
fn main() -> TractResult<()> {
    use clap::*;
    let mut app = command!()
        .setting(AppSettings::DeriveDisplayOrder)
        .allow_hyphen_values(true)
        .arg(arg!(--readings "Start readings instrumentation"))
        .arg(arg!(--"readings-heartbeat" [MS] "Heartbeat for readings background collector").default_value("5"))
        .arg(arg!(verbose: -v ... "Sets the level of verbosity."))
        .arg(arg!(--"keep-last" "Keep last model alive to dump if there is an error"))
        .arg(arg!([model] "Sets the model to use").required(false))
        .arg(arg!(-f --format [format]
                  "Hint the model format ('onnx', 'nnef', 'tflite' or 'tf') instead of guess from extension."))
        .arg(Arg::new("input").long("input").short('i').multiple_occurrences(true).takes_value(true).long_help(
                "Set input shape and type (@file.pb or @file.npz:thing.npy or 3,4,i32)."))
        .arg(Arg::new("constantize").long("constantize").multiple_occurrences(true).takes_value(true).long_help(
                "Transorm an input into a Constant"))

        .arg(arg!(--"assert").multiple_occurrences(true).takes_value(true).long_help("Adds a TDim pre-condition (prefix by optional \"scenario_name:\")"))
        .arg(arg!(--"scenario").multiple_occurrences(true).takes_value(true).long_help("Adds a scenario"))

        // deprecated
        .arg(arg!(--"input-bundle" [input_bundle] "Path to an input container (.npz). This sets input facts and tensor values.").hide(true))
        // deprecated
        .arg(arg!(--"allow-random-input" "Will use random generated input").hide(true))

        .arg(arg!(--"input-facts-from-bundle" [input_bundle] "Path to an input container (.npz). This only sets input facts."))

        .arg(arg!(--"edge-left-context" [frames] "Add lines of left context to input (dupping first time frame)").alias("kaldi-left-context"))
        .arg(arg!(--"edge-right-context" [frames] "Add lines of right context to input (dupping last time frame)").alias("kaldi-right-context"))

        .arg(arg!(--"onnx-test-data-set" [data_set] "Use onnx-test data-set as input (expect test_data_set_N dir with input_X.pb, etc. inside)"))
        .arg(arg!(--"onnx-ignore-output-shapes" "Ignore output shapes from model (workaround for pytorch export bug with mask axes)"))
        .arg(arg!(--"onnx-ignore-output-types" "Ignore output shapes from types (workaround for tdim conflicting with integer types)"))

        .arg(arg!(--"input-node" [node] ... "Override input nodes names (auto-detects otherwise)."))
        .arg(Arg::new("output-node").long("output-node").multiple_occurrences(true).takes_value(true).long_help(
                "Override output nodes by name."))
        .arg(arg!(--"label-wires" "Propagate node labels to wires"))

        .arg(arg!(--"tf-initializer-output-node" [node] "Set an initializer node"))

        .arg(arg!(--"override-fact" [fact] "Override a fact."))

        .arg(arg!(--"analyse-fail-fast" "Stop analyse at first error."))
        .arg(arg!(--recursive "Apply to sub graphes"))
        .arg(arg!(--proto "Keep proto model around after parse"))
        .arg(arg!(--determinize "Enforce a seed in random operator"))
        .arg(arg!(--partial "Before analyse, eliminate dead branches"))

        .arg(arg!(--pass [STAGE] "Pass to stop preprocessing after.").possible_values(STAGES))
        .arg(arg!(--"declutter-step" [STEP] "Stop decluttering process after application of patch number N"))
        .arg(arg!(--"declutter-set-step" [STEP] "Stop decluttering process (the one after --set application) at patch number N"))
        .arg(arg!(--"optimize-step" [STEP] "Stop optimizing process after application of patch number N"))
        .arg(arg!(--"extract-decluttered-sub" [SUB] "Zoom on a subgraph after decluttering by parent node name"))

        .arg(Arg::new("f32-to-f16").long("f32-to-f16").alias("half-floats").long_help("Convert the decluttered network from f32 to f16"))
        .arg(arg!(--"f16-to-f32" "Convert the decluttered network from f16 to f32"))
        .arg(arg!(--"metal").long_help("Convert supported operators to Metal GPU equivalent. Only available on MacOS and iOS"))
        .arg(Arg::new("force-metal-backend").long("force-metal-backend").takes_value(true).long_help("Force specific implementations for MM kernels. Possible values: mlx, ggml, mfa. Backend is dynamically selected if option is not present"))
        .arg(Arg::new("metal-gpu-trace").long("metal-gpu-trace").takes_value(true).help("Capture Metal GPU trace at given path. Only available on MacOS and iOS"))
        .arg(Arg::new("transform").short('t').long("transform").multiple_occurrences(true).takes_value(true).help("Apply a built-in transformation to the model"))
        .arg(Arg::new("set").long("set").multiple_occurrences(true).takes_value(true)
             .long_help("Set a symbol to a concrete value after decluttering"))

        // deprecated
        .arg(arg!(--"allow-float-casts" "Allow casting between f16, f32 and f64 around model").hide(true))

        .arg(arg!(--"nnef-cycle" "Perform NNEF dump and reload before optimizing"))
        .arg(arg!(--"tflite-cycle" "Perform TFLITE dump and reload before optimizing"))

        .arg(arg!(--"nnef-tract-core" "Allow usage of tract-core extension in NNEF dump and load"))
        .arg(arg!(--"nnef-tract-onnx" "Allow usage of tract-onnx extension in NNEF dump and load"))
        .arg(arg!(--"nnef-tract-pulse" "Allow usage of tract-pulse extension in NNEF dump and load"))
        .arg(arg!(--"nnef-tract-extra" "Allow usage of tract-extra extension in NNEF dump and load"))
        .arg(arg!(--"nnef-tract-transformers" "Allow usage of tract-transformers extension in NNEF dump and load"))
        .arg(arg!(--"nnef-extended-identifier" "Allow usage of the i\"...\" syntax to escape identifier names"))

        .arg(arg!(--"threads" [THREADS] "Setup a thread pool for computing. 0 will guess the number of physical cores"))

        .arg(arg!(-O --optimize "Optimize before running"))
        .arg(arg!(--"assert-maximal-mm-quality-cost" [MAX] "Maximum value for quality category (0=assembly, 4=dreadful rust code)"))
        .arg(arg!(--pulse [PULSE] "Translate to pulse network"))

        .arg(arg!(--"machine-friendly" "Machine friendly output"))

        .subcommand(Command::new("list-ops").about("List ops in TF/ONNX frameworks"))
        .subcommand(Command::new("kernels").about("Print kernels for the current plaform"));

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
    let compare = run_options(compare);
    let compare = assertions_options(compare);
    app = app.subcommand(output_options(compare));

    let bench =
        clap::Command::new("bench").long_about("Benchmarks tract on randomly generated input.");
    let bench = run_options(bench);
    let bench = output_options(bench);
    let bench = bench_options(bench);
    let bench = assertions_options(bench);
    app = app.subcommand(bench);

    let criterion = clap::Command::new("criterion")
        .long_about("Benchmarks tract on randomly generated input using criterion.");
    let criterion = run_options(criterion);
    app = app.subcommand(criterion);

    app = app.subcommand(dump_subcommand());

    let run = clap::Command::new("run")
        .long_about("Run the graph")
        .arg(Arg::new("dump").long("dump").help("Show output"))
        .arg(
            Arg::new("save-outputs-npz")
                .long("save-outputs-npz")
                .alias("save-outputs")
                .takes_value(true)
                .help("Save the outputs into a npz file"),
        )
        .arg(
            Arg::new("save-outputs-nnef")
                .long("save-outputs-nnef")
                .takes_value(true)
                .help("Save the output tensor into a folder of nnef .dat files"),
        )
        .arg(Arg::new("steps").long("steps").help("Show all inputs and outputs"))
        .arg(
            Arg::new("save-steps")
                .long("save-steps")
                .takes_value(true)
                .help("Save intermediary values as a npz file"),
        )
        .arg(
            Arg::new("check-f16-overflow")
                .long("check-f16-overflow")
                .help("Check for f16 overflow in all outputs"),
        )
        .arg(
            Arg::new("assert-sane-floats")
                .long("assert-sane-floats")
                .help("Check float for NaN and infinites at each step"),
        );
    let run = run_options(run);
    let run = output_options(run);
    let run = assertions_options(run);
    app = app.subcommand(run);

    let llm_bench =
        clap::Command::new("llm-bench").long_about("llamas.cpp-style bench (tg128 and pp512)");
    let llm_bench = assertions_options(llm_bench);
    let llm_bench = run_options(llm_bench);
    app = app.subcommand(llm_bench);

    let stream_check = clap::Command::new("stream-check")
        .long_about("Compare output of streamed and regular exec");
    app = app.subcommand(output_options(stream_check));

    let matches = app.get_matches();

    let probe = if matches.is_present("readings") {
        let file = fs::File::create("readings.out").unwrap();
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

    let env = env_logger::Env::default().filter_or("TRACT_LOG", "warn");

    env_logger::Builder::from_env(env).format_timestamp_nanos().init();
    info_usage("init", probe.as_ref());

    let res = if matches.is_present("metal-gpu-trace") {
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            let gpu_trace_path =
                std::path::Path::new(matches.value_of("metal-gpu-trace").unwrap()).to_path_buf();
            ensure!(gpu_trace_path.is_absolute(), "Metal GPU trace file has to be absolute");
            ensure!(
                !gpu_trace_path.exists(),
                format!("Given Metal GPU trace file {:?} already exists.", gpu_trace_path)
            );
            log::info!("Capturing Metal GPU trace at : {:?}", gpu_trace_path);
            std::env::set_var("METAL_CAPTURE_ENABLED", "1");
            std::env::set_var("METAL_DEVICE_WRAPPER_TYPE", "1");
            let probe_ref = probe.as_ref();
            tract_metal::METAL_STREAM.with_borrow(move |stream| {
                stream.capture_trace(gpu_trace_path, move |_stream| handle(matches, probe_ref))
            })
        }
        #[cfg(not(any(target_os = "macos", target_os = "ios")))]
        {
            bail!("`--metal-gpu-trace` present but it is only available on MacOS and iOS")
        }
    } else {
        handle(matches, probe.as_ref())
    };

    if let Err(e) = res {
        error!("{e:?}");
        std::process::exit(1);
    }

    info_usage("done", probe.as_ref());
    Ok(())
}

#[allow(clippy::let_and_return)]
fn dump_subcommand<'a>() -> clap::Command<'a> {
    use clap::*;
    let dump = clap::Command::new("dump")
        .long_about("Dumps the graph in human readable form.")
        .arg(
            Arg::new("axes")
            .long("axes")
            .help("Compute and display axis tracking")
            )
        .arg(
            Arg::new("axes-names")
            .takes_value(true)
            .number_of_values(1)
            .multiple_occurrences(true)
            .long("axes-names")
            .help("Gave meaningful names to axes: [node_name=]axis0,axis1,..,axisN (apply to first input if no node_name is provided)")
            )
        .arg(
            Arg::new("assert-cost")
            .takes_value(true)
            .long("assert-cost")
            .help("Checks computed against the provided value (form: \"FMA(F32)=2060448 DIV(F32)=24576\")")
            )
        .arg(
            Arg::new("memory-arena")
            .takes_value(true)
            .long("memory-arena")
            .help("Dump arena memory statistics to a JSON file (MacOS / iOS only)")
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
            Arg::new("tflite")
            .takes_value(true)
            .long("tflite")
            .help("Dump the network in TfLite format"),
            )
        .arg(
            Arg::new("compress-submodels")
            .long("compress-submodels")
            .help("Compress submodels if any (as a .tgz file)"),
        )
        .arg(
            Arg::new("nnef-deterministic")
            .long("nnef-deterministic")
            .help("If provided, will try to make output .nnef.tar files deterministic"),
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
    let dump = run_options(dump);
    let dump = output_options(dump);
    let dump = assertions_options(dump);
    let dump = bench_options(dump);
    dump
}

fn assertions_options(command: clap::Command) -> clap::Command {
    use clap::*;
    command
        .arg(
            Arg::new("approx")
            .takes_value(true)
            .possible_values(["exact", "close", "approximate", "very", "super", "ultra"])
            .default_value("close")
            .long("approx")
            .help("Approximation level used in assertions."),
            )
        .arg(
            Arg::new("approx-custom")
            .takes_value(true)
            .long("approx-custom")
            .help("Approximation level used in assertions (atol, rtol, outlier ratio). 3 coma-separated floats."),
            )
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
            Arg::new("allow-missing-outputs")
            .long("allow-missing-outputs")
            .help("Allow missing output in checks")
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

fn bench_options(command: clap::Command) -> clap::Command {
    use clap::*;
    command.args(&[
                 arg!(--"warmup-time" [warmup_time] "Time to run (approx.) before starting the clock."),
                 arg!(--"warmup-loops" [warmup_loops] "Number of loops to run before starting the clock."),
                 arg!(--"max-loops" [max_iters] "Sets the maximum number of iterations for each node [default: 100_000].").alias("max-iters"),
                 arg!(--"max-time" [max_time] "Sets the maximum execution time for each node (in ms) [default: 5000].") ])
}

fn run_options(command: clap::Command) -> clap::Command {
    use clap::*;
    command
        .arg(
            Arg::new("input-from-npz")
            .long("input-from-npz")
            .alias("input-from-bundle")
            .takes_value(true)
            .help("Path to an input container (.npz). This sets tensor values."),
            )
        .arg(
            Arg::new("set")
                .long("set")
                .takes_value(true)
                .multiple_occurrences(true)
                .number_of_values(1)
                .help("Set a symbol value before running the model (--set S=12)"),
        )
        .arg(
            Arg::new("input-from-nnef").long("input-from-nnef").takes_value(true).help(
                "Path to a directory containing input tensors in NNEF format (.dat files). This sets tensor values.",
                ),
                )
        .arg(arg!(--pp [pp] "Random input for LLM-like prompt processing"))
        .arg(arg!(--tg [tg] "Random input for LLM-like text generation"))
        .arg(Arg::new("skip-order-opt-ram")
            .long("skip-order-opt-ram")
            .help("Plan node evaluation order without RAM optimisation"),
            )
        .arg(
            Arg::new("allow-random-input")
            .short('R')
            .long("allow-random-input")
            .help("Will use random generated input"),
            )
        .arg(
            Arg::new("random-range")
            .long("random-range")
            .multiple_occurrences(true)
            .takes_value(true)
            .help("Constraint random values to a given range (example: input=1.0..10.0)"),
            )
        .arg(
            Arg::new("allow-float-casts")
            .long("allow-float-casts")
            .help("Allow casting between f16, f32 and f64 around model"),
            )
}

fn output_options(command: clap::Command) -> clap::Command {
    use clap::*;
    command
        .args(&[
            arg!(--"natural-order" "dump nodes in id order instead of evaluation order"),
            arg!(--"opt-ram-order" "dump nodes in RAM optimising order"),
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
        .arg(Arg::new("mm").long("mm").help("display Matrix Multiplication report"))
        .arg(Arg::new("outlet-labels").long("outlet-labels").help("display outlet labels"))
        .arg(Arg::new("cost").long("cost").help("Include const information"))
        .arg(
            Arg::new("tmp_mem_usage")
                .long("tmp-mem-usage")
                .help("Include temporary memory usage information"),
        )
        .arg(Arg::new("profile").long("profile").help("Include results for profile run"))
        .arg(Arg::new("folded").long("folded").help("Don't display submodel informations"))
        .arg(
            Arg::new("invariants")
                .takes_value(false)
                .long("invariants")
                .help("Display operators invariants"),
        )
}

/// Handles the command-line input.
fn handle(matches: clap::ArgMatches, probe: Option<&Probe>) -> TractResult<()> {
    match matches.subcommand() {
        Some(("list-ops", _)) => {
            #[cfg(feature = "onnx")]
            {
                let onnx = tract_onnx::onnx();
                let names = onnx.op_register.0.keys().sorted().join(", ");
                println!("Onnx:\n");
                println!("{names}");
                println!("\n");
            }
            #[cfg(feature = "tf")]
            {
                let tf = tract_tensorflow::tensorflow();
                let names = tf.op_register.0.keys().sorted().join(", ");
                println!("Tensorflow:\n");
                println!("{names}");
                println!("\n");
            }
            return Ok(());
        }
        Some(("kernels", _)) => {
            println!();
            println!("{}", White.bold().paint("# By implementation"));
            println!();
            for m in tract_linalg::ops().mmm_impls() {
                println!(
                    "{} -> {:?}",
                    QUALITY_COLORS[m.quality().cost()].paint(m.name()),
                    m.stores()
                );
                for packings in m.packings() {
                    println!("   - {:?} • {:?}", packings.0, packings.1);
                }
            }
            println!();
            println!("{}", White.bold().paint("# By weights"));
            println!();
            for w in [
                WeightType::Plain(f16::datum_type()),
                WeightType::Plain(f32::datum_type()),
                WeightType::Plain(f64::datum_type()),
                WeightType::Plain(i8::datum_type()),
                WeightType::from(Q4_0),
            ] {
                println!("{}", White.bold().paint(format!("{w:?}")));
                for packing in tract_linalg::ops()
                    .all_possible_packing(w)
                    .sorted_by_key(|f| format!("{f:?}"))
                    .dedup()
                {
                    println!("  * {packing:?}");
                    for mmm in tract_linalg::ops().mmm_impls() {
                        for (ix, p) in mmm.packings().iter().enumerate() {
                            if p.0.same_as(packing) {
                                println!(
                                    "    - {} ({ix}) {:?} {:?}",
                                    QUALITY_COLORS[mmm.quality().cost()].paint(mmm.name()),
                                    p.0,
                                    p.1
                                );
                            } else if let Some(pe) = tract_linalg::ops()
                                .panel_extractors()
                                .iter()
                                .find(|pe| pe.from.same_as(packing) && p.0.same_as(&pe.to))
                            {
                                println!(
                                    "    - {} ({ix}) {:?} {:?} using {}",
                                    QUALITY_COLORS[mmm.quality().cost()].paint(mmm.name()),
                                    p.0,
                                    p.1,
                                    pe.name
                                );
                            }
                        }
                    }
                }
            }
            return Ok(());
        }
        _ => (),
    }

    let builder_result = Parameters::from_clap(&matches, probe);
    #[allow(unused_mut)]
    let mut params = match builder_result {
        Ok(params) => params,
        Err(e) => {
            if let Some(params::ModelBuildingError(ref broken_model, _)) = e.downcast_ref() {
                let mut broken_model: Box<dyn Model> =
                    tract_hir::tract_core::dyn_clone::clone(broken_model);
                let annotations = Annotations::from_model(broken_model.as_ref())?;
                let display_params = if let Some(("dump", sm)) = matches.subcommand() {
                    display_params_from_clap(&matches, sm)?
                } else {
                    DisplayParams::default()
                };

                if broken_model.output_outlets().len() == 0 {
                    broken_model.auto_outputs()?;
                }
                tract_libcli::terminal::render(
                    broken_model.as_ref(),
                    &annotations,
                    &display_params,
                )?;
            }
            Err(e)?
        }
    };

    let mut need_optimisations = false;

    #[cfg(feature = "multithread-mm")]
    if let Some(threads) = matches.value_of("threads") {
        let threads: usize = threads.parse()?;
        let threads = if threads == 0 { num_cpus::get_physical() } else { threads };
        multithread::set_default_executor(multithread::Executor::multithread(threads));
    }
    #[cfg(not(feature = "multithread-mm"))]
    if matches.value_of("threads").is_some() {
        bail!("tract is compiled without multithread support")
    }

    match matches.subcommand() {
        Some(("bench", m)) => {
            need_optimisations = true;
            bench::handle(&params, &matches, m, &params::bench_limits_from_clap(m)?, probe)
        }

        Some(("criterion", m)) => {
            need_optimisations = true;
            bench::criterion(&params, &matches, m)
        }

        Some(("compare", m)) => {
            compare::handle(&mut params, &matches, m, display_params_from_clap(&matches, m)?)
        }

        Some(("run", m)) => run::handle(&params, &matches, m),

        #[cfg(feature = "pulse")]
        Some(("stream-check", m)) => {
            stream_check::handle(&params, &display_params_from_clap(&matches, m)?)
        }

        None => dump::handle(
            &params,
            &DisplayParams::default(),
            &matches,
            &dump_subcommand().get_matches_from::<_, &'static str>([]),
            &BenchLimits::default(),
            vec![],
        ),

        Some(("dump", m)) => {
            need_optimisations = m.is_present("profile");
            let inner = m
                .values_of("inner")
                .map(|ss| ss.map(|s| s.to_string()).collect())
                .unwrap_or_default();
            dump::handle(
                &params,
                &display_params_from_clap(&matches, m)?,
                &matches,
                m,
                &params::bench_limits_from_clap(m)?,
                inner,
            )
        }

        Some(("llm-bench", m)) => {
            need_optimisations = true;
            llm::handle(&params, &matches, m, &params::bench_limits_from_clap(m)?, probe)
        }

        Some((s, _)) => bail!("Unknown subcommand {}.", s),
    }?;

    if need_optimisations {
        let style = nu_ansi_term::Style::new().fg(nu_ansi_term::Color::Red).bold();
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
    if matches.is_present("nnef-tract-extra") {
        #[cfg(feature = "extra")]
        {
            use tract_extra::WithTractExtra;
            fw = fw.with_tract_extra();
        }
        #[cfg(not(feature = "extra"))]
        {
            panic!("tract is build without tract-extra support")
        }
    }
    if matches.is_present("nnef-tract-transformers") {
        #[cfg(feature = "transformers")]
        {
            use tract_transformers::WithTractTransformers;
            fw = fw.with_tract_transformers();
        }
        #[cfg(not(feature = "transformers"))]
        {
            panic!("tract is build without tract-transformers support")
        }
    }
    if matches.is_present("nnef-tract-core") {
        fw = fw.with_tract_core();
    }
    if matches.is_present("nnef-extended-identifier") {
        fw.allow_extended_identifier_syntax(true);
    }
    fw
}
