#[macro_use]
extern crate clap;
#[macro_use]
extern crate log;
#[macro_use]
extern crate error_chain;
extern crate simplelog;
extern crate tfdeploy;
#[cfg(feature="tensorflow")]
extern crate conform;
extern crate time;
extern crate rand;
extern crate ndarray;

use std::process::exit;
use std::path::Path;
use simplelog::{TermLogger, LevelFilter, Config};
use tfdeploy::tfpb;
use tfpb::types::DataType;
use time::PreciseTime;
use rand::Rng;


/// Configures error handling for this module.
error_chain! {
    links {
        Conform(conform::Error, conform::ErrorKind) #[cfg(feature="tensorflow")];
        Tfdeploy(tfdeploy::Error, tfdeploy::ErrorKind);
    }

    foreign_links {
        Io(std::io::Error);
        Int(std::num::ParseIntError);
    }
}

/// Structure holding the parsed parameters.
#[allow(dead_code)]
struct Parameters {
    tfd_model: tfdeploy::Model,

    #[cfg(feature="tensorflow")]
    tf_model: conform::tf::Tensorflow,

    input: String,
    output: String,
    size_x: usize,
    size_y: usize,
    size_d: DataType,
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
            "Sets the TensorFlow model to use (in Protobuf format)")

        (@arg input: -i --input [input]
            "Sets the input node name")

        (@arg output: -o --output [output]
            "Sets the output node name")

        (@arg size: -s --size <size>
            "Sets the input size, e.g. 32x64xf32")

        (@arg debug: -d ... "Sets the level of debugging information")

        (@subcommand compare =>
            (about: "Compares the output of tfdeploy and tensorflow on randomly generated input"))

        (@subcommand profile =>
            (about: "Benchmarks tfdeploy on randomly generated input")
            (@arg iters: -n [iters]
                "Sets the number of iterations for the average [default: 5]"))
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
        exit(1)
    }
}


/// Handles the command-line input.
fn handle(matches: clap::ArgMatches) -> Result<()> {
    let params = parse(&matches)?;

    match matches.subcommand() {
        ("compare", _) => handle_compare(params),
        ("profile", _) => handle_profile(params),
        (s, _) => bail!("Unknown subcommand {}.", s)
    }
}


/// Parses the command-line arguments.
fn parse(matches: &clap::ArgMatches) -> Result<Parameters> {
    let path = Path::new(matches.value_of("model").unwrap());
    let tfd_model = tfdeploy::for_path(&path)?;

    #[cfg(feature="tensorflow")]
    let tf_model = conform::tf::for_path(&path)?;

    let sizes: Vec<&str> = matches
        .value_of("size")
        .unwrap()
        .splitn(3, "x")
        .collect();

    if sizes.len() < 3 {
        bail!("Size should be formatted as {size}x{size}x{type}.");
    }

    let size_x = sizes[0].parse::<usize>()?;
    let size_y = sizes[1].parse::<usize>()?;
    let size_d = match sizes[2].to_lowercase().as_str() {
        "f64" => DataType::DT_DOUBLE,
        "f32" => DataType::DT_FLOAT,
        "i32" => DataType::DT_INT32,
        "i8" => DataType::DT_INT8,
        "u8" => DataType::DT_UINT8,
        _ => bail!("Type of the input should be f64, f32, i32, i8 or u8.")
    };

    let input = match matches.value_of("input") {
        Some(name) => name.to_string(),
        None => detect_input(&tfd_model)?
    };

    let output = match matches.value_of("output") {
        Some(name) => name.to_string(),
        None => detect_output(&tfd_model)?
    };

    #[cfg(feature="tensorflow")]
    return Ok(Parameters { tfd_model, tf_model, input, output, size_x, size_y, size_d });

    #[cfg(not(feature="tensorflow"))]
    return Ok(Parameters { tfd_model, input, output, size_x, size_y, size_d });
}


/// Tries to autodetect the name of the input node.
#[allow(unused_variables)]
fn detect_input(model: &tfdeploy::Model) -> Result<String> {
    unimplemented!()
}


/// Tries to autodetect the name of the output node.
#[allow(unused_variables)]
fn detect_output(model: &tfdeploy::Model) -> Result<String> {
    unimplemented!()
}


/// Handles the `compare` subcommand.
#[allow(unused_variables)]
#[cfg(not(feature="tensorflow"))]
fn handle_compare(params: Parameters) -> Result<()> {
    bail!("Comparison requires the `tensorflow` feature.")
}

#[allow(unused_variables)]
#[cfg(feature="tensorflow")]
fn handle_compare(params: Parameters) -> Result<()> {
    unimplemented!()
}


/// Handles the `profile` subcommand.
fn handle_profile(params: Parameters) -> Result<()> {
    let iters = 5; // todo(romain)

    let model = params.tfd_model;
    let input = model.get_node(params.input.as_str())?;
    let _output = model.get_node(params.output.as_str())?;
    let plan = input.eval_order(&model)?;

    let mut state = model.state();

    // First fill the input with randomly generated values.
    state.set_value(
        input.id,
        random_matrix(params.size_x, params.size_y, params.size_d)
    )?;

    // Then execute the plan while profiling each step.
    println!("{:?}", plan);
    for n in plan {
        if n == input.id {
            continue;
        }

        let start = PreciseTime::now();
        for _ in 0..iters { state.compute_one(n)?; }
        let end = PreciseTime::now();

        println!(
            "Node {}: {} ms on average.", n,
            start.to(end).num_milliseconds() / iters
        );
    }

    Ok(())
}


/// Generates a random matrix of a given size and type.
fn random_matrix(x: usize, y: usize, d: DataType) -> tfdeploy::Matrix {
    macro_rules! for_type {
        ($t:ty) => (
            ndarray::Array::from_shape_fn(
                (x, y),
                |_| rand::thread_rng().gen()
            ) as ndarray::Array2<$t>
        )
    }

    match d {
        DataType::DT_DOUBLE => for_type!(f64).into(),
        DataType::DT_FLOAT => for_type!(f32).into(),
        DataType::DT_INT32 => for_type!(i32).into(),
        DataType::DT_INT8 => for_type!(i8).into(),
        DataType::DT_UINT8 => for_type!(u8).into(),
        _ => unimplemented!()
    }
}