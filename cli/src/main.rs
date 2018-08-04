extern crate bincode;
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
#[macro_use]
extern crate rouille;
extern crate open;
extern crate serde;
#[macro_use]
extern crate serde_derive;
extern crate serde_json;

use std::fs::File;
use std::io::Read;
use std::process;
use std::str::FromStr;

use insideout::InsideOut;
use simplelog::Level::{Error, Trace};
use simplelog::{Config, LevelFilter, TermLogger};
use tfdeploy::tfpb;
use tfdeploy::{ DataType, Tensor };
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
mod graphviz;
mod profile;
mod prune;
mod rusage;
mod utils;
mod web;

/// The default maximum for iterations and time.
const DEFAULT_MAX_ITERS: u64 = 100_000;
const DEFAULT_MAX_TIME: u64 = 200;

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
            Arg::with_name("max_iters")
                .short("n")
                .help("Sets the maximum number of iterations for each node [default: 10_000]."),
        )
        .arg(
            Arg::with_name("max_time")
                .short("t")
                .help("Sets the maximum execution time for each node (in ms) [default: 500]."),
        )
        .arg(
            Arg::with_name("buffering")
                .short("b")
                .help("Run the stream network without inner instrumentations"),
        );
    app = app.subcommand(output_options(profile));

    let analyse = clap::SubCommand::with_name("analyse")
        .help("Analyses the graph to infer properties about tensors (experimental).");
    app = app.subcommand(output_options(analyse));

    let optimize = clap::SubCommand::with_name("optimize").help("Optimize the graph");
    app = app.subcommand(output_options(optimize));

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
            Arg::with_name("const")
                .long("const")
                .help("also display consts nodes"),
        )
}

/// Structure holding the parsed parameters.
pub struct Parameters {
    name: String,
    graph: GraphDef,
    tfd_model: tfdeploy::Model,

    #[cfg(feature = "tensorflow")]
    tf_model: conform::tf::Tensorflow,

    input: Option<InputParameters>,
    input_node_ids: Vec<usize>,
    output_node_id: usize,
}

impl Parameters {
    /// Parses the command-line arguments.
    pub fn from_clap(matches: &clap::ArgMatches) -> Result<Parameters> {
        let name = matches.value_of("model").unwrap();
        let graph = tfdeploy::Model::graphdef_for_path(&name)?;
        let tfd_model = tfdeploy::for_path(&name)?;

        #[cfg(feature = "tensorflow")]
        let tf_model = conform::tf::for_path(&name)?;

        let input = InputParameters::from_clap(matches)?;

        let input_node_ids = match matches.values_of("inputs") {
            Some(names) => names
                .map(|s| Ok(tfd_model.node_id_by_name(s)?))
                .collect::<Result<_>>()?,
            None => tfdeploy::analyser::detect_inputs(&tfd_model)?
                .ok_or("Impossible to auto-detect input nodes: no placeholder.")?,
        };

        let output_node_id = match matches.value_of("output") {
            Some(name) => tfd_model.node_id_by_name(name)?,
            None => tfdeploy::analyser::detect_output(&tfd_model)?
                .ok_or("Impossible to auto-detect output nodes.")?,
        };

        #[cfg(feature = "tensorflow")]
        return Ok(Parameters {
            name: name.to_string(),
            graph,
            tfd_model,
            tf_model,
            input_node_ids,
            output_node_id,
            input,
        });

        #[cfg(not(feature = "tensorflow"))]
        return Ok(Parameters {
            name: name.to_string(),
            graph,
            tfd_model,
            input_node_ids,
            output_node_id,
            input,
        });
    }
}

/// Structure holding the input parameters (eventually containing data).
pub struct InputParameters {
    data: Option<Tensor>,
    shape: Vec<Option<usize>>,
    datatype: DataType,
}

impl InputParameters {
    fn from_clap(matches: &clap::ArgMatches) -> Result<Option<InputParameters>> {
        let input = match (matches.value_of("size"), matches.value_of("data")) {
            (_, Some(filename)) => Some(Self::for_data(filename)?),
            (Some(size), _) => Some(Self::for_size(size)?),
            _ => None,
        };
        Ok(input)
    }

    fn for_size(size: &str) -> std::result::Result<InputParameters, errors::Error> {
        let splits = size.split("x").collect::<Vec<_>>();

        if splits.len() < 1 {
            bail!("The <size> argument should be formatted as {size}x{...}x{type}.");
        }

        let (datatype, shape) = splits.split_last().unwrap();

        let shape = shape
            .iter()
            .map(|s| match *s {
                "S" => Ok(None),           // Streaming dimension.
                _ => Ok(Some(s.parse()?)), // Regular dimension.
            })
            .collect::<Result<Vec<_>>>()?;

        if shape.iter().filter(|o| o.is_none()).count() > 1 {
            bail!("The <size> argument doesn't support more than one streaming dimension.");
        }

        let datatype = match datatype.to_lowercase().as_str() {
            "f64" => DataType::F64,
            "f32" => DataType::F32,
            "i32" => DataType::I32,
            "i8" => DataType::I8,
            "u8" => DataType::U8,
            _ => bail!("Type of the input should be f64, f32, i32, i8 or u8."),
        };

        Ok(InputParameters {
            data: None,
            shape,
            datatype,
        })
    }

    /// Parses the `data` command-line argument.
    fn for_data(filename: &str) -> Result<InputParameters> {
        let mut file = File::open(filename)?;
        let mut data = String::new();
        file.read_to_string(&mut data)?;

        let mut lines = data.lines();
        let InputParameters {
            shape, datatype, ..
        } = InputParameters::for_size(lines.next().ok_or("Empty data file")?)?;

        let values = lines.flat_map(|l| l.split_whitespace()).collect::<Vec<_>>();

        // We know there is at most one streaming dimension, so we can deduce the
        // missing value with a simple division.
        let product: usize = shape.iter().map(|o| o.unwrap_or(1)).product();
        let missing = values.len() / product;
        let data_shape = shape
            .iter()
            .map(|o| o.unwrap_or(missing))
            .collect::<Vec<_>>();

        macro_rules! for_type {
            ($t:ty) => {{
                let array = ndarray::Array::from_iter(values.iter().map(|v| v.parse::<$t>().unwrap()));

                array.into_shape(data_shape)?
            }};
        }

        let tensor = match datatype {
            DataType::F64 => for_type!(f64).into(),
            DataType::F32 => for_type!(f32).into(),
            DataType::I32 => for_type!(i32).into(),
            DataType::I8 => for_type!(i8).into(),
            DataType::U8 => for_type!(u8).into(),
            _ => unimplemented!(),
        };

        Ok(InputParameters {
            data: Some(tensor),
            shape,
            datatype,
        })
    }

    fn streaming(&self) -> bool {
        self.shape.iter().any(|dim| dim.is_none())
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
    pub fn from_clap(matches: &clap::ArgMatches, streaming: bool) -> Result<ProfilingMode> {
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
    json: Option<String>,
}

impl OutputParameters {
    pub fn from_clap(matches: &clap::ArgMatches) -> Result<OutputParameters> {
        Ok(OutputParameters {
            web: matches.is_present("web"),
            konst: matches.is_present("const"),
            json: matches.value_of("json").map(String::from),
        })
    }
}

/// Handles the command-line input.
fn handle(matches: clap::ArgMatches) -> Result<()> {
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
    let streaming = params
        .input
        .as_ref()
        .map(|i| i.streaming())
        .unwrap_or(false);

    match matches.subcommand() {
        ("compare", Some(m)) => compare::handle(params, OutputParameters::from_clap(m)?),

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
