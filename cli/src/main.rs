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
extern crate tfdeploy_tf;
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
use tfdeploy::analyser::TensorFact;
use tfdeploy_tf::tfpb;
use tfdeploy::*;
use tfdeploy::{DatumType, Tensor};
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
mod optimize_check;
mod profile;
mod prune;
mod rusage;
mod stream_check;
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

/// Structure holding the parsed parameters.
pub struct Parameters {
    name: String,
    graph: GraphDef,
    tfd_model: tfdeploy::Model,

    #[cfg(feature = "tensorflow")]
    tf_model: conform::tf::Tensorflow,

    input: Option<InputParameters>,
    input_nodes: Vec<String>,
    output_node: String,
}

impl Parameters {
    /// Parses the command-line arguments.
    pub fn from_clap(matches: &clap::ArgMatches) -> CliResult<Parameters> {
        let name = matches.value_of("model").unwrap();
        let graph = tfdeploy_tf::model::graphdef_for_path(&name)?;
        let tfd_model = tfdeploy_tf::for_path(&name)?;

        #[cfg(feature = "tensorflow")]
        let tf_model = conform::tf::for_path(&name)?;

        let input = InputParameters::from_clap(matches)?;

        let input_nodes = if let Some(inputs) = matches.values_of("inputs") {
            for input in inputs.clone() {
                let _ = tfd_model.node_by_name(&input)?;
            }
            inputs.map(|s| s.to_string()).collect()
        } else {
            tfd_model.guess_inputs()
                .iter()
                .map(|n| n.name.to_string())
                .collect()
        };

        let mut output_nodes: Vec<String> = if let Some(outputs) = matches.values_of("outputs") {
            for output in outputs.clone() {
                let _ = tfd_model.node_by_name(&output)?;
            }
            outputs.map(|s| s.to_string()).collect()
        } else {
            tfd_model.guess_outputs()
                .iter()
                .map(|n| n.name.to_string())
                .collect()
        };

        #[cfg(feature = "tensorflow")]
        return Ok(Parameters {
            name: name.to_string(),
            graph,
            tfd_model,
            tf_model,
            input_nodes,
            output_node: output_nodes.remove(0),
            input,
        });

        #[cfg(not(feature = "tensorflow"))]
        return Ok(Parameters {
            name: name.to_string(),
            graph,
            tfd_model,
            input_nodes,
            output_node: output_nodes.remove(0),
            input,
        });
    }
}

/// Structure holding the input parameters (eventually containing data).
#[derive(Debug, Clone, PartialEq)]
pub struct InputParameters {
    data: Option<Tensor>,
    shape: Vec<Option<usize>>,
    datum_type: DatumType,
}

impl InputParameters {
    fn from_clap(matches: &clap::ArgMatches) -> CliResult<Option<InputParameters>> {
        let input = match (matches.value_of("size"), matches.value_of("data")) {
            (_, Some(filename)) => Some(Self::for_data(filename)?),
            (Some(size), _) => Some(Self::for_size(size)?),
            _ => None,
        };
        Ok(input)
    }

    fn for_size(size: &str) -> CliResult<InputParameters> {
        let splits = size.split("x").collect::<Vec<_>>();

        if splits.len() < 1 {
            bail!("The <size> argument should be formatted as {size}x{...}x{type}.");
        }

        let (datum_type, shape) = splits.split_last().unwrap();

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

        let datum_type = match datum_type.to_lowercase().as_str() {
            "f64" => DatumType::F64,
            "f32" => DatumType::F32,
            "i32" => DatumType::I32,
            "i8" => DatumType::I8,
            "u8" => DatumType::U8,
            _ => bail!("Type of the input should be f64, f32, i32, i8 or u8."),
        };

        Ok(InputParameters {
            data: None,
            shape,
            datum_type,
        })
    }

    /// Parses the `data` command-line argument.
    fn for_data(filename: &str) -> CliResult<InputParameters> {
        let mut file = File::open(filename)?;
        let mut data = String::new();
        file.read_to_string(&mut data)?;

        let mut lines = data.lines();
        let InputParameters {
            shape, datum_type, ..
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

        let tensor = match datum_type {
            DatumType::F64 => for_type!(f64).into(),
            DatumType::F32 => for_type!(f32).into(),
            DatumType::I32 => for_type!(i32).into(),
            DatumType::I8 => for_type!(i8).into(),
            DatumType::U8 => for_type!(u8).into(),
            _ => unimplemented!(),
        };

        Ok(InputParameters {
            data: Some(tensor),
            shape,
            datum_type,
        })
    }

    fn streaming(&self) -> bool {
        self.shape.iter().any(|dim| dim.is_none())
    }

    fn to_fact(&self) -> TensorFact {
        if let Some(ref data) = self.data {
            return data.clone().into();
        }
        let dims = self
            .shape
            .iter()
            .map(|d| d.map(|i| i.into()).unwrap_or(TDim::stream()).into())
            .collect::<Vec<_>>();
        TensorFact {
            datum_type: typefact!(self.datum_type),
            shape: tfdeploy::analyser::ShapeFact::closed(dims.clone()),
            value: valuefact!(_),
        }
    }

    fn to_tensor(&self) -> CliResult<Tensor> {
        self.to_tensor_with_stream_dim(None)
    }

    fn to_tensor_with_stream_dim(&self, streaming_dim: Option<usize>) -> CliResult<Tensor> {
        if let Some(value) = self.data.as_ref() {
            Ok(value.clone())
        } else {
            if self.streaming() && streaming_dim.is_none() {
                Err("random tensor requires a streaming dim")?
            }
            Ok(utils::random_tensor(
                self.shape
                    .iter()
                    .map(|d| d.or(streaming_dim).unwrap())
                    .collect(),
                self.datum_type,
            ))
        }
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
    let streaming = params
        .input
        .as_ref()
        .map(|i| i.streaming())
        .unwrap_or(false);

    match matches.subcommand() {
        ("compare", Some(m)) => compare::handle(params, OutputParameters::from_clap(m)?),

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
