#[macro_use]
extern crate clap;
#[macro_use]
extern crate log;
#[macro_use]
extern crate error_chain;

#[cfg(feature="tensorflow")]
extern crate conform;

extern crate simplelog;
extern crate tfdeploy;
extern crate time;
extern crate rand;
extern crate ndarray;
extern crate colored;

use std::process::exit;
use std::path::Path;
use simplelog::{TermLogger, LevelFilter, Config};
use tfdeploy::tfpb;
use tfpb::types::DataType;
use time::PreciseTime;
use rand::Rng;

use tfdeploy::Matrix;


/// The default number of iterations for the profiler.
const DEFAULT_ITERS: usize = 100000;


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
    graph: tfpb::graph::GraphDef,
    tfd_model: tfdeploy::Model,

    #[cfg(feature="tensorflow")]
    tf_model: conform::tf::Tensorflow,

    inputs: Vec<String>,
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
            "Sets the TensorFlow model to use (in Protobuf format).")

        (@arg inputs: -i --input ... [input]
            "Sets the input nodes names (auto-detects otherwise).")

        (@arg output: -o --output [output]
            "Sets the output node name (auto-detects otherwise).")

        (@arg size: -s --size <size>
            "Sets the input size, e.g. 32x64xf32.")

        (@arg debug: -d ... "Sets the level of debugging information.")

        (@subcommand compare =>
            (about: "Compares the output of tfdeploy and tensorflow on randomly generated input."))

        (@subcommand profile =>
            (about: "Benchmarks tfdeploy on randomly generated input.")
            (@arg iters: -n [iters]
                "Sets the number of iterations for the average [default: 100000]."))
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
        ("compare", _) =>
            handle_compare(params),

        ("profile", None) =>
            handle_profile(params, DEFAULT_ITERS),

        ("profile", Some(m)) =>
            handle_profile(params, match m.value_of("iters") {
                None => DEFAULT_ITERS,
                Some(s) => s.parse::<usize>()?
            }),

        (s, _) => bail!("Unknown subcommand {}.", s)
    }
}


/// Parses the command-line arguments.
fn parse(matches: &clap::ArgMatches) -> Result<Parameters> {
    let path = Path::new(matches.value_of("model").unwrap());
    let graph = tfdeploy::Model::graphdef_for_path(&path)?;
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

    let inputs = match matches.values_of("inputs") {
        Some(names) => names.map(|s| s.to_string()).collect(),
        None => detect_inputs(&tfd_model)?
    };

    let output = match matches.value_of("output") {
        Some(name) => name.to_string(),
        None => detect_output(&tfd_model)?
    };

    #[cfg(feature="tensorflow")]
    return Ok(Parameters { graph, tfd_model, tf_model, inputs, output, size_x, size_y, size_d });

    #[cfg(not(feature="tensorflow"))]
    return Ok(Parameters { graph, tfd_model, inputs, output, size_x, size_y, size_d });
}


/// Tries to autodetect the names of the input nodes.
#[allow(unused_variables)]
fn detect_inputs(model: &tfdeploy::Model) -> Result<Vec<String>> {
    let mut inputs = Vec::new();

    for node in model.nodes() {
        if node.op_name == "Placeholder" {
            inputs.push(node.name.clone());
        }
    }

    if inputs.len() > 0 {
        info!("Autodetecting input nodes: {:?}.", inputs);
        Ok(inputs)
    } else {
        bail!("Impossible to auto-detect input nodes: no placeholder.");
    }
}


/// Tries to autodetect the name of the output node.
#[allow(unused_variables)]
fn detect_output(model: &tfdeploy::Model) -> Result<String> {
    // We search for the only node in the graph with no successor.
    let mut succs: Vec<Vec<usize>> = vec![Vec::new();  model.nodes().len()];

    for node in model.nodes() {
        for &link in &node.inputs {
            succs[link.0].push(node.id);
        }
    }

    for (i, s) in succs.iter().enumerate() {
        if s.len() == 0 {
            let output = model.get_node_by_id(i)?.name.clone();
            info!("Autodetecting output node: {:?}.", output);

            return Ok(output);
        }
    }

    bail!("Impossible to auto-detect output nodes.")
}


/// Prints information about a node.
fn dump_node(
    node: &tfdeploy::Node,
    graph: &tfpb::graph::GraphDef,
    state: &::tfdeploy::ModelState,
) -> Result<()> {
    use colored::Colorize;
    println!(
        "{:3} {:20} {}\n",
        format!("{:3}", node.id).bold(),
        node.op_name.blue().bold(),
        node.name.bold()
    );
    let gnode = graph
        .get_node()
        .iter()
        .find(|n| n.get_name() == node.name)
        .unwrap();
    for attr in gnode.get_attr() {
        if attr.1.has_tensor() {
            println!(
                "{:>20} Tensor of shape {:?}",
                attr.0.bold(),
                attr.1.get_shape()
            )
        } else {
            println!("{:>20} {:?}", attr.0.bold(), attr.1)
        }
    }
    println!("");
    for (ix, &(n, i)) in node.inputs.iter().enumerate() {
        let data = &state.outputs[n].as_ref().unwrap()[i.unwrap_or(0)];
        println!(
            "{} <{}/{}> {}",
            format!(" INPUT {}", ix).bold(),
            n,
            i.unwrap_or(0),
            data.partial_dump(true).unwrap()
        );
    }
    Ok(())
}


/// Print a colored dump of a Matrix.
fn dump_output(output: &[Matrix]) -> Result<()> {
    use colored::Colorize;
    for (ix, data) in output.iter().enumerate() {
        println!(
            "{} {}",
            format!("OUTPUT {}", ix).bold(),
            data.partial_dump(true).unwrap()
        );
    }
    Ok(())
}


/// Handles the `compare` subcommand.
#[allow(unused_variables)]
#[cfg(not(feature="tensorflow"))]
fn handle_compare(params: Parameters) -> Result<()> {
    bail!("Comparison requires the `tensorflow` feature.")
}

#[allow(unused_mut)]
#[allow(unused_imports)]
#[allow(unused_variables)]
#[cfg(feature="tensorflow")]
fn handle_compare(params: Parameters) -> Result<()> {
    use colored::Colorize;

    let tfd = params.tfd_model;
    let mut tf = params.tf_model;

    let output = tfd.get_node(params.output.as_str())?;
    let mut state = tfd.state();
    let mut errors = 0;

    // First generate random values for the inputs.
    let mut generated = Vec::new();
    for s in &params.inputs {
        generated.push((
            s.as_str(),
            random_matrix(params.size_x, params.size_y, params.size_d)
        ));
    }

    // Execute the model on tensorflow first.
    info!("Running the model on tensorflow.");
    let mut tf_outputs = tf.run_get_all(generated.clone())?;

    // Execute the model step-by-step on tfdeploy.
    state.set_values(generated)?;
    let plan = output.eval_order(&tfd)?;
    info!("Using execution plan: {:?}", plan);

    for n in plan {
        let node = tfd.get_node_by_id(n)?;

        if node.op_name == "Placeholder" {
            println!(" * Skipping placeholder: {}", node.name);
            continue;
        }

        dump_node(node, &params.graph, &state)?;

        let rtf = tf_outputs
            .remove(&node.name.to_string())
            .expect(format!("No node with name {} was computed by tensorflow.", node.name).as_str());

        // if let Err(ref e) = rtf {
        //     if e.description().contains("String vs") {
        //         println!(" * Skipping string: {}", node.name);
        //         continue;
        //     }
        // }
        // let rtf = rtf?;

        dump_output(&rtf)?;

        if let Err(e) = state.compute_one(n) {
            println!("\n{} {:?}\n", "ERROR".red().bold(), e);
            errors += 1;
        } else {
            let rtfd = state.outputs[n].as_ref().unwrap();
            let views = rtfd.iter().map(|m| &**m).collect::<Vec<&Matrix>>();
            match compare_outputs(&rtf, &views) {
                Err(e) => {
                    for (n, data) in rtfd.iter().enumerate() {
                        if n >= rtf.len() {
                            println!(
                                "{} {}",
                                format!("   TFD {}", n).red().bold(),
                                data.partial_dump(true).unwrap()
                            )
                        } else {
                            if rtf[n].shape() != data.shape() {
                                println!(
                                    "{} {}",
                                    format!("   TFD {}", n).red().bold(),
                                    data.partial_dump(true).unwrap()
                                )
                            } else if !rtf[n].close_enough(data) {
                                println!(
                                    "{} {}",
                                    format!("   TFD {}", n).yellow(),
                                    data.partial_dump(true).unwrap()
                                )
                            } else {
                                println!(
                                    "{} {}",
                                    format!("   TFD {}", n).green().yellow(),
                                    data.partial_dump(true).unwrap()
                                )
                            }
                        }
                    }
                    println!("\n{}", "MISMATCH".red().bold());
                    errors += 1
                }
                Ok(_) => {
                    println!("\n{}", "OK".green().bold());
                }
            }
        }

        // Re-use the output from tensorflow to keep tfdeploy from drifting.
        state.set_outputs(node.id, rtf)?;
        println!("");
    }

    if errors != 0 {
        bail!("{} errors", errors)
    } else {
        Ok(())
    }
}


/// Compares the outputs of a node in tfdeploy and tensorflow.
#[cfg(feature="tensorflow")]
fn compare_outputs<M2: ::std::borrow::Borrow<Matrix>>(
    rtf: &Vec<Matrix>,
    rtfd: &[M2],
) -> Result<()> {
    if rtf.len() != rtfd.len() {
        bail!(
            "Number of output differ: tf={}, tfd={}",
            rtf.len(),
            rtfd.len()
        )
    }

    for (ix, (mtf, mtfd)) in rtf.iter().zip(rtfd.iter()).enumerate() {
        if mtf.shape().len() != 0 && mtf.shape() != mtfd.borrow().shape() {
            bail!(
                "Shape mismatch for output {}: tf={:?}, tfd={:?}",
                ix,
                mtf.shape(),
                mtfd.borrow().shape()
            )
        } else {
            if !mtf.close_enough(mtfd.borrow()) {
                bail!(
                    "Data mismatch: tf={:?}, tfd={:?}",
                    mtf,
                    mtfd.borrow()
                )
            }
        }
    }

    Ok(())
}


/// Handles the `profile` subcommand.
fn handle_profile(params: Parameters, iters: usize) -> Result<()> {
    let model = params.tfd_model;
    let output = model.get_node(params.output.as_str())?;
    let mut state = model.state();

    // First fill the input with randomly generated values.
    for s in params.inputs {
        let input = model.get_node(s.as_str())?;
        state.set_value(
            input.id,
            random_matrix(params.size_x, params.size_y, params.size_d)
        )?;
    }

    let plan = output.eval_order(&model)?;
    info!("Using execution plan: {:?}.", plan);
    info!("Running {} iterations at each step.", iters);

    // Then execute the plan while profiling each step.
    for n in plan {
        if !state.outputs[n].is_none() {
            continue;
        }

        let node = model.get_node_by_id(n)?;
        let start = PreciseTime::now();
        for _ in 0..iters { state.compute_one(n)?; }
        let end = PreciseTime::now();

        let header = format!("{} ({}, {}):", n, node.name, node.op_name);
        println!(
            "- Node {:<20} {} ms on average.", header,
            start.to(end).num_milliseconds() as f64 / iters as f64
        );
    }

    Ok(())
}


/// Generates a random matrix of a given size and type.
fn random_matrix(x: usize, y: usize, d: DataType) -> Matrix {
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