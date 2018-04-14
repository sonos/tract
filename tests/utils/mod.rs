use std::path;

use tfdeploy;
use tfdeploy::Matrix;
use tfdeploy::errors::*;

use itertools::Itertools;

#[allow(dead_code)]
pub fn run_both(
    tf: &mut ::tfdeploy::tf::Tensorflow,
    state: &mut ::tfdeploy::ModelState,
    inputs: Vec<(&str, Matrix)>,
    output: &str,
) -> Result<(Vec<Matrix>, Vec<Matrix>)> {
    let rtf = tf.run(inputs.clone(), output);
    if let Err(ref e) = rtf {
        if e.description().contains("String vs") {
            println!("Ignore output named (is a string) {}", output);
            return Err(ErrorKind::TFString.into());
        }
    }
    let rtf = rtf?;
    let inputs = inputs
        .into_iter()
        .map(|(s, t)| (state.model().node_id_by_name(s).unwrap(), t))
        .collect();
    let output = state.model().node_id_by_name(output)?;
    let rtfd = state.run(inputs, output)?;
    Ok((rtf, rtfd))
}

pub fn compare_outputs(rtf:&Vec<Matrix>, rtfd:&Vec<Matrix>) -> Result<()> {
    if rtf.len() != rtfd.len() {
        Err(format!(
            "number of output differ tf:{} tfd:{}",
            rtf.len(),
            rtfd.len()
        ))?
    }
    for (ix, (mtf, mtfd)) in rtf.iter().zip(rtfd.iter()).enumerate() {
        if mtf.shape().len() != 0 && mtf.shape() != mtfd.shape() {
            Err(format!(
                "Shape mismatch, output:{} tf:{:?} tfd:{:?}",
                ix,
                mtf.shape(),
                mtfd.shape()
            ))?
        } else {
            if !mtf.close_enough(&mtfd) {
                Err("data mismatch")?
            }
        }
    }
    Ok(())
}

#[allow(dead_code)]
pub fn compare_all<P: AsRef<path::Path>>(
    model: P,
    inputs: Vec<(&str, Matrix)>,
    output_name: &str,
) -> Result<()> {
    use colored::Colorize;
    let mut tf = tfdeploy::tf::for_path(&model)?;
    let tfd = tfdeploy::for_path(&model)?;
    let mut state = tfd.state();
    let output_node = tfd.get_node(output_name)?;
    for node in output_node.eval_order(&tfd)? {
        let node = &tfd.nodes()[node];
        if node.op_name == "Placeholder" {
            println!(" * skipping Placeholder `{}'", node.name);
            continue;
        }
        let (rtf, rtfd) = run_both(&mut tf, &mut state, inputs.clone(), &* node.name)?;
        match compare_outputs(&rtf, &rtfd) {
            Err(Error(ErrorKind::TFString, _)) => continue,
            Err(e) => {
                println!("name: {}", node.name.red());
                println!("op: {}", node.op_name);
                let graph = tfdeploy::Model::graphdef_for_path(&model)?;
                let gnode = graph.get_node().iter().find(|n| n.get_name() == node.name).unwrap();
                for attr in gnode.get_attr() {
                    println!("- attr:{} {:?}", attr.0, attr.1);
                }
                println!("");
                for (ix, &(n, i)) in node.inputs.iter().enumerate() {
                    let data = &state.outputs[n].as_ref().unwrap()[i.unwrap_or(0)];
                    println!("{}", format!("INPUT {}", ix).bold());
                    println!("  {}", data.partial_dump(true).unwrap());
                }
                for (ix, pair) in rtf.iter().zip_longest(rtfd.iter()).enumerate() {
                    match pair {
                        ::itertools::EitherOrBoth::Both(mtf,mtfd) => {
                            println!("{}", format!("OUTPUT {}", ix).bold());
                            let tfd = if mtf.shape() != mtfd.shape() {"TFD".red()} else if mtf.close_enough(mtfd) { "TFD".green() } else { "TFD".yellow() };
                            println!("  TF {}", mtf.partial_dump(true).unwrap());
                            println!("  {} {}", tfd, mtfd.partial_dump(true).unwrap());
                        },
                        ::itertools::EitherOrBoth::Left(mtf) => {
                            println!("  TF {}", mtf.partial_dump(true).unwrap());
                            println!("{}", "  TFD MISSING".red());
                        },
                        ::itertools::EitherOrBoth::Right(mtfd) => {
                            println!("  TF UNEXPECTED {}", mtfd.partial_dump(true).unwrap());
                        }
                    }
                }
                Err(e)?
            }
            Ok(_) => {
                println!(" * {} ({})", node.name.green(), node.op_name);
                state.set_outputs(node.id, rtf)?
            }
        }
    }
    Ok(())
}

