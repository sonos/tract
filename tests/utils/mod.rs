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

pub fn compare_outputs(rtf: &Vec<Matrix>, rtfd: &Vec<Matrix>) -> Result<()> {
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
    let mut errors = 0;
    for ix in output_node.eval_order(&tfd)? {
        let node = &tfd.nodes()[ix];
        if node.op_name == "Placeholder" {
            println!(" * skipping Placeholder `{}'", node.name);
            continue;
        }
        let (rtf, rtfd) = match run_both(&mut tf, &mut state, inputs.clone(), &*node.name) {
            Ok((a, b)) => (a, b),
            Err(e) => {
                println!("{:3} {}", ix, node.name.red());
                dump_node(node, &model, &state, &[])?;
                Err(e)?
            }
        };
        match compare_outputs(&rtf, &rtfd) {
            Err(Error(ErrorKind::TFString, _)) => continue,
            Err(e) => {
                println!("{:3} {}", ix, node.name.yellow());
                dump_node(node, &model, &state, &rtf)?;
                for (ix, data) in rtfd.iter().enumerate() {
                    if ix >= rtf.len() {
                        println!("{} {}", format!("   TFD {}", ix).red().bold(), data.partial_dump(true).unwrap())
                    } else {
                        if rtf[ix].shape() != data.shape() {
                            println!("{} {}", format!("   TFD {}", ix).red().bold(), data.partial_dump(true).unwrap())
                        } else if !rtf[ix].close_enough(data) {
                            println!("{} {}", format!("   TFD {}", ix).yellow(), data.partial_dump(true).unwrap())
                        } else {
                            println!("{} {}", format!("   TFD {}", ix).green().yellow(), data.partial_dump(true).unwrap())
                        }
                    }
                }
                panic!("KABOUM");
                errors += 1
            }
            Ok(_) => {
                println!("{:3} {}", ix, node.name.green());
                dump_node(node, &model, &state, &*rtf);
            }
        }
        state.set_outputs(node.id, rtf)?;
        println!("");
    }
    if errors != 0 { Err(format!("{} errors", errors).into()) } else { Ok(()) }
}

fn dump_node<P: AsRef<path::Path>>(
    node: &tfdeploy::Node,
    model: P,
    state: &::tfdeploy::ModelState,
    output: &[Matrix],
) -> Result<()> {
    use colored::Colorize;
    println!("  {}  ", node.op_name.blue().bold());
    let graph = tfdeploy::Model::graphdef_for_path(model)?;
    let gnode = graph
        .get_node()
        .iter()
        .find(|n| n.get_name() == node.name)
        .unwrap();
    for attr in gnode.get_attr() {
        if attr.1.has_tensor() {
            println!("    {} -> {:?}", attr.0.bold(), attr.1.get_shape())
        } else {
            println!("    {} -> {:?}", attr.0.bold(), attr.1)
        }
    }
    for (ix, &(n, i)) in node.inputs.iter().enumerate() {
        let data = &state.outputs[n].as_ref().unwrap()[i.unwrap_or(0)];
        println!("{} {}/{} {}", format!(" INPUT {}", ix).bold(), n, i.unwrap_or(0), data.partial_dump(true).unwrap());
    }
    for (ix, data) in output.iter().enumerate() {
        println!("{} {}", format!("OUTPUT {}", ix).bold(), data.partial_dump(true).unwrap());
    }
    Ok(())
}
