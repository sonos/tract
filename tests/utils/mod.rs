use std::path;

use tfdeploy;
use tfdeploy::Matrix;
use tfdeploy::errors::*;

use itertools::Itertools;

pub fn compare_outputs<M2: ::std::borrow::Borrow<Matrix>>(rtf: &Vec<Matrix>, rtfd: &[M2]) -> Result<()> {
    if rtf.len() != rtfd.len() {
        Err(format!(
            "number of output differ tf:{} tfd:{}",
            rtf.len(),
            rtfd.len()
        ))?
    }
    for (ix, (mtf, mtfd)) in rtf.iter().zip(rtfd.iter()).enumerate() {
        if mtf.shape().len() != 0 && mtf.shape() != mtfd.borrow().shape() {
            Err(format!(
                "Shape mismatch, output:{} tf:{:?} tfd:{:?}",
                ix,
                mtf.shape(),
                mtfd.borrow().shape()
            ))?
        } else {
            if !mtf.close_enough(mtfd.borrow()) {
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
    for i in inputs.iter() {
        let node_id = state.model().node_id_by_name(i.0)?;
        state.set_outputs(node_id, vec!(i.1.clone()))?;
    }
    for ix in output_node.eval_order(&tfd)? {
        let node = &tfd.nodes()[ix];
        if node.op_name == "Placeholder" {
            println!(" * skipping Placeholder `{}'", node.name);
            continue;
        }
        dump_node(node, &model, &state)?;
        let rtf = tf.run(inputs.clone(), &node.name);
        if let Err(ref e) = rtf {
            if e.description().contains("String vs") {
                println!("Ignore output named (is a string) {}", node.name);
                return Err(ErrorKind::TFString.into());
            }
        }
        let rtf = rtf?;
        dump_output(&rtf)?;
        if let Err(e) = state.compute_one(ix) {
            println!("\n{} {:?}\n", "ERROR".red().bold(), e);
            errors += 1
        } else {
            let rtfd = state.outputs[ix].as_ref().unwrap();
            let views = rtfd.iter().map(|m| &**m).collect::<Vec<&Matrix>>();
            match compare_outputs(&rtf, &views) {
                Err(Error(ErrorKind::TFString, _)) => continue,
                Err(e) => {
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
                    println!("\n{}", "MISMATCH".red().bold());
                    errors += 1
                }
                Ok(_) => {
                    println!("\n{}", "OK".green().bold());
                }
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
) -> Result<()> {
    use colored::Colorize;
    println!("{:3} {:20} {}\n", format!("{:3}", node.id).bold(), node.op_name.blue().bold(), node.name.bold());
    let graph = tfdeploy::Model::graphdef_for_path(model)?;
    let gnode = graph
        .get_node()
        .iter()
        .find(|n| n.get_name() == node.name)
        .unwrap();
    for attr in gnode.get_attr() {
        if attr.1.has_tensor() {
            println!("{:>20} Tensor of shape {:?}", attr.0.bold(), attr.1.get_shape())
        } else {
            println!("{:>20} {:?}", attr.0.bold(), attr.1)
        }
    }
    println!("");
    for (ix, &(n, i)) in node.inputs.iter().enumerate() {
        let data = &state.outputs[n].as_ref().unwrap()[i.unwrap_or(0)];
        println!("{} <{}/{}> {}", format!(" INPUT {}", ix).bold(), n, i.unwrap_or(0), data.partial_dump(true).unwrap());
    }
    Ok(())
}

fn dump_output(output: &[Matrix]) -> Result<()> {
    use colored::Colorize;
    for (ix, data) in output.iter().enumerate() {
        println!("{} {}", format!("OUTPUT {}", ix).bold(), data.partial_dump(true).unwrap());
    }
    Ok(())
}
