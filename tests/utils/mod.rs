use std::path;

use tfdeploy;
use tfdeploy::Matrix;
use tfdeploy::errors::*;

#[allow(dead_code)]
pub fn compare(
    tf: &mut ::tfdeploy::tf::Tensorflow,
    state: &mut ::tfdeploy::ModelState,
    inputs: Vec<(&str, Matrix)>,
    output: &str,
) -> Result<Vec<Matrix>> {
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
    if rtf.len() != rtfd.len() {
        Err(format!(
            "number of output differ tf:{} tfd:{}",
            rtf.len(),
            rtfd.len()
        ))?
    }
    for (ix, (mtf, mtfd)) in rtf.iter().zip(rtfd.into_iter()).enumerate() {
        if mtf.shape().len() != 0 && mtf.shape() != mtfd.shape() {
            Err(format!(
                "Shape mismatch, output:{} tf:{:?} tfd:{:?}",
                ix,
                mtf.shape(),
                mtfd.shape()
            ))?
        } else {
            if !mtf.close_enough(&mtfd) {
                println!(
                    "\n\n\n#### TENSORFLOW ####\n\n\n{:?}",
                    mtf.partial_dump(false)
                );
                println!(
                    "\n\n\n#### TFDEPLOY ####\n\n\n{:?}",
                    mtfd.partial_dump(false)
                );
                Err("data mismatch")?
            }
        }
    }
    Ok(rtf)
}

#[allow(dead_code)]
pub fn compare_one<P: AsRef<path::Path>>(
    model: P,
    inputs: Vec<(&str, Matrix)>,
    output_name: &str,
) -> Result<()> {
    let mut tf = ::tfdeploy::tf::for_path(&model)?;
    let tfd = tfdeploy::for_path(&model)?;
    compare(&mut tf, &mut tfd.state(), inputs, output_name)?;
    Ok(())
}

#[allow(dead_code)]
pub fn compare_all<P: AsRef<path::Path>>(
    model: P,
    inputs: Vec<(&str, Matrix)>,
    output_name: &str,
) -> Result<()> {
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
        println!(" * comparing outputs for {} ({})", node.name, node.op_name);
        match compare(&mut tf, &mut state, inputs.clone(), &*node.name) {
            Err(Error(ErrorKind::TFString, _)) => continue,
            Err(e) => {
                println!("error !");
                for (ix, &(_, ref i)) in inputs.iter().enumerate() {
                    println!("input #{}\n{:?}", ix, i.partial_dump(true));
                }
                Err(e)?
            }
            Ok(it) => state.set_outputs(node.id, it)?,
        }
    }
    Ok(())
}

