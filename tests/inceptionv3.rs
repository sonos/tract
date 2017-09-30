#![cfg(feature="tensorflow")]
extern crate flate2;
extern crate image;
extern crate itertools;
extern crate ndarray;
extern crate reqwest;
extern crate tar;
extern crate tfdeploy;

#[path = "../examples/inceptionv3.rs"]
mod inceptionv3;

use std::path;

use tfdeploy::errors::*;

use std::error::Error;
use tfdeploy::Matrix;

#[derive(Clone, Copy, PartialEq, PartialOrd, Default, Debug)]
pub struct SaneF32(pub f32);
impl ::std::cmp::Eq for SaneF32 {}
impl ::std::cmp::Ord for SaneF32 {
    fn cmp(&self, other: &SaneF32) -> ::std::cmp::Ordering {
        self.0.partial_cmp(&other.0).unwrap()
    }
}

#[allow(dead_code)]
fn compare(
    tf: &mut ::tfdeploy::tf::Tensorflow,
    tfd: &::tfdeploy::Model,
    state: &mut ::tfdeploy::ModelState,
    inputs: Vec<(&str, Matrix)>,
    output_name: &str,
) -> Result<Vec<Matrix>> {
    let rtf = tf.run(inputs.clone(), output_name);
    if let Err(ref e) = rtf {
        if e.description().contains("String vs") {
            println!("Ignore output named (is a string) {}", output_name);
            return Err(ErrorKind::TFString.into());
        }
    }
    let rtf = rtf?;
    let rtfd = state.run(inputs.clone(), output_name)?;
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
fn compare_one<P: AsRef<path::Path>>(
    model: P,
    inputs: Vec<(&str, Matrix)>,
    output_name: &str,
) -> Result<()> {
    let mut tf = ::tfdeploy::tf::for_path(&model)?;
    let mut tfd = tfdeploy::for_path(&model)?;
    compare(&mut tf, &tfd, &mut tfd.state(), inputs, output_name)?;
    Ok(())
}

#[allow(dead_code)]
fn compare_all<P: AsRef<path::Path>>(
    model: P,
    inputs: Vec<(&str, Matrix)>,
    output_name: &str,
) -> Result<()> {
    let mut tf = tfdeploy::tf::for_path(&model)?;
    let tfd = tfdeploy::for_path(&model)?;
    let mut state = tfd.state();
    println!("{:?}", tfd.node_names());
    let output_node = tfd.get_node(output_name)?;
    for node in output_node.eval_order()? {
        if node.op_name == "Placeholder" {
            println!(" * skipping Placeholder `{}'", node.name);
            continue;
        }
        println!(" * comparing outputs for {} ({})", node.name, node.op_name);
        match compare(&mut tf, &tfd, &mut state, inputs.clone(), &*node.name) {
            Err(Error(ErrorKind::TFString, _)) => continue,
            Err(e) => {
                println!("error !");
                for (ix, &(_, ref i)) in inputs.iter().enumerate() {
                    println!("input #{}\n{:?}", ix, i.partial_dump(true));
                }
                Err(e)?
            }
            Ok(it) => state.set_outputs(&*node.name, it)?,
        }
    }
    Ok(())
}

#[test]
fn test_tf() {
    let mut tf = ::tfdeploy::tf::for_path(inceptionv3::INCEPTION_V3).unwrap();
    let input = inceptionv3::load_image(inceptionv3::HOPPER);
    let mut output = tf.run(vec![("input", input)], "InceptionV3/Predictions/Reshape_1")
        .unwrap();
    let labels = inceptionv3::load_labels();
    for (ix, c) in output.remove(0).take_f32s().unwrap().iter().enumerate() {
        if *c >= 0.01 {
            println!("{}: {} {}", ix, c, labels[ix]);
        }
    }
}

#[test]
fn test_compare_all() {
    inceptionv3::download().unwrap();

    ::compare_all(
        inceptionv3::INCEPTION_V3,
        vec![("input", inceptionv3::load_image(inceptionv3::HOPPER))],
        "InceptionV3/Predictions/Reshape_1",
    ).unwrap();
}
