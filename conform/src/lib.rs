#![allow(dead_code)]

extern crate flate2;
extern crate jpeg_decoder;
extern crate ndarray;
extern crate protobuf;
extern crate reqwest;
extern crate tar;
#[cfg(features="tensorflow")]
extern crate tensorflow;
extern crate tfdeploy;

use std::path;

mod imagenet;

use tfdeploy::Matrix;
use tfdeploy::errors::*;

#[derive(Clone, Copy, PartialEq, PartialOrd, Default, Debug)]
pub struct SaneF32(pub f32);
impl ::std::cmp::Eq for SaneF32 {}
impl ::std::cmp::Ord for SaneF32 {
    fn cmp(&self, other: &SaneF32) -> ::std::cmp::Ordering {
        self.0.partial_cmp(&other.0).unwrap()
    }
}

fn compare(
    tf: &mut ::tfdeploy::tf::Tensorflow,
    tfd: &mut ::tfdeploy::GraphAnalyser,
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
    let rtfd = tfd.run(inputs.clone(), output_name)?;
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
                println!("\n\n\n#### TENSORFLOW ####\n\n\n{:?}", mtf.partial_dump(false));
                println!("\n\n\n#### TFDEPLOY ####\n\n\n{:?}", mtfd.partial_dump(false));
                Err("data mismatch")?
            }
        }
    }
    Ok(rtf)
}

fn compare_one<P: AsRef<path::Path>>(
    model: P,
    inputs: Vec<(&str, Matrix)>,
    output_name: &str,
) -> Result<()> {
    let mut tf = ::tfdeploy::tf::for_path(&model)?;
    let mut tfd = tfdeploy::for_path(&model)?;
    compare(&mut tf, &mut tfd, inputs, output_name)?;
    Ok(())
}

fn compare_all<P: AsRef<path::Path>>(
    model: P,
    inputs: Vec<(&str, Matrix)>,
    output_name: &str,
) -> Result<()> {
    let mut tf = tfdeploy::tf::for_path(&model)?;
    let mut tfd = tfdeploy::for_path(&model)?;
    println!("{:?}", tfd.node_names());
    let output_node = tfd.get_node(output_name)?;
    for node in output_node.eval_order()? {
        if node.op_name == "Placeholder" {
            println!(" * skipping Placeholder `{}'", node.name);
            continue;
        }
        println!(" * comparing outputs for {} ({})", node.name, node.op_name);
        for (k, v) in tfd.get_pbnode(&*node.name)?.get_attr() {
            if v.has_tensor() {
                println!("     {}:tensor", k);
            } else {
                println!("     {}:{:?}", k, v);
            }
        }
        match compare(&mut tf, &mut tfd, inputs.clone(), &*node.name) {
            Err(Error(ErrorKind::TFString, _)) => continue,
            Err(e) => {
                println!("error !");
                for (ix, &(_, ref i)) in inputs.iter().enumerate() {
                    println!("input #{}\n{:?}", ix, i.partial_dump(true));
                }
                Err(e)?
            }
            Ok(it) => tfd.set_outputs(&*node.name, it)?,
        }
    }
    Ok(())
}


