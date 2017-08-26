#![allow(dead_code)]

#[macro_use]
extern crate error_chain;
extern crate flate2;
extern crate jpeg_decoder;
extern crate ndarray;
#[macro_use]
extern crate proptest;
extern crate protobuf;
extern crate reqwest;
extern crate tar;
extern crate tensorflow;
extern crate tfdeploy;

use std::path;

mod errors;
mod imagenet;
mod prop;
mod tf;
mod tfd;

use tfdeploy::Matrix;
use errors::*;

#[derive(Clone, Copy, PartialEq, PartialOrd, Default, Debug)]
pub struct SaneF32(pub f32);
impl ::std::cmp::Eq for SaneF32 {}
impl ::std::cmp::Ord for SaneF32 {
    fn cmp(&self, other: &SaneF32) -> ::std::cmp::Ordering {
        self.0.partial_cmp(&other.0).unwrap()
    }
}

pub trait TfExecutor {
    fn run(&mut self, inputs: Vec<(&str, Matrix)>, output_name: &str) -> Result<Vec<Matrix>>;
}

fn compare(
    tf: &mut tf::Tensorflow,
    tfd: &mut tfd::TfDeploy,
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
            //            println!("comparing {:?} - {:?}", mtf.datatype(), mtfd.datatype());
            let eq = match (mtf, &mtfd) {
                (&Matrix::U8(ref tf), &Matrix::U8(ref tfd)) => {
                    let max = tf.iter()
                        .zip(tfd.iter())
                        .map(|(&a, &b)| (a as isize - b as isize).abs())
                        .max()
                        .unwrap();
                    max < 10
                }
                (&Matrix::F32(ref tf), &Matrix::F32(ref tfd)) => {
                    let avg = tf.iter().map(|&a| a.abs()).sum::<f32>() / tf.len() as f32;
                    let dev = (tf.iter().map(|&a| (a - avg).powi(2)).sum::<f32>() /
                        tf.len() as f32)
                        .sqrt();
                    tf.iter().zip(tfd.iter()).all(|(&a, &b)| {
                        (b - a).abs() <= dev / 10.0
                    })
                }
                (&Matrix::I32(ref tf), &Matrix::I32(ref tfd)) => {
                    tf.iter().zip(tfd.iter()).all(|(&a, &b)| {
                        (a as isize - b as isize).abs() < 10
                    })
                }
                _ => unimplemented!(),
            };
            if !eq {
                println!("\n\n\n#### TENSORFLOW ####\n\n\n{:?}", mtf);
                println!("\n\n\n#### TFDEPLOY ####\n\n\n{:?}", mtfd);
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
    let mut tf = tf::for_path(&model)?;
    let mut tfd = tfd::for_path(&model)?;
    compare(&mut tf, &mut tfd, inputs, output_name)?;
    Ok(())
}

fn compare_all<P: AsRef<path::Path>>(
    model: P,
    inputs: Vec<(&str, Matrix)>,
    output_name: &str,
) -> Result<()> {
    let mut tf = tf::for_path(&model)?;
    let mut tfd = tfd::for_path(&model)?;
    println!("{:?}", tfd.graph.node_names());
    let output_node = tfd.graph.get_node(output_name)?;
    for node in output_node.eval_order()? {
        if node.op_name == "Placeholder" {
            println!(" * skipping Placeholder `{}'", node.name);
            continue;
        }
        println!(" * comparing outputs for {} ({})", node.name, node.op_name);
        for (k, v) in tfd.graph.get_pbnode(&*node.name)?.get_attr() {
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
                    println!("input #{}\n{:?}", ix, i);
                }
                Err(e)?
            }
            Ok(it) => tfd.graph.set_outputs(&*node.name, it)?,
        }
    }
    Ok(())
}


