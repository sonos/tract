#![allow(dead_code)]

#[macro_use]
extern crate error_chain;
extern crate flate2;
extern crate jpeg_decoder;
extern crate ndarray;
extern crate reqwest;
extern crate tar;
extern crate tensorflow;
extern crate tfdeploy;

use std::path;

mod errors;
mod imagenet;
mod tf;
mod tfd;

use tfdeploy::Matrix;
use errors::*;

pub trait TfExecutor {
    fn run(&mut self, inputs: Vec<(&str, Matrix)>, output_name: &str) -> Result<Vec<Matrix>>;
}

fn compare<P: AsRef<path::Path>>(
    model: P,
    inputs: Vec<(&str, Matrix)>,
    output_name: &str,
) -> Result<()> {
    let tf = tf::build(&model)?.run(inputs.clone(), output_name)?;
    let tfd = tfd::build(&model)?.run(inputs.clone(), output_name)?;
    for (mtf, mtfd) in tf.into_iter().zip(tfd.into_iter()) {
        if mtf.shape() != mtfd.shape() {
            Err(format!("tf:{:?}\ntfd:{:?}", mtf.shape(), mtfd.shape()))?
        } else {
            let eq = match (&mtf, &mtfd) {
                (&Matrix::U8(ref tf), &Matrix::U8(ref tfd)) => {
                    tf.iter().zip(tfd.iter()).all(|(&a, &b)| {
                        (a as isize - b as isize).abs() < 10
                    })
                }
                (&Matrix::F32(ref tf), &Matrix::F32(ref tfd)) => tf.all_close(&tfd, 0.01),
                _ => unimplemented!(),
            };
            if !eq {
                println!("\n\n\n#### TENSORFLOW ####\n\n\n{:?}", mtf);
                println!("\n\n\n#### TFDEPLOY ####\n\n\n{:?}", mtfd);
                Err("data mismatch")?
            }
        }
    }
    Ok(())
}

fn compare_all<P: AsRef<path::Path>>(
    model: P,
    inputs: Vec<(&str, Matrix)>,
    output_name: &str,
) -> Result<()> {
    let mut tfd = tfdeploy::GraphAnalyser::from_file(model)?;
    let names:Vec<String> = tfd.node_names().iter().map(|s| s.to_string()).collect();
    for n in names {
        let node = tfd.get_pbnode(&*n)?;
        println!(" - node: {} {} {:?}", n, node.get_op(), node.get_input());
    }
    let output_node = tfd.get_node(output_name)?;
    println!("\n\n final output node: {:?}", output_name);
    for node in output_node.eval_order()? {
        println!(" - {}", node.name);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {}
}
