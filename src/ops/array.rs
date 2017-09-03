use ndarray::prelude::*;

use {Matrix, Result};
use super::Op;

#[derive(Debug)]
pub struct ConcatV2 {
    n: usize,
}

impl ConcatV2 {
    pub fn build(pb: &::tfpb::node_def::NodeDef) -> Result<ConcatV2> {
        Ok(ConcatV2 {
            n: pb.get_attr().get("N").unwrap().get_i() as _,
        })
    }
}

impl Op for ConcatV2 {
    fn eval(&self, mut inputs: Vec<Matrix>) -> Result<Vec<Matrix>> {
        let axis: i32 = *inputs.remove(self.n).take_i32s().unwrap().into_iter().next().unwrap();
        let mats:Vec<ArrayD<f32>> = inputs.into_iter().map(|mat| mat.take_f32s().unwrap()).collect();
        let views:Vec<ArrayViewD<f32>> = mats.iter().map(|m| m.view()).collect();
        let result = ::ndarray::stack(Axis(axis as usize), &*views)?;
        Ok(vec!(result.into()))
    }
}

#[derive(Debug)]
pub struct Reshape {
}

impl Reshape {
    pub fn build(_pb: &::tfpb::node_def::NodeDef) -> Result<Reshape> {
        Ok(Reshape {
        })
    }
}

impl Op for Reshape {
    fn eval(&self, mut inputs: Vec<Matrix>) -> Result<Vec<Matrix>> {
        let mut dims:Vec<i32> = inputs.remove(1).take_i32s().unwrap().iter().cloned().collect();
        let input = inputs.remove(0).take_f32s().unwrap();
        if dims.contains(&-1) {
            let prod:i32 = dims.iter().map(|a| *a).filter(|a| *a != -1i32).product();
            for a in dims.iter_mut() {
                if *a == -1 {
                    *a = input.len() as i32 / prod;
                }
            }
        }
        let dims:Vec<usize> = dims.into_iter().map(|a| a as usize).collect();
        Ok(vec!(input.into_shape(&*dims)?.into_dyn().into()))
    }
}
