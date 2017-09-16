use ndarray::prelude::*;

use {Matrix, Result};
use super::{ Op, OpRegister };

pub fn register_all_ops(reg: &mut OpRegister) {
    reg.insert("ConcatV2", ConcatV2::build);
    reg.insert("ExpandDims", ExpandDims::build);
    reg.insert("Identity", Identity::build);
    reg.insert("Placeholder", Placeholder::build);
    reg.insert("Squeeze", Squeeze::build);
    reg.insert("Reshape", Reshape::build);
}

#[derive(Debug)]
pub struct ConcatV2 {
    n: usize,
}

impl ConcatV2 {
    pub fn build(pb: &::tfpb::node_def::NodeDef) -> Result<Box<Op>> {
        Ok(Box::new(ConcatV2 { n: pb.get_attr().get("N").unwrap().get_i() as _ }))
    }
}

impl Op for ConcatV2 {
    fn eval(&self, mut inputs: Vec<Matrix>) -> Result<Vec<Matrix>> {
        let axis: i32 = *inputs
            .remove(self.n)
            .take_i32s()
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        let mats: Vec<ArrayD<f32>> = inputs
            .into_iter()
            .map(|mat| mat.take_f32s().unwrap())
            .collect();
        let views: Vec<ArrayViewD<f32>> = mats.iter().map(|m| m.view()).collect();
        let result = ::ndarray::stack(Axis(axis as usize), &*views)?;
        Ok(vec![result.into()])
    }
}

#[derive(Debug)]
pub struct ExpandDims;

impl ExpandDims {
    pub fn build(_pb: &::tfpb::node_def::NodeDef) -> Result<Box<Op>> {
        Ok(Box::new(ExpandDims))
    }
}

impl Op for ExpandDims {
    fn eval(&self, mut inputs: Vec<Matrix>) -> Result<Vec<Matrix>> {
        let dims = inputs.remove(1).take_i32s().ok_or(
            "Expect input #1 to be i32",
        )?;
        let data = inputs.remove(0).take_f32s().ok_or(
            "Expect input #0 to be f32",
        )?;
        let mut shape = data.shape().to_vec();
        for d in &dims {
            if *d >= 0 {
                shape.insert(*d as usize, 1);
            } else {
                Err(format!("unimplemented ExpandDims with negative parameter"))?
            }
        }
        Ok(vec![data.into_shape(shape)?.into()])
    }
}

#[derive(Debug)]
pub struct Identity;

impl Identity {
    pub fn build(_: &::tfpb::node_def::NodeDef) -> Result<Box<Op>> {
        Ok(Box::new(Identity))
    }
}

impl Op for Identity {
    fn eval(&self, inputs: Vec<Matrix>) -> Result<Vec<Matrix>> {
        Ok(inputs)
    }
}

#[derive(Debug)]
pub struct Placeholder;

impl Placeholder {
    pub fn build(_pb: &::tfpb::node_def::NodeDef) -> Result<Box<Op>> {
        Ok(Box::new(Placeholder))
    }
}

impl Op for Placeholder {
    fn eval(&self, _inputs: Vec<Matrix>) -> Result<Vec<Matrix>> {
        panic!("Placeholder should not get evaluated")
    }
}

#[derive(Debug)]
pub struct Reshape {}

impl Reshape {
    pub fn build(_pb: &::tfpb::node_def::NodeDef) -> Result<Box<Op>> {
        Ok(Box::new(Reshape {}))
    }
}

impl Op for Reshape {
    fn eval(&self, mut inputs: Vec<Matrix>) -> Result<Vec<Matrix>> {
        let mut dims: Vec<i32> = inputs
            .remove(1)
            .take_i32s()
            .unwrap()
            .iter()
            .cloned()
            .collect();
        let input = inputs.remove(0).take_f32s().unwrap();
        if dims.contains(&-1) {
            let prod: i32 = dims.iter().map(|a| *a).filter(|a| *a != -1i32).product();
            for a in dims.iter_mut() {
                if *a == -1 {
                    *a = input.len() as i32 / prod;
                }
            }
        }
        let dims: Vec<usize> = dims.into_iter().map(|a| a as usize).collect();
        Ok(vec![input.into_shape(&*dims)?.into_dyn().into()])
    }
}

#[derive(Debug)]
pub struct Squeeze {
    dims: Vec<isize>,
}

impl Squeeze {
    pub fn build(pb: &::tfpb::node_def::NodeDef) -> Result<Box<Op>> {
        let dims = pb.get_attr().get("squeeze_dims").ok_or(
            "Squeeze expect squeeze_dims attribute",
        )?;
        let mut dims: Vec<isize> = dims.get_list()
            .get_i()
            .into_iter()
            .map(|x| *x as isize)
            .collect();
        dims.sort();
        dims.reverse();
        Ok(Box::new(Squeeze { dims }))
    }
}

impl Op for Squeeze {
    fn eval(&self, inputs: Vec<Matrix>) -> Result<Vec<Matrix>> {
        let data = inputs[0].as_f32s().ok_or("Expect input #0 to be f32")?;
        let mut shape = data.shape().to_vec();
        for d in &self.dims {
            if *d >= 0 {
                shape.remove(*d as usize);
            } else {
                Err(format!("unimplemented Squeeze with negative parameter"))?
            }
        }
        Ok(vec![data.clone().into_shape(shape)?.into()])
    }
}
