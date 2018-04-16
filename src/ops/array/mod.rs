use ndarray::prelude::*;

mod pack;
mod strided_slice;

use {Matrix, Result};
use super::{Input, Op, OpRegister};

pub fn register_all_ops(reg: &mut OpRegister) {
    reg.insert("ConcatV2", ConcatV2::build);
    reg.insert("ExpandDims", ExpandDims::build);
    reg.insert("Identity", Identity::build);
    reg.insert("Pack", pack::pack);
    reg.insert("Placeholder", Placeholder::build);
    reg.insert("Reshape", Reshape::build);
    reg.insert("Shape", Shape::build);
    reg.insert("Squeeze", Squeeze::build);
    reg.insert("StridedSlice", strided_slice::build);
}

#[derive(Debug)]
pub struct ConcatV2 {
    n: usize,
}

impl ConcatV2 {
    pub fn build(pb: &::tfpb::node_def::NodeDef) -> Result<Box<Op>> {
        Ok(Box::new(ConcatV2 {
            n: pb.get_attr().get("N").unwrap().get_i() as _,
        }))
    }
}

impl Op for ConcatV2 {
    fn eval(&self, inputs: Vec<Input>) -> Result<Vec<Input>> {
        let axis: i32 = inputs[self.n]
            .as_i32s()
            .ok_or("Expected a i32 matrix")?
            .iter()
            .next()
            .unwrap()
            .clone();
        let mats: Vec<_> = inputs[0..self.n]
            .iter()
            .map(|mat| mat.as_f32s().unwrap().view())
            .collect();
        let result = ::ndarray::stack(Axis(axis as usize), &*mats)?;
        let result = Matrix::from(result);
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
    fn eval(&self, mut inputs: Vec<Input>) -> Result<Vec<Input>> {
        let (data, dims) = args_2!(inputs);
        let data = data.into_matrix()
            .take_f32s()
            .ok_or("Expected a f32 matrix")?;
        let dims = dims.as_i32s().ok_or("Expected a i32 matrix")?;
        let mut shape = data.shape().to_vec();
        for d in dims.iter() {
            if *d >= 0 {
                shape.insert(*d as usize, 1);
            } else {
                Err(format!("unimplemented ExpandDims with negative parameter"))?
            }
        }
        Ok(vec![Matrix::from(data.into_shape(shape)?).into()])
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
    fn eval(&self, inputs: Vec<Input>) -> Result<Vec<Input>> {
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
    fn eval(&self, _inputs: Vec<Input>) -> Result<Vec<Input>> {
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
    fn eval(&self, mut inputs: Vec<Input>) -> Result<Vec<Input>> {
        let (input, dims) = args_2!(inputs);
        let input = input
            .into_matrix()
            .take_f32s()
            .ok_or("Expected a f32 matrix")?;
        let mut dims: Vec<i32> = dims.as_i32s()
            .ok_or("Expected a i32 matrix")?
            .iter()
            .cloned()
            .collect();
        if dims.contains(&-1) {
            let prod: i32 = dims.iter().map(|a| *a).filter(|a| *a != -1i32).product();
            for a in dims.iter_mut() {
                if *a == -1 {
                    *a = input.len() as i32 / prod;
                }
            }
        }
        let dims: Vec<usize> = dims.into_iter().map(|a| a as usize).collect();
        Ok(vec![
            Matrix::from(input.into_shape(&*dims)?.into_dyn()).into(),
        ])
    }
}

#[derive(Debug)]
pub struct Shape;

impl Shape {
    pub fn build(_pb: &::tfpb::node_def::NodeDef) -> Result<Box<Op>> {
        Ok(Box::new(Shape))
    }
}

impl Op for Shape {
    fn eval(&self, inputs: Vec<Input>) -> Result<Vec<Input>> {
        let data = inputs[0].as_f32s().ok_or("Expect input #0 to be f32")?;
        let shape: Vec<i32> = data.shape().into_iter().map(|s| *s as i32).collect();
        Ok(vec![Matrix::from(Array1::from_vec(shape)).into()])
    }
}

#[derive(Debug)]
pub struct Squeeze {
    dims: Vec<isize>,
}

impl Squeeze {
    pub fn build(pb: &::tfpb::node_def::NodeDef) -> Result<Box<Op>> {
        let dims = pb.get_attr()
            .get("squeeze_dims")
            .ok_or("Squeeze expect squeeze_dims attribute")?;
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
    fn eval(&self, inputs: Vec<Input>) -> Result<Vec<Input>> {
        let data = inputs[0].as_f32s().ok_or("Expect input #0 to be f32")?;
        let mut shape = data.shape().to_vec();
        for d in &self.dims {
            if *d >= 0 {
                shape.remove(*d as usize);
            } else {
                Err(format!("unimplemented Squeeze with negative parameter"))?
            }
        }
        Ok(vec![Matrix::from(data.clone().into_shape(shape)?).into()])
    }
}
