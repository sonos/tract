use std::fmt::Debug;

//use ndarray::{Array, ArrayD, IxDyn};
//use num_traits::identities::Zero;
//use tfpb::types::DataType;
use {Matrix, Result};

mod activ;
mod arith;
pub mod conv;
pub mod image;
mod shape;
pub mod trivial;

pub trait Op: ::downcast_rs::Downcast + Debug {
    fn eval(&self, inputs: Vec<Matrix>) -> Result<Vec<Matrix>>;
}
impl_downcast!(Op);

pub struct OpBuilder {}

impl OpBuilder {
    pub fn new() -> OpBuilder {
        OpBuilder {}
    }

    pub fn build(&self, pb: &::tfpb::node_def::NodeDef) -> Result<Box<Op>> {
        match pb.get_op() {
            "BiasAdd" => Ok(Box::new(arith::Add::build(pb)?)),
            "Const" => Ok(Box::new(trivial::Const::build(pb)?)),
            "Conv2D" => Ok(Box::new(conv::Conv2D::build(pb)?)),
            "DecodeJpeg" => Ok((Box::new(image::DecodeJpeg::build(pb)?))),
            "ExpandDims" => Ok(Box::new(shape::ExpandDims)),
            "Placeholder" => Ok(Box::new(trivial::Placeholder::build(pb)?)),
            "Relu" => Ok(Box::new(activ::Relu::build(pb)?)),
            "Squeeze" => Ok(Box::new(shape::Squeeze::build(pb)?)),
            _ => Ok(Box::new(UnimplementedOp(pb.get_op().to_string()))),
        }
    }
}

#[derive(Debug)]
pub struct UnimplementedOp(String);

impl Op for UnimplementedOp {
    fn eval(&self, _inputs: Vec<Matrix>) -> Result<Vec<Matrix>> {
        Err(format!("unimplemented operation: {}", self.0))?
    }
}
