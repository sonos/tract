use std::fmt::Debug;

//use ndarray::{Array, ArrayD, IxDyn};
//use num_traits::identities::Zero;
//use tfpb::types::DataType;
use {Matrix, Result};

#[macro_use]
mod macros;
mod activ;
mod arith;
mod cast;
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
        fn build_op(pb: &::tfpb::node_def::NodeDef) -> Result<Box<Op>> {
            match pb.get_op() {
                "Add" => Ok(Box::new(arith::Add::build(pb)?)),
                "BiasAdd" => Ok(Box::new(arith::Add::build(pb)?)),
                "Cast" => Ok(Box::new(cast::Cast::build(pb)?)),
                "Const" => Ok(Box::new(trivial::Const::build(pb)?)),
                "Conv2D" => Ok(Box::new(conv::Conv2D::build(pb)?)),
                "DecodeJpeg" => Ok((Box::new(image::DecodeJpeg::build(pb)?))),
                "ExpandDims" => Ok(Box::new(shape::ExpandDims)),
                "Identity" => Ok((Box::new(trivial::Identity::build(pb)?))),
                "Mul" => Ok(Box::new(arith::Mul::build(pb)?)),
                "Placeholder" => Ok(Box::new(trivial::Placeholder::build(pb)?)),
                "Relu" => Ok(Box::new(activ::Relu::build(pb)?)),
                "Rsqrt" => Ok(Box::new(arith::Rsqrt::build(pb)?)),
                "ResizeBilinear" => Ok(Box::new(image::ResizeBilinear::build(pb)?)),
                "Sub" => Ok(Box::new(arith::Sub::build(pb)?)),
                "Squeeze" => Ok(Box::new(shape::Squeeze::build(pb)?)),
                _ => Ok(Box::new(
                    UnimplementedOp(pb.get_op().to_string(), pb.to_owned()),
                )),
            }
        }
        build_op(pb).map_err(|e| {
            format!(
                "Error while building a {} op: {}",
                pb.get_op(),
                e.description()
            ).into()
        })
    }
}

#[derive(Debug)]
pub struct UnimplementedOp(String, ::tfpb::node_def::NodeDef);

impl Op for UnimplementedOp {
    fn eval(&self, inputs: Vec<Matrix>) -> Result<Vec<Matrix>> {
        println!("Unimplemented op: {}", self.0);
        println!(" * attrs:");
        for (k, v) in self.1.get_attr() {
            println!("    - {}: {:?}", k, v);
        }
        println!(" * inputs: {}", inputs.len());
        for (ix, i) in inputs.iter().enumerate() {
            print!(" #{}\n{}\n", ix, i.partial_dump(true)?);
        }
        Err(format!("unimplemented operation: {}", self.0))?
    }
}
