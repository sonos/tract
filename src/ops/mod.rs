//! TensorFlow Ops

use std::fmt::Debug;
use std::collections::HashMap;

use {Matrix, Result};

#[macro_use]
mod macros;

mod array;
mod math;
mod cast;
pub mod nn;
pub mod image;
pub mod konst;

pub trait Op: ::downcast_rs::Downcast + Debug {
    fn eval(&self, inputs: Vec<Matrix>) -> Result<Vec<Matrix>>;
}
impl_downcast!(Op);

type OpRegister = HashMap<&'static str, fn(&::tfpb::node_def::NodeDef) -> Result<Box<Op>>>;

pub struct OpBuilder(OpRegister);

impl OpBuilder {
    pub fn new() -> OpBuilder {
        let mut reg = OpRegister::new();
        array::register_all_ops(&mut reg);
        cast::register_all_ops(&mut reg);
        konst::register_all_ops(&mut reg);
        math::register_all_ops(&mut reg);
        nn::register_all_ops(&mut reg);
        OpBuilder(reg)
    }

    pub fn build(&self, pb: &::tfpb::node_def::NodeDef) -> Result<Box<Op>> {
        match self.0.get(pb.get_op()) {
            Some(builder) => builder(pb),
            None => Ok(Box::new(UnimplementedOp(pb.get_op().to_string(), pb.to_owned())))
        }
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
