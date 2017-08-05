//use ndarray::{Array, ArrayD, IxDyn};
//use num_traits::identities::Zero;
//use tfpb::types::DataType;
use ::{Matrix, Result};

pub trait Op {
    fn eval(&self, inputs:Vec<Matrix>) -> Vec<Matrix>;
}

pub struct OpBuilder {
}

impl OpBuilder {
    pub fn new() -> OpBuilder {
        OpBuilder {}
    }

    pub fn build(&self, pb: &::tfpb::node_def::NodeDef) -> Result<Box<Op>> {
        match pb.get_op() {
            "Placeholder" => Ok(Box::new(Placeholder::build(pb)?)),
            "Const" => Ok(Box::new(Const::build(pb)?)),
            _ => Ok(Box::new(UnimplementedOp))
        }
    }
}

pub struct Placeholder {
    value: Option<Matrix>,
}

impl Placeholder {
    pub fn build(_pb: &::tfpb::node_def::NodeDef) -> Result<Placeholder> {
        Ok(Placeholder { value: None })
    }
}

impl Placeholder {
    pub fn set(&mut self, v: Matrix) {
        self.value = Some(v)
    }
}

impl Op for Placeholder {
    fn eval(&self, _inputs:Vec<Matrix>) -> Vec<Matrix> {
        vec!(self.value.clone().unwrap())
    }
}

pub struct Const {
    value: Matrix,
}

impl Const {
    pub fn build(pb: &::tfpb::node_def::NodeDef) -> Result<Const> {
        let tensor = pb.get_attr().get("value").unwrap().get_tensor();
        Ok(Const { value: Matrix::from_pb(&tensor)? })
    }
}

impl Op for Const {
    fn eval(&self, _inputs:Vec<Matrix>) -> Vec<Matrix> {
        vec!(self.value.clone())
    }
}

pub struct UnimplementedOp;

impl Op for UnimplementedOp {
    fn eval(&self, _inputs:Vec<Matrix>) -> Vec<Matrix> {
        unimplemented!()
    }
}
