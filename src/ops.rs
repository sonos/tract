//use ndarray::{Array, ArrayD, IxDyn};
//use num_traits::identities::Zero;
//use tfpb::types::DataType;
use std::cell;
use ::{Matrix, Result};

pub trait Op: ::downcast_rs::Downcast {
    fn eval(&self, inputs:Vec<Matrix>) -> Result<Vec<Matrix>>;
}
impl_downcast!(Op);

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
            "ExpandDims" => Ok(Box::new(ExpandDims)),
            _ => Ok(Box::new(UnimplementedOp(pb.get_op().to_string())))
        }
    }
}

pub struct Placeholder {
    value: cell::Cell<Option<Matrix>>,
}

impl Placeholder {
    pub fn build(_pb: &::tfpb::node_def::NodeDef) -> Result<Placeholder> {
        Ok(Placeholder { value: None.into() })
    }
}

impl Placeholder {
    pub fn set(&self, v: Matrix) {
        self.value.set(Some(v))
    }
}

impl Op for Placeholder {
    fn eval(&self, _inputs:Vec<Matrix>) -> Result<Vec<Matrix>> {
        unsafe {
            Ok(vec!((*self.value.as_ptr()).as_ref().unwrap().clone()))
        }
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
    fn eval(&self, _inputs:Vec<Matrix>) -> Result<Vec<Matrix>> {
        Ok(vec!(self.value.clone()))
    }
}

pub struct ExpandDims;

impl Op for ExpandDims {
    fn eval(&self, inputs:Vec<Matrix>) -> Result<Vec<Matrix>> {
        use ndarray::Dimension;

        let data = inputs[0].as_f32s().ok_or("Expect input #0 to be f32")?;
        let dims = inputs[1].as_i32s().ok_or("Expect input #1 to be i32")?;
        let mut shape = data.shape().to_vec();
        for d in dims {
            if *d >= 0 {
                shape.insert(*d as usize, 1);
            } else {
                Err(format!("unimplemented ExpandDims with negative parameter"))?
            }
        }
        Ok(vec!(Matrix::F32(data.clone().into_shape(shape)?)))
    }
}

pub struct UnimplementedOp(String);

impl Op for UnimplementedOp {
    fn eval(&self, _inputs:Vec<Matrix>) -> Result<Vec<Matrix>> {
        Err(format!("unimplemented operation: {}", self.0))?
    }
}
