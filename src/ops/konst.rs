use {Matrix, Result};
use super::{Op, OpRegister};

pub fn register_all_ops(reg: &mut OpRegister) {
    reg.insert("Const", Const::build);
}

#[derive(Debug)]
pub struct Const {
    value: Matrix,
}

impl Const {
    pub fn build(pb: &::tfpb::node_def::NodeDef) -> Result<Box<Op>> {
        let tensor = pb.get_attr().get("value").unwrap().get_tensor();
        Ok(Box::new(Const { value: Matrix::from_pb(&tensor)? }))
    }
}

impl Op for Const {
    fn eval(&self, _inputs: Vec<Matrix>) -> Result<Vec<Matrix>> {
        Ok(vec![self.value.clone()])
    }
}
