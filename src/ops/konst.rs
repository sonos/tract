use {Matrix, Result};
use super::{Input, Op, OpRegister};
use std::sync::Arc;

pub fn register_all_ops(reg: &mut OpRegister) {
    reg.insert("Const", Const::build);
}

#[derive(Debug)]
pub struct Const {
    value: Arc<Matrix>,
}

impl Const {
    pub fn build(pb: &::tfpb::node_def::NodeDef) -> Result<Box<Op>> {
        let value = pb.get_attr().get("value").unwrap();
        let tensor = value.get_tensor();
        let mat = Matrix::from_pb(&tensor)?;
        Ok(Box::new(Const { value: Arc::new(mat) }))
    }
}

impl Op for Const {
    fn eval(&self, _inputs: Vec<Input>) -> Result<Vec<Input>> {
        Ok(vec![self.value.clone().into()])
    }
}
