use {Matrix, Result};
use super::{Op, OpRegister};

pub mod local_patch;

pub fn register_all_ops(reg: &mut OpRegister) {
    reg.insert("AvgPool", local_patch::AvgPool::build);
    reg.insert("Conv2D", local_patch::Conv2D::build);
    reg.insert("MaxPool", local_patch::MaxPool::build);
    reg.insert("Relu", Relu::build);
    reg.insert("Softmax", Softmax::build);
}

element_map!(Relu, |x| if x < 0.0 { 0.0 } else { x });

#[derive(Debug)]
pub struct Softmax {}

impl Softmax {
    pub fn build(_pb: &::tfpb::node_def::NodeDef) -> Result<Box<Op>> {
        Ok(Box::new(Softmax {}))
    }
}

impl Op for Softmax {
    fn eval(&self, mut inputs: Vec<Matrix>) -> Result<Vec<Matrix>> {
        let mut input = inputs.remove(0).take_f32s().ok_or(
            "Expect input #0 to be f32",
        )?;
        input.map_inplace(|a| *a = a.exp());
        let norm: f32 = input.iter().sum();
        input.map_inplace(|a| *a = *a / norm);
        Ok(vec![input.into()])
    }
}
