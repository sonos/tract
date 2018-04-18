use {Matrix, Result};
use super::{Input, Op, OpRegister};

mod local_patch;
pub mod conv2d;
pub mod pools;
pub mod space_to_batch;

pub fn register_all_ops(reg: &mut OpRegister) {
    reg.insert("AvgPool", pools::pool::<pools::AvgPooler>);
    reg.insert("Conv2D", conv2d::conv2d);
    reg.insert("MaxPool", pools::pool::<pools::MaxPooler>);
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
    fn eval(&self, mut inputs: Vec<Input>) -> Result<Vec<Input>> {
        let m_input = args_1!(inputs);
        let mut input = m_input
            .into_matrix()
            .take_f32s()
            .ok_or("Expect input #0 to be f32")?;
        input.map_inplace(|a| *a = a.exp());
        let norm: f32 = input.iter().sum();
        input.map_inplace(|a| *a = *a / norm);
        let result = Matrix::from(input);
        Ok(vec![result.into()])
    }
}
