use {Matrix, Result};
use super::Op;

pub struct Add {}

impl Add {
    pub fn build(pb: &::tfpb::node_def::NodeDef) -> Result<Add> {
        Ok(Add{})
    }
}

impl Op for Add {
    fn eval(&self, mut inputs: Vec<Matrix>) -> Result<Vec<Matrix>> {
        let mut input1 = inputs.remove(0).take_f32s().ok_or("Expect input #0 to be f32")?;
        let input2 = inputs.remove(0).take_f32s().ok_or("Expect input #1 to be f32")?;
        input1 += &input2;
        Ok(vec![Matrix::F32(input1)])
    }
}
