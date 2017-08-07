use {Matrix, Result};
use super::Op;

pub struct Relu {}

impl Relu {
    pub fn build(_pb: &::tfpb::node_def::NodeDef) -> Result<Relu> {
        Ok(Relu {})
    }
}

impl Op for Relu {
    fn eval(&self, mut inputs: Vec<Matrix>) -> Result<Vec<Matrix>> {
        let mut input1 = inputs.remove(0).take_f32s().ok_or(
            "Expect input #0 to be f32",
        )?;
        for mut x in &mut input1 {
            if *x < 0.0 {
                *x = 0.0;
            }
        }
        Ok(vec![Matrix::F32(input1)])
    }
}
