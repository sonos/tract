use {Matrix, Result};
use super::Op;

element_map!(Rsqrt, |x: f32| 1.0 / (x.sqrt()));

#[derive(Debug)]
pub struct Add {}

impl Add {
    pub fn build(_pb: &::tfpb::node_def::NodeDef) -> Result<Add> {
        Ok(Add {})
    }
}

impl Op for Add {
    fn eval(&self, mut inputs: Vec<Matrix>) -> Result<Vec<Matrix>> {
        let mut input1 = inputs.remove(0).take_f32s().ok_or(
            "Expect input #0 to be f32",
        )?;
        let input2 = inputs.remove(0).take_f32s().ok_or(
            "Expect input #1 to be f32",
        )?;
        input1 += &input2;
        Ok(vec![input1.into()])
    }
}

#[derive(Debug)]
pub struct Sub {}

impl Sub {
    pub fn build(_pb: &::tfpb::node_def::NodeDef) -> Result<Sub> {
        Ok(Sub {})
    }
}

impl Op for Sub {
    fn eval(&self, mut inputs: Vec<Matrix>) -> Result<Vec<Matrix>> {
        let mut input1 = inputs.remove(0).take_f32s().ok_or(
            "Expect input #0 to be f32",
        )?;
        let input2 = inputs.remove(0).take_f32s().ok_or(
            "Expect input #1 to be f32",
        )?;
        input1 -= &input2;
        Ok(vec![input1.into()])
    }
}

#[derive(Debug)]
pub struct Mul {}

impl Mul {
    pub fn build(_pb: &::tfpb::node_def::NodeDef) -> Result<Mul> {
        Ok(Mul {})
    }
}

impl Op for Mul {
    fn eval(&self, mut inputs: Vec<Matrix>) -> Result<Vec<Matrix>> {
        let mut input1 = inputs.remove(0).take_f32s().ok_or(
            "Expect input #0 to be f32",
        )?;
        let input2 = inputs.remove(0).take_f32s().ok_or(
            "Expect input #1 to be f32",
        )?;
        input1 *= &input2;
        Ok(vec![input1.into()])
    }
}
