use Result;

use super::{Input, Op, OpRegister};

pub fn register_all_ops(reg: &mut OpRegister) {
    reg.insert("Cast", Cast::build);
}

#[derive(Debug)]
pub struct Cast;

impl Cast {
    pub fn build(_pb: &::tfpb::node_def::NodeDef) -> Result<Box<Op>> {
        Ok(Box::new(Cast {}))
    }
}

impl ::ops::Op for Cast {
    fn eval(&self, mut _inputs: Vec<Input>) -> Result<Vec<Input>> {
        panic!(
            "nope, fixme. parse two args in build to get src and dst types, then generalize (macro ?)"
        );
        /*
        let input = inputs.remove(0).take_f32s().ok_or(
            "Expect input #0 to be f32",
        )?;
        Ok(vec![Matrix::F32(input.mapv(|i| i as _))])
        */
    }
}
