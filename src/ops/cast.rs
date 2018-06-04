use Result;

use super::{Op, OpRegister, TensorView};
use analyser::TensorFact;

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
    /// Evaluates the operation given the input tensors.
    fn eval(&self, mut _inputs: Vec<TensorView>) -> Result<Vec<TensorView>> {
        panic!(
            "nope, fixme. parse two args in build to get src and dst types, then generalize (macro ?)"
        );
        /*
        let input = inputs.remove(0).take_f32s().ok_or(
            "Expect input #0 to be f32",
        )?;
        Ok(vec![Tensor::F32(input.mapv(|i| i as _))])
        */
    }

    /// Infers properties about the output tensors from the input tensors.
    fn infer_forward(&self, _inputs: Vec<&TensorFact>) -> Result<Option<Vec<TensorFact>>> {
        unimplemented!()
    }

    /// Infers properties about the input tensors from the output tensors.
    fn infer_backward(&self, _outputs: Vec<&TensorFact>) -> Result<Option<Vec<TensorFact>>> {
        unimplemented!()
    }
}
