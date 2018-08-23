use std::collections::HashMap;
use Result;

use super::prelude::*;
use analyser::interface::*;

pub fn register_all_ops(reg: &mut OpRegister) {
    reg.insert("Cast", Cast::build);
}

#[derive(Debug, Clone)]
pub struct Cast;

impl Cast {
    pub fn build(_pb: &::tfpb::node_def::NodeDef) -> Result<Box<Op>> {
        Ok(Box::new(Cast {}))
    }
}

impl ::ops::Op for Cast {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, mut _inputs: TVec<Value>) -> Result<TVec<Value>> {
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

    /// Returns the attributes of the operation and their values.
    fn get_attributes(&self) -> HashMap<&'static str, Attr> {
        unimplemented!("Cast op")
    }
}

impl InferenceRulesOp for Cast {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        _: &mut Solver<'r>,
        _: &'p TensorsProxy,
        _: &'p TensorsProxy,
    ) {
        unimplemented!("Cast op")
    }
}
