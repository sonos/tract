use tfdeploy::analyser::rules::prelude::*;
use tfdeploy::ops::prelude::*;

use ops::OpRegister;
use tfpb::node_def::NodeDef;

pub mod conv2d;
pub mod local_patch;
pub mod pools;
pub mod space_to_batch;

pub fn register_all_ops(reg: &mut OpRegister) {
    reg.insert("AvgPool", pools::pool::<pools::AvgPooler>);
    reg.insert("Conv2D", conv2d::conv2d);
    reg.insert("MaxPool", pools::pool::<pools::MaxPooler>);
    reg.insert("Relu", with_T!(::tfdeploy::ops::nn::Relu));
    reg.insert("Sigmoid", with_T!(::tfdeploy::ops::nn::Sigmoid));
    reg.insert("Softmax", Softmax::build);
    reg.insert("SpaceToBatchND", space_to_batch::space_to_batch_nd);
    reg.insert("BatchToSpaceND", space_to_batch::batch_to_space_nd);
}

#[derive(Debug, Clone)]
pub struct Softmax {}

impl Softmax {
    pub fn build(_pb: &NodeDef) -> TfdResult<Box<Op>> {
        Ok(Box::new(Softmax {}))
    }
}

impl Op for Softmax {
    fn name(&self) -> &str {
        "Softmax"
    }
    /// Evaluates the operation given the input tensors.
    fn eval(&self, mut inputs: TVec<Value>) -> TfdResult<TVec<Value>> {
        let m_input = args_1!(inputs);
        let mut input = m_input
            .into_tensor()
            .take_f32s()
            .ok_or("Expect input #0 to be f32")?;
        input.map_inplace(|a| *a = a.exp());
        let norm: f32 = input.iter().sum();
        input.map_inplace(|a| *a = *a / norm);
        let result = Tensor::from(input);
        Ok(tvec![result.into()])
    }
}

impl InferenceRulesOp for Softmax {
    /// Registers the inference rules of the operator.
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        solver: &mut Solver<'r>,
        inputs: &'p TensorsProxy,
        outputs: &'p TensorsProxy,
    ) {
        solver
            .equals(&inputs.len, 1)
            .equals(&outputs.len, 1)
            .equals(&inputs[0].datum_type, &outputs[0].datum_type)
            .equals(&inputs[0].shape, &outputs[0].shape);
    }
}
