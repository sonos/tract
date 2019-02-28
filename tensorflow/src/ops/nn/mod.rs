use tract_core::ops::nn::{DataFormat, PaddingSpec};
use tract_core::ops::prelude::*;

use crate::ops::OpRegister;
use crate::tfpb::node_def::NodeDef;

pub mod conv2d;
pub mod fused_batch_norm;
pub mod pools;
pub mod s2b;

pub fn register_all_ops(reg: &mut OpRegister) {
    reg.insert("AvgPool", pools::avgpool);
    reg.insert("Conv2D", conv2d::conv2d);
    reg.insert("FusedBatchNorm", fused_batch_norm::fused_batch_norm);
    reg.insert("MaxPool", pools::maxpool);
    reg.insert("Relu", with_T!(::tract_core::ops::nn::Relu));
    reg.insert("Sigmoid", with_T!(::tract_core::ops::nn::Sigmoid));
    reg.insert("Softmax", Softmax::build);
    reg.insert("SpaceToBatchND", s2b::space_to_batch_nd);
    reg.insert("BatchToSpaceND", s2b::batch_to_space_nd);
}

pub fn strides(pb: &NodeDef) -> TractResult<Vec<usize>> {
    let strides: Vec<usize> = pb.get_attr_list_int("strides")?;
    if strides.len() != 4 || strides[0] != 1 && strides[3] != 1 {
        Err(format!(
            "strides must be of the form [1, h, v, 1], found {:?}",
            strides
        ))?
    };
    Ok(strides)
}

pub fn data_format(pb: &NodeDef) -> TractResult<DataFormat> {
    let df = if pb.get_attr_opt_raw_str("data_format")?.unwrap_or(b"NHWC") == b"NHWC" {
        DataFormat::NHWC
    } else {
        DataFormat::NCHW
    };
    Ok(df)
}

pub fn padding(pb: &NodeDef) -> TractResult<PaddingSpec> {
    let padding = pb.get_attr_raw_str("padding")?;
    match padding {
        b"VALID" => Ok(PaddingSpec::Valid),
        b"SAME" => Ok(PaddingSpec::SameUpper),
        s => Err(format!(
            "unsupported Padding {}",
            String::from_utf8_lossy(s)
        ))?,
    }
}

#[derive(Debug, Clone)]
pub struct Softmax {}

impl Softmax {
    pub fn build(_pb: &NodeDef) -> TractResult<Box<Op>> {
        Ok(Box::new(Softmax {}))
    }
}

impl Op for Softmax {
    fn name(&self) -> Cow<str> {
        "Softmax".into()
    }

    fn rounding_errors(&self) -> bool {
        true
    }
}

impl StatelessOp for Softmax {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, mut inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        let input = args_1!(inputs);
        let mut input = input.to_array::<f32>()?;
        let max: f32 = input
            .iter()
            .cloned()
            .max_by(|a, b| a.partial_cmp(&b).unwrap_or(::std::cmp::Ordering::Equal))
            .unwrap_or(0.0);
        input.map_inplace(|a| *a = (*a - max).exp());
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
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 1)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].shape, &outputs[0].shape)
    }
}
