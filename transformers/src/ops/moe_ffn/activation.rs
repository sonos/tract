use crate::ops::{gelu_approximate::gelu_approximate, silu::silu};
use tract_nnef::internal::*;

/// gpt-oss clamped SwiGLU, as an op so the expert subplans can express it:
///   gate = min(gate, limit); up = clamp(up, -limit, limit)
///   out  = (up + 1) * gate * sigmoid(alpha * gate)
#[derive(Clone, Debug, PartialEq)]
pub(super) struct ClampedSwiGlu {
    pub(super) alpha: f32,
    pub(super) limit: f32,
}

impl Eq for ClampedSwiGlu {}

impl Op for ClampedSwiGlu {
    fn name(&self) -> StaticName {
        "ClampedSwiGlu".into()
    }

    op_as_typed_op!();
}

impl EvalOp for ClampedSwiGlu {
    op_out_of_plan!();

    fn eval(&self, _ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        ensure!(inputs.len() == 2, "ClampedSwiGlu expects gate and up inputs");
        let gate = inputs[0].cast_to::<f32>()?.into_owned();
        let up = inputs[1].cast_to::<f32>()?.into_owned();
        ensure!(
            gate.shape() == up.shape(),
            "ClampedSwiGlu gate/up shape mismatch: {:?} vs {:?}",
            gate.shape(),
            up.shape()
        );
        let mut output = Tensor::zero_dt(f32::datum_type(), gate.shape())?;
        let gate_ram = gate.try_as_plain_ram()?;
        let up_ram = up.try_as_plain_ram()?;
        let mut output_ram = output.try_as_plain_ram_mut()?;
        let gate = gate_ram.as_slice::<f32>()?;
        let up = up_ram.as_slice::<f32>()?;
        let output_slice = output_ram.as_slice_mut::<f32>()?;
        for ((out, &gate), &up) in output_slice.iter_mut().zip(gate).zip(up) {
            let gate = gate.min(self.limit);
            let up = up.clamp(-self.limit, self.limit);
            let glu = gate / (1.0 + (-self.alpha * gate).exp());
            *out = (up + 1.0) * glu;
        }
        Ok(tvec![output.into_tvalue()])
    }
}

impl TypedOp for ClampedSwiGlu {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(inputs.len() == 2, "ClampedSwiGlu expects gate and up inputs");
        ensure!(
            inputs[0].shape == inputs[1].shape,
            "ClampedSwiGlu gate/up shape mismatch: {:?} vs {:?}",
            inputs[0].shape,
            inputs[1].shape
        );
        Ok(tvec!(f32::datum_type().fact(inputs[0].shape.clone())))
    }

    as_op!();
}

pub(super) fn activation_op(name: &str, has_w3: bool) -> Option<Box<dyn TypedOp>> {
    match name {
        "silu" => Some(Box::new(silu())),
        // SwiGLU: the inner activation is silu, w3 provides the gate branch
        "swiglu" if has_w3 => Some(Box::new(silu())),
        "gelu" => Some(Box::new(gelu_approximate(false))),
        "relu" => Some(Box::new(tract_nnef::tract_core::ops::nn::leaky_relu(0.0))),
        _ => None,
    }
}
