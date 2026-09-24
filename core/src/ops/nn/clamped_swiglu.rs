use crate::internal::*;

/// Two-input GPT-OSS clamped SwiGLU, without broadcasting.
///
/// For `g = min(gate, limit)` and `u = clamp(up, -limit, limit)`, computes
/// `(u + 1) * g * sigmoid(alpha * g)`. Inputs must have equal shapes and
/// f16 or f32 elements; computation and output are f32. Both parameters must
/// be finite, and `limit` must be positive.
#[derive(Clone, Debug, PartialEq)]
pub struct ClampedSwiGlu {
    pub alpha: f32,
    pub limit: f32,
}

impl Eq for ClampedSwiGlu {}

impl ClampedSwiGlu {
    fn validate(&self) -> TractResult<()> {
        ensure!(self.alpha.is_finite(), "ClampedSwiGlu alpha must be finite");
        ensure!(
            self.limit.is_finite() && self.limit > 0.0,
            "ClampedSwiGlu limit must be finite and positive"
        );
        Ok(())
    }
}

impl Op for ClampedSwiGlu {
    fn name(&self) -> StaticName {
        "ClampedSwiGlu".into()
    }
    op_as_typed_op!();
}

impl EvalOp for ClampedSwiGlu {
    op_out_of_plan!();

    fn eval(&self, _ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        self.validate()?;
        ensure!(inputs.len() == 2, "ClampedSwiGlu expects gate and up inputs");
        ensure!(inputs[0].shape() == inputs[1].shape(), "ClampedSwiGlu gate/up shape mismatch");
        ensure!(
            inputs.iter().all(|t| matches!(t.datum_type(), DatumType::F16 | DatumType::F32)),
            "ClampedSwiGlu inputs must be f16 or f32"
        );
        let mut output = Tensor::zero_dt(f32::datum_type(), inputs[0].shape())?;
        if output.len() == 0 {
            return Ok(tvec![output.into_tvalue()]);
        }
        let gate = inputs[0].cast_to::<f32>()?.into_owned();
        let up = inputs[1].cast_to::<f32>()?.into_owned();
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
        self.validate()?;
        ensure!(inputs.len() == 2, "ClampedSwiGlu expects gate and up inputs");
        ensure!(inputs[0].shape == inputs[1].shape, "ClampedSwiGlu gate/up shape mismatch");
        ensure!(
            inputs.iter().all(|f| matches!(f.datum_type, DatumType::F16 | DatumType::F32)),
            "ClampedSwiGlu inputs must be f16 or f32"
        );
        Ok(tvec![f32::datum_type().fact(inputs[0].shape.clone())])
    }
    as_op!();
}
