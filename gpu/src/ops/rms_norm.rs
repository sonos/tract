use crate::tensor::{DeviceTensor, DeviceTensorExt};
use derive_new::new;
use std::sync::Arc;
use tract_core::internal::*;

/// (input, residual?, scale?, axis, eps, output, sum_out?): residual and
/// sum_out come and go together (see `GpuRmsNorm::has_residual`).
pub type DispatchRmsNormFn = fn(
    &DeviceTensor,
    Option<&DeviceTensor>,
    Option<&DeviceTensor>,
    usize,
    &Tensor,
    &DeviceTensor,
    Option<&DeviceTensor>,
) -> TractResult<()>;

#[derive(Clone, new)]
pub struct GpuRmsNorm {
    pub axis: usize,
    pub eps: Arc<Tensor>,
    /// When true the op takes a second input: a rank-1 F32 scale vector
    /// multiplied along `axis` after normalization (fused learned weight).
    pub has_scale: bool,
    /// When true the op takes a same-shape residual input right after the
    /// main input: it normalizes `input + residual` (sum computed in the
    /// input dtype, bit-identical to a standalone Add) and returns the sum
    /// as a SECOND output for the consumers of the residual stream.
    pub has_residual: bool,
    /// Output dtype when it differs from the input dtype (fused cast).
    pub out_dt: Option<DatumType>,
    pub backend_name: &'static str,
    pub dispatch: DispatchRmsNormFn,
}

impl std::fmt::Debug for GpuRmsNorm {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "{}RmsNorm(axis: {:?}, eps: {:?}, scaled: {})",
            self.backend_name, self.axis, self.eps, self.has_scale
        )
    }
}

impl PartialEq for GpuRmsNorm {
    fn eq(&self, other: &Self) -> bool {
        self.backend_name == other.backend_name
            && self.axis == other.axis
            && self.eps == other.eps
            && self.has_scale == other.has_scale
            && self.has_residual == other.has_residual
            && self.out_dt == other.out_dt
    }
}

impl Eq for GpuRmsNorm {}

impl std::hash::Hash for GpuRmsNorm {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.backend_name.hash(state);
        self.axis.hash(state);
        self.eps.hash(state);
        self.has_scale.hash(state);
        self.has_residual.hash(state);
        self.out_dt.hash(state);
    }
}

impl Op for GpuRmsNorm {
    fn name(&self) -> StaticName {
        format!("{}RmsNorm", self.backend_name).into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("axis: {:?}, eps: {:?}", self.axis, self.eps)])
    }

    op_as_typed_op!();
}

impl EvalOp for GpuRmsNorm {
    op_out_of_plan!();

    fn eval(&self, ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        ensure!(inputs.len() == 1 + self.has_scale as usize + self.has_residual as usize);
        let mut inputs = inputs;
        let scale_value = if self.has_scale { Some(inputs.pop().unwrap()) } else { None };
        let residual_value = if self.has_residual { Some(inputs.pop().unwrap()) } else { None };
        let input_value = inputs.pop().unwrap();
        let input = input_value.to_device_tensor()?;
        let residual = residual_value.as_ref().map(|s| s.to_device_tensor()).transpose()?;
        let scale = scale_value.as_ref().map(|s| s.to_device_tensor()).transpose()?;
        if self.has_residual {
            let output = crate::turn_handler::make_tensor_for_node_output(
                ctx,
                0,
                self.out_dt.unwrap_or(input.datum_type()),
                input.shape(),
            )?;
            let sum_out = crate::turn_handler::make_tensor_for_node_output(
                ctx,
                1,
                input.datum_type(),
                input.shape(),
            )?;
            (self.dispatch)(input, residual, scale, self.axis, &self.eps, &output, Some(&sum_out))?;
            return Ok(tvec!(
                output.into_tensor().into_tvalue(),
                sum_out.into_tensor().into_tvalue()
            ));
        }
        let output = crate::turn_handler::make_tensor_for_node(
            ctx,
            self.out_dt.unwrap_or(input.datum_type()),
            input.shape(),
        )?;
        (self.dispatch)(input, None, scale, self.axis, &self.eps, &output, None)?;
        Ok(tvec!(output.into_tensor().into_tvalue()))
    }
}

impl TypedOp for GpuRmsNorm {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        crate::utils::facts_to_device_facts(inputs, |facts| {
            let dt = self.out_dt.unwrap_or(facts[0].datum_type);
            let fact = dt.fact(facts[0].shape.clone());
            if self.has_residual {
                let sum_fact = facts[0].datum_type.fact(facts[0].shape.clone());
                Ok(tvec!(fact, sum_fact))
            } else {
                Ok(tvec!(fact))
            }
        })
        .with_context(|| format!("Error while computing facts for {:?}", self.name()))
    }

    as_op!();
}
