use crate::tensor::{DeviceTensor, DeviceTensorExt};
use tract_core::internal::*;

/// (input, (scale, bias)?, axis, eps, output)
pub type DispatchLayerNormFn = fn(
    &DeviceTensor,
    Option<(&DeviceTensor, &DeviceTensor)>,
    usize,
    f32,
    &DeviceTensor,
) -> TractResult<()>;

/// GPU form of core's `LayerNorm`: inputs `[input]`, or `[input, scale, bias]` with F32 rank-1
/// scale and bias along `axis`.
#[derive(Clone)]
pub struct GpuLayerNorm {
    pub axis: usize,
    pub eps: Arc<Tensor>,
    pub backend_name: &'static str,
    pub dispatch: DispatchLayerNormFn,
}

impl std::fmt::Debug for GpuLayerNorm {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}LayerNorm(axis: {}, eps: {:?})", self.backend_name, self.axis, self.eps)
    }
}

impl PartialEq for GpuLayerNorm {
    fn eq(&self, other: &Self) -> bool {
        self.backend_name == other.backend_name && self.axis == other.axis && self.eps == other.eps
    }
}

impl Eq for GpuLayerNorm {}

impl std::hash::Hash for GpuLayerNorm {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.backend_name.hash(state);
        self.axis.hash(state);
        self.eps.hash(state);
    }
}

impl Op for GpuLayerNorm {
    fn name(&self) -> StaticName {
        format!("{}LayerNorm", self.backend_name).into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("axis: {}, eps: {:?}", self.axis, self.eps)])
    }

    op_as_typed_op!();
}

impl EvalOp for GpuLayerNorm {
    op_out_of_plan!();

    fn eval(&self, ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        ensure!(inputs.len() == 1 || inputs.len() == 3);
        let input = inputs[0].to_device_tensor()?;
        let affine = if inputs.len() == 3 {
            Some((inputs[1].to_device_tensor()?, inputs[2].to_device_tensor()?))
        } else {
            None
        };
        let output =
            crate::turn_handler::make_tensor_for_node(ctx, input.datum_type(), input.shape())?;
        (self.dispatch)(input, affine, self.axis, self.eps.cast_to_scalar::<f32>()?, &output)?;
        Ok(tvec!(output.into_tensor().into_tvalue()))
    }
}

impl TypedOp for GpuLayerNorm {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        crate::utils::facts_to_device_facts(inputs, |facts| {
            Ok(tvec!(facts[0].datum_type.fact(facts[0].shape.clone())))
        })
        .with_context(|| format!("Error while computing facts for {:?}", self.name()))
    }

    as_op!();
}
