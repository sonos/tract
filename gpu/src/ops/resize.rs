use crate::tensor::{DeviceTensor, DeviceTensorExt, IntoDevice};
use derive_new::new;
use tract_core::internal::*;

/// Resamples one axis: `output[.., x, ..] = sum_k weights[x, k] * input[.., indices[x, k], ..]`,
/// with `indices` already clamped into the axis by the host-built plan.
pub type DispatchResizeAxisFn = fn(
    input: &DeviceTensor,
    axis: usize,
    indices: &DeviceTensor,
    weights: &DeviceTensor,
    window: usize,
    output: &DeviceTensor,
) -> TractResult<()>;

/// Resize against a plan baked at translation time, one dispatch per resampled
/// axis. The plan makes the op independent of the interpolator: nearest, linear
/// and cubic differ only in window size and weights, so translation is limited
/// to nodes whose shapes and scales are known then. The scales/sizes input is
/// kept for arity but no longer read.
#[derive(Clone, new)]
pub struct GpuResize {
    pub axes: TVec<usize>,
    pub windows: TVec<usize>,
    pub plans: TVec<(Arc<Tensor>, Arc<Tensor>)>,
    pub output_shape: TVec<usize>,
    pub backend_name: &'static str,
    pub dispatch: DispatchResizeAxisFn,
}

impl std::fmt::Debug for GpuResize {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}Resize", self.backend_name)
    }
}

impl PartialEq for GpuResize {
    fn eq(&self, other: &Self) -> bool {
        self.backend_name == other.backend_name
            && self.axes == other.axes
            && self.windows == other.windows
            && self.output_shape == other.output_shape
    }
}
impl Eq for GpuResize {}

impl std::hash::Hash for GpuResize {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.backend_name.hash(state);
        self.axes.hash(state);
        self.windows.hash(state);
        self.output_shape.hash(state);
    }
}

impl Op for GpuResize {
    fn name(&self) -> StaticName {
        format!("{}Resize", self.backend_name).into()
    }
    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("axes={:?} windows={:?}", self.axes, self.windows)])
    }
    op_as_typed_op!();
}

impl EvalOp for GpuResize {
    op_out_of_plan!();

    fn eval(&self, ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let data = inputs[0].to_device_tensor()?;
        let dt = data.datum_type();
        let mut shape: TVec<usize> = data.shape().into();
        let mut current = data.clone();
        for (step, (&axis, &window)) in self.axes.iter().zip(&self.windows).enumerate() {
            let (indices, weights) = &self.plans[step];
            let indices = indices.as_ref().clone().into_device()?;
            let weights = weights.as_ref().clone().into_device()?;
            shape[axis] = self.output_shape[axis];
            let last = step + 1 == self.axes.len();
            let output = if last {
                crate::turn_handler::make_tensor_for_node(ctx, dt, &shape)?
            } else {
                DeviceTensor::uninitialized_dt(dt, &shape)?
            };
            (self.dispatch)(&current, axis, &indices, &weights, window, &output)?;
            current = output;
        }
        Ok(tvec!(current.into_tensor().into_tvalue()))
    }
}

impl TypedOp for GpuResize {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        crate::utils::facts_to_device_facts(inputs, |facts| {
            ensure!(facts.len() == 1);
            let shape: TVec<TDim> = self.output_shape.iter().map(|d| d.to_dim()).collect();
            Ok(tvec!(facts[0].datum_type.fact(&shape)))
        })
        .with_context(|| format!("Error while computing facts for {:?}", self.name()))
    }
    as_op!();
}
