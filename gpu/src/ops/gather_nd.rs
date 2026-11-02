use crate::tensor::{DeviceTensor, DeviceTensorExt};
use derive_new::new;
use tract_core::internal::*;
use tract_core::ops::array::GatherNd;

/// Coordinates an index row may hold: the gathered axes a backend kernel walks.
pub const MAX_COORDS: usize = 4;

/// Gathers the slices of `data` that rows of f32 `indices` point at, after `batch_dims` axes
/// both share. Negative coordinates wrap.
pub type DispatchGatherNdFn = fn(
    batch_dims: usize,
    data: &DeviceTensor,
    indices: &DeviceTensor,
    output: &DeviceTensor,
) -> TractResult<()>;

/// [`GatherNd`] on the device, reading its indices as f32. `float_gather_nd_indices` produces
/// such indices from the integer coordinates a deformable convolution computes.
#[derive(Clone, new)]
pub struct GpuGatherNd {
    pub batch_dims: usize,
    pub backend_name: &'static str,
    pub dispatch: DispatchGatherNdFn,
}

impl GpuGatherNd {
    /// Whether a backend translates a `GatherNd` with these input facts.
    pub fn is_supported(facts: &[&TypedFact]) -> bool {
        facts[1].datum_type == f32::datum_type()
            && facts[1]
                .shape
                .last()
                .and_then(|d| d.to_usize().ok())
                .is_some_and(|n| 0 < n && n <= MAX_COORDS)
    }
}

impl std::fmt::Debug for GpuGatherNd {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}GatherNd", self.backend_name)
    }
}

impl PartialEq for GpuGatherNd {
    fn eq(&self, other: &Self) -> bool {
        self.backend_name == other.backend_name && self.batch_dims == other.batch_dims
    }
}
impl Eq for GpuGatherNd {}

impl std::hash::Hash for GpuGatherNd {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.backend_name.hash(state);
        self.batch_dims.hash(state);
    }
}

impl Op for GpuGatherNd {
    fn name(&self) -> StaticName {
        format!("{}GatherNd", self.backend_name).into()
    }
    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("batch_dims={}", self.batch_dims)])
    }
    op_as_typed_op!();
}

impl EvalOp for GpuGatherNd {
    op_out_of_plan!();

    fn eval(&self, ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let (data, indices) = args_2!(inputs);
        let data = data.to_device_tensor()?;
        let indices = indices.to_device_tensor()?;
        let n = *indices.shape().last().context("GatherNd indices of rank 0")?;
        let mut shape: TVec<usize> = indices.shape()[..indices.rank() - 1].into();
        shape.extend(data.shape()[self.batch_dims + n..].iter().copied());
        let output = crate::turn_handler::make_tensor_for_node(ctx, data.datum_type(), &shape)?;
        if output.len() > 0 {
            (self.dispatch)(self.batch_dims, data, indices, &output)?;
        }
        Ok(tvec!(output.into_tensor().into_tvalue()))
    }
}

impl TypedOp for GpuGatherNd {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        crate::utils::facts_to_device_facts(inputs, |facts| {
            GatherNd::new(self.batch_dims).output_facts(facts)
        })
        .with_context(|| format!("Error while computing facts for {:?}", self.name()))
    }
    as_op!();
}
