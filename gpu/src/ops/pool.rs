use crate::tensor::{DeviceTensor, DeviceTensorExt};
use tract_core::internal::*;
use tract_core::ops::cnn::{MaxPool, OptMaxPool, OptSumPool, PoolSpec, SumPool};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PoolKind {
    Max,
    /// `normalize` turns the sum into an average, over the whole window when
    /// `count_include_pad`, over the positions inside the input otherwise.
    Sum {
        count_include_pad: bool,
        normalize: bool,
    },
}

/// Launches the backend's pooling kernel from `input` into the already shaped `output`.
pub type DispatchPoolFn = fn(
    pool_spec: &PoolSpec,
    kind: PoolKind,
    input: &DeviceTensor,
    output: &DeviceTensor,
) -> TractResult<()>;

/// Max or sum pooling on the device. Each backend decides which geometries and
/// datum types its kernel covers before building one.
#[derive(Clone)]
pub struct GpuPool {
    pub pool_spec: PoolSpec,
    pub kind: PoolKind,
    pub backend_name: &'static str,
    pub dispatch: DispatchPoolFn,
}

impl GpuPool {
    /// Maps a core `MaxPool`, `SumPool` or their `Opt*` forms. A max pool with an
    /// index output has no device equivalent and maps to `None`.
    pub fn from_core(
        op: &dyn TypedOp,
        backend_name: &'static str,
        dispatch: DispatchPoolFn,
    ) -> Option<GpuPool> {
        let (pool_spec, kind) = if let Some(op) = op.downcast_ref::<MaxPool>() {
            if op.with_index_outputs.is_some() {
                return None;
            }
            (&op.pool_spec, PoolKind::Max)
        } else if let Some(op) = op.downcast_ref::<OptMaxPool>() {
            if op.with_index_outputs.is_some() {
                return None;
            }
            (&op.pool_spec, PoolKind::Max)
        } else if let Some(op) = op.downcast_ref::<SumPool>() {
            let kind =
                PoolKind::Sum { count_include_pad: op.count_include_pad, normalize: op.normalize };
            (&op.pool_spec, kind)
        } else {
            let op = op.downcast_ref::<OptSumPool>()?;
            let kind =
                PoolKind::Sum { count_include_pad: op.count_include_pad, normalize: op.normalize };
            (&op.pool_spec, kind)
        };
        Some(GpuPool { pool_spec: pool_spec.clone(), kind, backend_name, dispatch })
    }
}

impl std::fmt::Debug for GpuPool {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}Pool({:?}, {:?})", self.backend_name, self.kind, self.pool_spec)
    }
}

impl PartialEq for GpuPool {
    fn eq(&self, other: &Self) -> bool {
        self.backend_name == other.backend_name
            && self.kind == other.kind
            && self.pool_spec == other.pool_spec
    }
}
impl Eq for GpuPool {}

impl std::hash::Hash for GpuPool {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.backend_name.hash(state);
        self.kind.hash(state);
        self.pool_spec.hash(state);
    }
}

impl Op for GpuPool {
    fn name(&self) -> StaticName {
        match self.kind {
            PoolKind::Max => format!("{}MaxPool", self.backend_name).into(),
            PoolKind::Sum { .. } => format!("{}SumPool", self.backend_name).into(),
        }
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(self.pool_spec.info())
    }

    op_as_typed_op!();
}

impl EvalOp for GpuPool {
    op_out_of_plan!();

    fn eval(&self, ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input = inputs[0].to_device_tensor()?;
        let output_shape = self.pool_spec.output_shape(input.shape())?;
        let output = crate::turn_handler::make_tensor_for_node(
            ctx,
            input.datum_type(),
            &output_shape.shape,
        )?;
        if output.len() > 0 {
            (self.dispatch)(&self.pool_spec, self.kind, input, &output)?;
        }
        Ok(tvec!(output.into_tensor().into_tvalue()))
    }
}

impl TypedOp for GpuPool {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        crate::utils::facts_to_device_facts(inputs, |facts| {
            let shape = self.pool_spec.output_shape(&facts[0].shape)?;
            Ok(tvec!(facts[0].datum_type.fact(shape.shape)))
        })
        .with_context(|| format!("Error while computing facts for {}", self.name()))
    }
}
