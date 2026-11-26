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

/// A 1D or 2D pool as the 2D geometry a kernel reads: a 1D pool gets a unit height, a format
/// without a batch axis a unit batch. In both layouts these lengths fully describe the
/// contiguous input and output buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolGeometry2d {
    pub n: usize,
    pub c: usize,
    pub ih: usize,
    pub iw: usize,
    pub oh: usize,
    pub ow: usize,
    pub kh: usize,
    pub kw: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    pub pad_h: usize,
    pub pad_w: usize,
    pub dil_h: usize,
    pub dil_w: usize,
    pub channels_last: bool,
}

impl PoolGeometry2d {
    /// The geometry of `pool_spec` from `input_shape` to `output_shape`; `None` for pools over
    /// more than two spatial axes.
    pub fn new(
        pool_spec: &PoolSpec,
        input_shape: &[usize],
        output_shape: &[usize],
    ) -> TractResult<Option<PoolGeometry2d>> {
        let input = pool_spec.data_format.shape(input_shape)?;
        let output = pool_spec.data_format.shape(output_shape)?;
        let rank = input.hw_rank();
        if !(1..=2).contains(&rank) {
            return Ok(None);
        }
        let padding = pool_spec.computed_padding(input.hw_dims());
        let pad_before: TVec<usize> = padding.iter().map(|p| p.pad_before).collect();
        let hw = |unit: usize, v: &[usize]| if rank == 2 { (v[0], v[1]) } else { (unit, v[0]) };
        let (ih, iw) = hw(1, input.hw_dims());
        let (oh, ow) = hw(1, output.hw_dims());
        let (kh, kw) = hw(1, &pool_spec.kernel_shape);
        let (stride_h, stride_w) = hw(1, &pool_spec.strides());
        let (dil_h, dil_w) = hw(1, &pool_spec.dilations());
        let (pad_h, pad_w) = hw(0, &pad_before);
        Ok(Some(PoolGeometry2d {
            n: *input.n().unwrap_or(&1),
            c: *input.c(),
            ih,
            iw,
            oh,
            ow,
            kh,
            kw,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dil_h,
            dil_w,
            channels_last: pool_spec.data_format.c_is_last(),
        }))
    }

    /// Whether the window covers the whole unpadded input in one output position.
    pub fn is_global(&self) -> bool {
        self.oh == 1
            && self.ow == 1
            && self.pad_h == 0
            && self.pad_w == 0
            && self.dil_h == 1
            && self.dil_w == 1
            && self.kh == self.ih
            && self.kw == self.iw
    }
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
