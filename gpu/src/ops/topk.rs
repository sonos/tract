use crate::tensor::{DeviceTensor, DeviceTensorExt};
use tract_core::internal::*;
use tract_core::ops::array::Topk;

/// Elements one top-k kernel workgroup sorts in shared memory.
pub const TOPK_CHUNK: usize = 2048;
/// Largest `k` [`GpuTopK`] takes: half a chunk, so every level of the reduction at least
/// halves the candidates.
pub const TOPK_MAX_K: usize = TOPK_CHUNK / 2;

/// One backend's top-k kernels, all over rows of contiguous tensors.
///
/// They work on u64 sort keys: an order-preserving u32 image of the value in the high half
/// (inverted when `largest` is false; NaN maps to 1, padding to 0, every number above both)
/// and `u32::MAX - index` in the low half, so a larger key is a better entry and equal
/// values prefer the lower index. Each kernel sorts chunks of [`TOPK_CHUNK`] keys and keeps
/// the best `k` of each, padding a chunk shorter than `k` with zero keys.
#[derive(Clone, Copy)]
pub struct TopKKernels {
    /// `values` `[rows, n]` (f32 or f16) to `keys` `[rows, chunks(n) * k]`.
    pub chunk_values:
        fn(bool, &DeviceTensor, usize, usize, usize, &DeviceTensor) -> TractResult<()>,
    /// `keys` `[rows, m]` to `out` `[rows, chunks(m) * k]`.
    pub chunk_keys: fn(&DeviceTensor, usize, usize, usize, &DeviceTensor) -> TractResult<()>,
    /// Sorted `keys` `[rows, k]` and `values` `[rows, n]` to the `k` values and their i64
    /// indices, both `[rows, k]`.
    #[allow(clippy::type_complexity)]
    pub finish: fn(
        &DeviceTensor,
        &DeviceTensor,
        usize,
        usize,
        usize,
        &DeviceTensor,
        &DeviceTensor,
    ) -> TractResult<()>,
}

fn chunks(len: usize) -> usize {
    len.div_ceil(TOPK_CHUNK)
}

/// Backend-agnostic `core::ops::array::Topk` over the last axis, for f32 and f16 values and
/// a constant `k` up to [`TOPK_MAX_K`]: the same total order as core, ties to the lowest
/// index and NaN after every number. Each row is cut in chunks whose best `k` keys survive,
/// and the survivors are cut again until a single chunk holds them all.
#[derive(Clone)]
pub struct GpuTopK {
    pub largest: bool,
    pub k: usize,
    pub backend_name: &'static str,
    pub kernels: TopKKernels,
}

impl GpuTopK {
    pub fn is_supported_dt(dt: DatumType) -> bool {
        matches!(dt, DatumType::F32 | DatumType::F16)
    }

    /// The GPU form of `op` at `node`, when it sorts the last axis of f32 or f16 values
    /// for a constant `k` within [`TOPK_MAX_K`].
    pub fn from_tract_core(
        source: &TypedModel,
        node: &TypedNode,
        op: &Topk,
        backend_name: &'static str,
        kernels: TopKKernels,
    ) -> TractResult<Option<GpuTopK>> {
        let facts = source.node_input_facts(node.id)?;
        rule_if!(Self::is_supported_dt(facts[0].datum_type));
        rule_if!(op.axis + 1 == facts[0].rank());
        rule_if_some!(k = &facts[1].konst);
        let k = k.cast_to_scalar::<i64>()?;
        rule_if!(k >= 1 && k as usize <= TOPK_MAX_K);
        Ok(Some(GpuTopK { largest: op.largest, k: k as usize, backend_name, kernels }))
    }
}

impl std::fmt::Debug for GpuTopK {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}TopK(k={}, largest={})", self.backend_name, self.k, self.largest)
    }
}

impl PartialEq for GpuTopK {
    fn eq(&self, other: &Self) -> bool {
        self.backend_name == other.backend_name
            && self.k == other.k
            && self.largest == other.largest
    }
}

impl Eq for GpuTopK {}

impl std::hash::Hash for GpuTopK {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.backend_name.hash(state);
        self.k.hash(state);
        self.largest.hash(state);
    }
}

impl Op for GpuTopK {
    fn name(&self) -> StaticName {
        format!("{}TopK", self.backend_name).into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("k: {} largest: {}", self.k, self.largest)])
    }

    op_as_typed_op!();
}

impl EvalOp for GpuTopK {
    op_out_of_plan!();

    fn eval(&self, ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let values = inputs[0].to_device_tensor()?;
        ensure!(values.strides() == &*Tensor::natural_strides(values.shape()));
        let shape = values.shape();
        let n = *shape.last().context("TopK on a scalar")?;
        let rows: usize = shape[..shape.len() - 1].iter().product();
        let k = self.k;
        ensure!(k <= n, "{} takes {k} of a row of {n}", self.name());
        let mut out_shape: TVec<usize> = shape.into();
        *out_shape.last_mut().unwrap() = k;
        let out_values = crate::turn_handler::make_tensor_for_node_output(
            ctx,
            0,
            values.datum_type(),
            &out_shape,
        )?;
        let out_indices = crate::turn_handler::make_tensor_for_node_output(
            ctx,
            1,
            i64::datum_type(),
            &out_shape,
        )?;
        if rows == 0 {
            return Ok(tvec!(
                out_values.into_tensor().into_tvalue(),
                out_indices.into_tensor().into_tvalue()
            ));
        }

        let mut len = chunks(n) * k;
        let mut keys = DeviceTensor::uninitialized_dt(u64::datum_type(), &[rows, len])?;
        (self.kernels.chunk_values)(self.largest, values, rows, n, k, &keys)?;
        while len > k {
            let next_len = chunks(len) * k;
            let next = DeviceTensor::uninitialized_dt(u64::datum_type(), &[rows, next_len])?;
            (self.kernels.chunk_keys)(&keys, rows, len, k, &next)?;
            keys = next;
            len = next_len;
        }
        (self.kernels.finish)(&keys, values, rows, n, k, &out_values, &out_indices)
            .with_context(|| format!("Error while dispatching eval for {}", self.name()))?;
        Ok(tvec!(out_values.into_tensor().into_tvalue(), out_indices.into_tensor().into_tvalue()))
    }
}

impl TypedOp for GpuTopK {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        crate::utils::facts_to_device_facts(inputs, |facts| {
            let mut values = facts[0].without_value();
            let rank = values.rank();
            ensure!(rank >= 1, "{} needs at least one axis", self.name());
            values.shape.set(rank - 1, self.k.to_dim());
            let mut indices = values.clone();
            indices.datum_type = i64::datum_type();
            Ok(tvec!(values, indices))
        })
        .with_context(|| format!("Error while computing facts for {:?}", self.name()))
    }

    as_op!();
}
