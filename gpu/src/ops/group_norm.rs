use crate::tensor::{DeviceTensor, DeviceTensorExt};
use tract_core::internal::*;
use tract_core::ops::nn::GroupNorm;

/// Elements each channel-first partial-statistics chunk covers.
const CHUNK: usize = 4096;
const MAX_SPLITS: usize = 64;
/// Positions each channel-last partial-statistics chunk covers, per channel.
const CHANNEL_LAST_CHUNK: usize = 64;
const MAX_CHANNEL_LAST_SPLITS: usize = 1024;

/// Static geometry of a GPU group norm. The input is viewed as `(outer, pre, channels, post)`:
/// axes before `block_start` are `outer`, the channel `axis` splits the rest into `pre` and
/// `post`, all of which are reduced.
///
/// Channel-first (`post > 1`), each of the `outer * groups` groups is reduced in `splits`
/// chunks of its flattened `(pre, channel, post)` range. Channel-last (`axis` is the last axis),
/// each channel is reduced in `splits` chunks of `pre`, so reads stay contiguous across
/// channels. Either way a group ends up with `entries` partial statistics.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct GroupNormParams {
    pub block_start: usize,
    pub axis: usize,
    pub groups: usize,
    pub channel_last: bool,
    pub splits: usize,
    pub entries: usize,
}

/// Runtime view of a group norm input, see `GroupNormParams`.
#[derive(Clone, Copy, Debug)]
pub struct GroupNormView {
    pub outer: usize,
    pub pre: usize,
    pub channels: usize,
    pub post: usize,
    pub groups: usize,
    pub channel_last: bool,
    pub splits: usize,
    pub entries: usize,
}

impl GroupNormParams {
    /// GPU kernels need the channel and reduced axes to form one trailing block, with
    /// concrete dims so the chunking is static.
    pub fn from_core(op: &GroupNorm, fact: &TypedFact) -> Option<GroupNormParams> {
        let rank = fact.rank();
        let block_start = op.reduced_axes.iter().copied().chain([op.axis]).min()?;
        let block: TVec<usize> = (block_start..rank).collect();
        let mut covered: TVec<usize> = op.reduced_axes.iter().copied().chain([op.axis]).collect();
        covered.sort();
        if covered != block {
            return None;
        }
        let dims: TVec<usize> =
            (block_start..rank).map(|ax| fact.shape[ax].to_usize().ok()).collect::<Option<_>>()?;
        let channels = dims[op.axis - block_start];
        let per_group_channels = channels / op.groups;
        let positions = dims.iter().product::<usize>() / channels;
        let channel_last = op.axis == rank - 1 && positions > 1;
        let (splits, entries) = if channel_last {
            let splits = positions.div_ceil(CHANNEL_LAST_CHUNK).clamp(1, MAX_CHANNEL_LAST_SPLITS);
            (splits, splits * per_group_channels)
        } else {
            let splits = (positions * per_group_channels).div_ceil(CHUNK).clamp(1, MAX_SPLITS);
            (splits, splits)
        };
        Some(GroupNormParams {
            block_start,
            axis: op.axis,
            groups: op.groups,
            channel_last,
            splits,
            entries,
        })
    }

    pub fn view(&self, shape: &[usize]) -> GroupNormView {
        GroupNormView {
            outer: shape[..self.block_start].iter().product(),
            pre: shape[self.block_start..self.axis].iter().product(),
            channels: shape[self.axis],
            post: shape[self.axis + 1..].iter().product(),
            groups: self.groups,
            channel_last: self.channel_last,
            splits: self.splits,
            entries: self.entries,
        }
    }
}

/// (input, view, partials): writes `entries` `(count, mean, m2)` triples per group.
pub type DispatchGroupNormStatsFn =
    fn(&DeviceTensor, &GroupNormView, &DeviceTensor) -> TractResult<()>;
/// (input, partials, view, eps, output): merges each group's partials, then normalizes.
/// The merged statistics may overwrite the partials.
pub type DispatchGroupNormApplyFn =
    fn(&DeviceTensor, &DeviceTensor, &GroupNormView, f32, &DeviceTensor) -> TractResult<()>;

/// First half of a GPU group norm: partial statistics, as an F32 `[outer * groups, entries, 3]`
/// tensor consumed by `GpuGroupNormApply`.
#[derive(Clone, Debug)]
pub struct GpuGroupNormStats {
    pub params: GroupNormParams,
    pub backend_name: &'static str,
    pub dispatch: DispatchGroupNormStatsFn,
}

impl PartialEq for GpuGroupNormStats {
    fn eq(&self, other: &Self) -> bool {
        self.backend_name == other.backend_name && self.params == other.params
    }
}

impl Eq for GpuGroupNormStats {}

impl std::hash::Hash for GpuGroupNormStats {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.backend_name.hash(state);
        self.params.hash(state);
    }
}

impl Op for GpuGroupNormStats {
    fn name(&self) -> StaticName {
        format!("{}GroupNormStats", self.backend_name).into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("{:?}", self.params)])
    }

    op_as_typed_op!();
}

impl EvalOp for GpuGroupNormStats {
    op_out_of_plan!();

    fn eval(&self, ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input_value = args_1!(inputs);
        let input = input_value.to_device_tensor()?;
        let view = self.params.view(input.shape());
        let shape = [view.outer * view.groups, view.entries, 3];
        let partials = crate::turn_handler::make_tensor_for_node(ctx, DatumType::F32, &shape)?;
        (self.dispatch)(input, &view, &partials)?;
        Ok(tvec!(partials.into_tensor().into_tvalue()))
    }
}

impl TypedOp for GpuGroupNormStats {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        crate::utils::facts_to_device_facts(inputs, |facts| {
            let outer: TDim = facts[0].shape[..self.params.block_start].iter().product();
            let shape = [outer * self.params.groups, self.params.entries.into(), 3.into()];
            Ok(tvec!(DatumType::F32.fact(shape)))
        })
        .with_context(|| format!("Error while computing facts for {:?}", self.name()))
    }

    as_op!();
}

/// Second half of a GPU group norm: normalizes the input with the merged partials of
/// `GpuGroupNormStats`.
#[derive(Clone, Debug)]
pub struct GpuGroupNormApply {
    pub params: GroupNormParams,
    pub eps: Arc<Tensor>,
    pub backend_name: &'static str,
    pub dispatch: DispatchGroupNormApplyFn,
}

impl PartialEq for GpuGroupNormApply {
    fn eq(&self, other: &Self) -> bool {
        self.backend_name == other.backend_name
            && self.params == other.params
            && self.eps == other.eps
    }
}

impl Eq for GpuGroupNormApply {}

impl std::hash::Hash for GpuGroupNormApply {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.backend_name.hash(state);
        self.params.hash(state);
        self.eps.hash(state);
    }
}

impl Op for GpuGroupNormApply {
    fn name(&self) -> StaticName {
        format!("{}GroupNormApply", self.backend_name).into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("{:?}, eps: {:?}", self.params, self.eps)])
    }

    op_as_typed_op!();
}

impl EvalOp for GpuGroupNormApply {
    op_out_of_plan!();

    fn eval(&self, ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let (input_value, partials_value) = args_2!(inputs);
        let input = input_value.to_device_tensor()?;
        let partials = partials_value.to_device_tensor()?;
        let view = self.params.view(input.shape());
        let output =
            crate::turn_handler::make_tensor_for_node(ctx, input.datum_type(), input.shape())?;
        let eps = self.eps.cast_to_scalar::<f32>()?;
        (self.dispatch)(input, partials, &view, eps, &output)?;
        Ok(tvec!(output.into_tensor().into_tvalue()))
    }
}

impl TypedOp for GpuGroupNormApply {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        crate::utils::facts_to_device_facts(inputs, |facts| {
            Ok(tvec!(facts[0].datum_type.fact(facts[0].shape.clone())))
        })
        .with_context(|| format!("Error while computing facts for {:?}", self.name()))
    }

    as_op!();
}

/// Wires a GPU group norm: `stats` on the input, then `apply` on the input and the partials.
pub fn wire_gpu_group_norm(
    target: &mut TypedModel,
    name: &str,
    input: OutletId,
    stats: GpuGroupNormStats,
    apply: GpuGroupNormApply,
) -> TractResult<TVec<OutletId>> {
    let partials = target.wire_node(format!("{name}.stats"), stats, &[input])?[0];
    target.wire_node(name, apply, &[input, partials])
}
