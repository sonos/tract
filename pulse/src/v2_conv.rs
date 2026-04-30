use crate::internal::*;
use crate::v2::{AxisRegion, PulseV2Action, PulseV2Region, PulseV2Symbols, RegionTransform};
use tract_pulse_opl::tract_core::ops::cnn::MaxPool;
use tract_pulse_opl::tract_core::ops::cnn::{Conv, pools::PoolSpec};

/// Compute input regions for any op that has a PoolSpec (Conv, MaxPool, etc.).
/// Extends the streaming axis start backward by the overlap.
fn pool_spec_input_regions(
    pool_spec: &PoolSpec,
    source_region: &PulseV2Region,
    n_inputs: usize,
) -> TractResult<Option<PulseV2Action>> {
    let dilations = pool_spec.dilations();
    let strides = pool_spec.strides();
    let kernel_shape = &pool_spec.kernel_shape;

    let geo_axes =
        pool_spec.data_format.h_axis()..pool_spec.data_format.h_axis() + kernel_shape.len();

    let mut axes = source_region.axes.clone();
    let mut overlap_per_axis = tvec![0usize; source_region.rank()];
    for (geo_ix, ax_ix) in geo_axes.enumerate() {
        if let Some(AxisRegion::Streaming { start, .. }) = axes.get_mut(ax_ix) {
            let kernel_field = (kernel_shape[geo_ix] - 1) * dilations[geo_ix];
            let s = strides[geo_ix];
            let overlap = kernel_field.saturating_sub(s - 1);
            let lookback = if s > 1 && overlap > 0 { ((overlap + s - 1) / s) * s } else { overlap };
            overlap_per_axis[ax_ix] = overlap;
            if lookback > 0 {
                *start = start.clone() - TDim::Val(lookback as i64);
            }
        }
    }
    let data_region = PulseV2Region { axes };

    let mut regions = tvec![Some(data_region)];
    for _ in 1..n_inputs {
        regions.push(None);
    }
    // Pass per-input overlap hints (only for the data input).
    let overlaps = tvec![overlap_per_axis];
    Ok(Some(PulseV2Action::InputRegions(regions, Some(overlaps))))
}

fn conv_input_regions(
    op: &dyn TypedOp,
    source_region: &PulseV2Region,
    _symbols: &PulseV2Symbols,
) -> TractResult<Option<PulseV2Action>> {
    let conv = op.downcast_ref::<Conv>().unwrap();
    // Padding on streaming axes is decomposed into Pad + Conv(valid) by
    // decompose_streaming_padding() before pulsification reaches here.
    pool_spec_input_regions(&conv.pool_spec, source_region, 3)
}

fn maxpool_input_regions(
    op: &dyn TypedOp,
    source_region: &PulseV2Region,
    _symbols: &PulseV2Symbols,
) -> TractResult<Option<PulseV2Action>> {
    let pool = op.downcast_ref::<MaxPool>().unwrap();
    pool_spec_input_regions(&pool.pool_spec, source_region, 1)
}

inventory::submit! {
    RegionTransform {
        type_id: std::any::TypeId::of::<Conv>(),
        func: conv_input_regions,
    }
}

inventory::submit! {
    RegionTransform {
        type_id: std::any::TypeId::of::<MaxPool>(),
        func: maxpool_input_regions,
    }
}
