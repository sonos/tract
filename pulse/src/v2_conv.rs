use crate::internal::*;
use crate::v2::{AxisRegion, PulseV2Action, PulseV2Region, PulseV2Symbols, RegionTransform};
use tract_pulse_opl::tract_core::ops::cnn::Conv;

fn conv_input_regions(
    op: &dyn TypedOp,
    source_region: &PulseV2Region,
    _symbols: &PulseV2Symbols,
) -> TractResult<Option<PulseV2Action>> {
    let conv = op.downcast_ref::<Conv>().unwrap();

    // Use PoolSpec's computed_padding to get the kernel field size and padding.
    // We pass a dummy input size of 0 — we only need the kernel geometry, not
    // the output size.
    let dilations = conv.pool_spec.dilations();
    let strides = conv.pool_spec.strides();
    let kernel_shape = &conv.pool_spec.kernel_shape;

    // Data input: for each spatial axis, extend the start backward by
    // (kernel_field - 1) where kernel_field = (K-1)*D + 1.
    // This is the receptive field: how far back the conv needs to look.
    let geo_axes = conv.pool_spec.data_format.h_axis()
        ..conv.pool_spec.data_format.h_axis() + kernel_shape.len();

    let mut axes = source_region.axes.clone();
    for (geo_ix, ax_ix) in geo_axes.enumerate() {
        if let Some(AxisRegion::Streaming { start, .. }) = axes.get_mut(ax_ix) {
            // overlap = (K-1)*D - (S-1): how much the receptive field extends
            // beyond one stride step. Round up to stride for grid alignment.
            let kernel_field = (kernel_shape[geo_ix] - 1) * dilations[geo_ix];
            let s = strides[geo_ix];
            let overlap = kernel_field.saturating_sub(s - 1);
            let lookback = if s > 1 && overlap > 0 { ((overlap + s - 1) / s) * s } else { overlap };
            if lookback > 0 {
                *start = start.clone() - TDim::Val(lookback as i64);
            }
        }
    }
    let data_region = PulseV2Region { axes };

    // Kernel and bias: not streaming.
    Ok(Some(PulseV2Action::InputRegions(tvec![Some(data_region), None, None])))
}

inventory::submit! {
    RegionTransform {
        type_id: std::any::TypeId::of::<Conv>(),
        func: conv_input_regions,
    }
}
