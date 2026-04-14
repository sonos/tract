use crate::internal::*;
use crate::v2::{AxisRegion, PulseV2Region, RegionTransform};
use tract_pulse_opl::tract_core::ops::cnn::Conv;

fn conv_input_regions(
    op: &dyn TypedOp,
    output_region: &PulseV2Region,
) -> TractResult<TVec<Option<PulseV2Region>>> {
    let conv = op.downcast_ref::<Conv>().unwrap();
    let kernel_len = conv.pool_spec.kernel_shape[0];
    let dilation = conv.pool_spec.dilations.as_ref().map_or(1, |d| d[0]);
    let receptive_field = (kernel_len - 1) * dilation;

    // Data input: extend the streaming axis start backward by receptive_field.
    // Conv at output position j needs input [j, j+K). So to produce outputs
    // starting at `start`, the input must start at `start` (same), but the
    // input region must extend K-1 further than the output region. Since the
    // output region == the source increment, the input region's start must
    // go K-1 earlier to cover the receptive field.
    let mut axes = output_region.axes.clone();
    for ax in &mut axes {
        if let AxisRegion::Streaming { start, .. } = ax {
            *start = start.clone() - TDim::Val(receptive_field as i64);
        }
    }
    let data_region = PulseV2Region { axes };

    // Kernel and bias: not streaming.
    Ok(tvec![Some(data_region), None, None])
}

inventory::submit! {
    RegionTransform {
        type_id: std::any::TypeId::of::<Conv>(),
        func: conv_input_regions,
    }
}
