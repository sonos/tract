use suite_unit::conv_f32::{ConvProblem, ConvProblemParams};
use suite_unit::conv_q::{QConvProblem, QConvProblemParams};

#[allow(clippy::needless_update)]
pub fn suite() -> infra::TestSuite {
    let mut onnx = suite_onnx::suite().clone();
    onnx.ignore(&ignore_onnx);
    let mut unit = suite_unit::suite().unwrap().clone();
    unit.ignore(&ignore_conv);
    let cv =
        ConvProblemParams { no_group: true, geo_rank: Some(1..3), ..ConvProblemParams::default() };
    unit.get_sub_mut("conv_f32").add_arbitrary::<ConvProblem>("proptest", cv.clone());
    unit.get_sub_mut("conv_q").add_arbitrary::<QConvProblem>(
        "proptest",
        QConvProblemParams { conv: cv, no_kernel_zero_point: true, .. QConvProblemParams::default() },
    );
    infra::TestSuite::default().with("onnx", onnx).with("unit", unit)
}

fn ignore_onnx(t: &[String]) -> bool {
    let name = t.last().unwrap();
    let included = "_conv_ Conv1d Conv2d squeeze _transpose_ test_reshape test_flatten where less greater equal slice";
    let excluded = "
            test_slice_start_out_of_bounds
            test_Conv1d_groups
            test_Conv2d_groups
            test_Conv1d_depthwise_with_multiplier
            test_Conv2d_depthwise_with_multiplier
            test_Conv2d_groups_thnn
            test_reshape_allowzero_reordered";
    !included.split_whitespace().any(|s| name.contains(s))
        || excluded.split_whitespace().any(|s| s == name)
}

fn ignore_conv(t: &[String]) -> bool {
    let [section, unit] = t else { return false };
    ["deconv"].contains(&&**section)
        // grouping and depthwise
        || unit.starts_with("group")
        // conv 3D
        || unit == "lazy_im2col_big"
        || unit == "lazy_im2col_big_2"
        || unit == "batch_3d"
        || unit == "bias_3d_1"
        // kernel with non 0 zero_point
        || unit == "kernel_zp"
}
