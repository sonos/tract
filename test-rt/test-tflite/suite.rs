use suite_conv::conv_f32::{ConvProblem, ConvProblemParams};
use suite_conv::conv_q::{QConvProblem, QConvProblemParams};

pub fn suite() -> infra::TestSuite {
    let mut onnx = suite_onnx::suite().clone();
    onnx.ignore(&ignore_onnx);
    let mut conv = suite_conv::suite().unwrap().clone();
    conv.ignore(&ignore_conv);
    let cv =
        ConvProblemParams { no_group: true, geo_rank: Some(1..3), ..ConvProblemParams::default() };
    conv.get_sub_mut("f32").add_arbitrary::<ConvProblem>("proptest", cv.clone());
    conv.get_sub_mut("q").add_arbitrary::<QConvProblem>(
        "proptest",
        QConvProblemParams { conv: cv, no_kernel_zp: true },
    );
    infra::TestSuite::default().with("onnx", onnx).with("conv", conv)
}

fn ignore_onnx(t: &[String]) -> bool {
    let name = t.last().unwrap();
    let included = "_conv_ Conv1d Conv2d squeeze _transpose_ test_reshape test_flatten";
    let excluded = "
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
    section == "deconv"
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
