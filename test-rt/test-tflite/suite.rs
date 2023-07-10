use suite_conv::conv_f32::{ConvProblem, ConvProblemParams};

pub fn suite() -> infra::TestSuite {
    let mut onnx = suite_onnx::suite().clone();
    onnx.ignore(&ignore_onnx);
    let mut conv = suite_conv::suite().unwrap().clone();
    conv.ignore(&ignore_conv);
    conv.add_arbitrary::<ConvProblem>(
        "proptest",
        ConvProblemParams { no_group: true, geo_rank: Some(1..3), ..ConvProblemParams::default() },
    );
    infra::TestSuite::default().with("onnx", onnx).with("conv", conv)
}

fn ignore_onnx(t: &[String]) -> bool {
    let name = t.last().unwrap();
    !name.contains("_conv_") || name == "test_conv_with_strides_and_asymmetric_padding"
}

fn ignore_conv(t: &[String]) -> bool {
    let unit: &str = t.last().map(|s| &**s).unwrap();
    t[0] == "q"
        || unit == "proptest"
        // grouping and depthwise
        || unit == "depthwise_0"
        || unit.starts_with("group")
        // conv 3D
        || unit == "lazy_im2col_big"
        || unit == "lazy_im2col_big_2"
        || unit == "batch_3d"
        || unit == "bias_3d_1"
        // nonsense. bug in tfl ? hole in the spec ?
        || unit == "same_1d_1"
}
