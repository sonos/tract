pub fn suite() -> infra::TestSuite {
    let mut onnx = suite_onnx::suite().clone();
    onnx.ignore(&ignore_onnx);
    let mut conv = suite_conv::suite().unwrap().clone();
    conv.ignore(&ignore_conv);
    infra::TestSuite::default().with("onnx", onnx).with("conv", conv)
}

fn ignore_onnx(t: &[String]) -> bool {
    let name = t.last().unwrap();
    !name.contains("_conv_") || name == "test_conv_with_strides_and_asymmetric_padding"
}

fn ignore_conv(t: &[String]) -> bool {
    let unit: &str = t.last().map(|s| &**s).unwrap();
    t[0] == "q" || unit.starts_with("group")
}
